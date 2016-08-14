'''
A closed-chain motion planner. 
Modified from ClosedChainPlanner.py developed by Puttichai.
'''

from openravepy import *
import numpy as np
import random
import time
import traceback
import TOPP
from utils.utils import colorize
from utils import utils, heap, lie


################################################################################
#                             Global Parameters
################################################################################
# Planner parameters
FW       = 0
BW       = 1
REACHED  = 0
ADVANCED = 1
TRAPPED  = 2

IK_CHECK_COLLISION = IkFilterOptions.CheckEnvCollisions

RNG = random.SystemRandom()

################################################################################
#                                  Config
################################################################################
class SE3Config(object):
    def __init__(self, q, p, qd=None, pd=None):
        quat_length = np.linalg.norm(q)
        self.q = q / quat_length
        if qd is None:
            self.qd = np.zeros(3)
        else:
            self.qd = qd

        self.p = p
        if pd is None:
            self.pd = np.zeros(3)
        else:
            self.pd = pd

        self.T = matrixFromPose(np.hstack([self.q, self.p]))

    @staticmethod
    def from_matrix(T):
        '''
        Initialize an SE3Config object from a transformation matrix.
        T can be a None, since SE3Config of a goal vertex can be a None.
        '''    
        if T is None:
            return None

        quat = quatFromRotationMatrix(T[0:3, 0:3])
        p = T[0:3, 3]
        return SE3Config(quat, p)        


class CCTrajectory(object):
    def __init__(self, lie_traj, translation_traj, bimanual_wpts, timestamps):
        self.lie_traj      = lie_traj
        self.translation_traj    = translation_traj
        self.bimanual_wpts = bimanual_wpts
        self.timestamps    = timestamps[:]
            

class CCConfig(object):
    def __init__(self, q_robots, SE3_config):
        self.q_robots   = q_robots
        self.SE3_config = SE3_config


class CCVertex(object):  
    def __init__(self, config):
        self.config = config

        # These parameters are to be assigned when the vertex is added to the tree
        self.index            = 0
        self.parent_index     = None
        self.rot_traj         = None # TOPP trajectory
        self.translation_traj = None # TOPP trajectory
        self.bimanual_wpts    = []
        self.timestamps       = []
        self.level            = 0


class CCTree(object):  
    def __init__(self, vroot=None, treetype=FW):
        self.vertices = []
        self.length = 0
        if vroot is not None:
            self.vertices.append(vroot)
            self.length += 1

        self.treetype = treetype

    def __len__(self):
        return len(self.vertices)

    def __getitem__(self, index):
        return self.vertices[index]        
    
    def add_vertex(self, vnew, parent_index, rot_traj, translation_traj, bimanual_wpts, timestamps):
        vnew.parent_index     = parent_index
        vnew.rot_traj         = rot_traj
        vnew.translation_traj = translation_traj
        vnew.bimanual_wpts    = bimanual_wpts
        vnew.timestamps       = timestamps
        vnew.index            = self.length
        vnew.level            = self.vertices[parent_index].level + 1
        
        self.vertices.append(vnew)
        self.length += 1
        

    def generate_rot_traj_list(self, startindex=-1):
        rot_traj_list = []

        vertex = self.vertices[startindex]
        while (vertex.parent_index is not None):
            parent = self.vertices[vertex.parent_index]
            rot_traj_list.append(vertex.rot_traj)
            vertex = parent

        if (self.treetype == FW):
            rot_traj_list.reverse()

        return rot_traj_list


    def generate_rot_mat_list(self, startindex=-1):
        rot_mat_list = []

        vertex = self.vertices[startindex]
        while (vertex.parent_index is not None):
            parent = self.vertices[vertex.parent_index]
            rot_mat_list.append(vertex.config.SE3_config.T[0:3, 0:3])
            vertex = parent
        rot_mat_list.append(self.vertices[0].config.SE3_config.T[0:3, 0:3])

        if (self.treetype == FW):
            rot_mat_list.reverse()

        return rot_mat_list


    def generate_translation_traj_list(self, startindex=-1):
        translation_traj_list = []
            
        vertex = self.vertices[startindex]
        while (vertex.parent_index is not None):
            parent = self.vertices[vertex.parent_index]
            translation_traj_list.append(vertex.translation_traj)
            vertex = parent

        if (self.treetype == FW):
            translation_traj_list.reverse()

        return translation_traj_list


    def generate_bimanual_wpts_list(self, startindex=-1):
        bimanual_wpts_list = [[], []]
            
        vertex = self.vertices[startindex]
        while (vertex.parent_index is not None):
            parent = self.vertices[vertex.parent_index]
            for i in xrange(2):
                bimanual_wpts_list[i].append(vertex.bimanual_wpts[i])
            vertex = parent

        if (self.treetype == FW):
            for i in xrange(2):
                bimanual_wpts_list[i].reverse()
                
        return bimanual_wpts_list


    def generate_timestamps_list(self, startindex=-1):
        timestamps_list = []
        
        vertex = self.vertices[startindex]
        while (vertex.parent_index is not None):
            parent = self.vertices[vertex.parent_index]
            timestamps_list.append(vertex.timestamps)
            vertex = parent

        if (self.treetype == FW):
            timestamps_list.reverse()

        return timestamps_list            


class CCQuery(object):
    '''
    Class Query stores everything related to a single query.
    '''
    def __init__(self, obj_translation_limits, q_robots_start, q_robots_goal,
                T_obj_start, T_obj_goal=None, nn=-1, step_size=0.7,
                interpolation_duration=0.5, discr_timestep=1e-2):
        # Initialize v_start and v_goal
        SE3_config_start = SE3Config.from_matrix(T_obj_start)
        SE3_config_goal  = SE3Config.from_matrix(T_obj_goal)
        cc_config_start  = CCConfig(q_robots_start, SE3_config_start)
        cc_config_goal   = CCConfig(q_robots_goal, SE3_config_goal)
        self.v_start     = CCVertex(cc_config_start)
        self.v_goal      = CCVertex(cc_config_goal)

        # Initialize RRTs
        self.treestart              = CCTree(self.v_start, FW)
        self.treeend                = None # to be initialized when being solved (after grasping pose check is passed)
        self.nn                     = nn
        self.step_size              = step_size # for tree extension
        self.interpolation_duration = interpolation_duration
        self.discr_timestep         = discr_timestep # for collision checking

        # traj information
        self.connecting_rot_traj         = None
        self.connecting_translation_traj = None
        self.connecting_bimanual_wpts    = None
        self.connecting_timestamps       = None
        self.rot_traj_list               = None
        self.rot_mat_list                = None
        self.lie_traj                    = None
        self.translation_traj_list       = None
        self.translation_traj            = None
        self.timestamps                  = None
        self.bimanual_wpts               = None

        # Statistics
        self.running_time    = 0.0
        self.iteration_count = 0
        self.solved          = False
        
        # Parameters    
        self.upper_limits = obj_translation_limits[0]
        self.lower_limits = obj_translation_limits[1]
        
    def generate_final_lie_traj(self):
        if (not self.solved):
            raise CCPlannerException('Query not solved.')
            return

        # Generate rot_traj_list
        self.rot_traj_list = self.treestart.generate_rot_traj_list()
        if (self.connecting_rot_traj is not None):
            self.rot_traj_list.append(self.connecting_rot_traj)
        self.rot_traj_list += self.treeend.generate_rot_traj_list()

        # Generate rot_mat_list
        self.rot_mat_list = self.treestart.generate_rot_mat_list()
        self.rot_mat_list += self.treeend.generate_rot_mat_list()

        # Combine rot_traj_list and rot_mat_list to generate lie_traj
        self.lie_traj = lie.LieTraj(self.rot_mat_list, self.rot_traj_list)

    def generate_final_translation_traj(self):
        if (not self.solved):
            raise CCPlannerException('Query not solved.')
            return

        # Generate translation_traj_list
        self.translation_traj_list = self.treestart.generate_translation_traj_list()
        if (self.connecting_translation_traj is not None):
            self.translation_traj_list.append(self.connecting_translation_traj)
        self.translation_traj_list += self.treeend.generate_translation_traj_list()

        # Convert translation_traj_list to translation_traj
        self.translation_traj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(utils.traj_str_from_traj_list(self.translation_traj_list))

    def generate_final_bimanual_wpts(self):
        if (not self.solved):
            raise CCPlannerException('Query not solved.')
            return

        bimanual_wpts_list = self.treestart.generate_bimanual_wpts_list()  
        if (self.connecting_bimanual_wpts is not None):
            for i in xrange(2):
                bimanual_wpts_list[i].append(self.connecting_bimanual_wpts[i])        
        bimanual_wpts_list_bw = self.treeend.generate_bimanual_wpts_list()
        for i in xrange(2):
            bimanual_wpts_list[i] += bimanual_wpts_list_bw[i]

        left_wpts = utils.merge_wpts_list(bimanual_wpts_list[0])
        right_wpts = utils.merge_wpts_list(bimanual_wpts_list[1])
        self.bimanual_wpts = [left_wpts, right_wpts]

    def generate_final_timestamps(self):
        if (not self.solved):
            raise CCPlannerException('Query not solved.')
            return

        timestamps_list = self.treestart.generate_timestamps_list()
        if (self.connecting_timestamps is not None):
            timestamps_list.append(self.connecting_timestamps)
        timestamps_list += self.treeend.generate_timestamps_list()

        self.timestamps = utils.merge_timestamps_list(timestamps_list)

    def generate_final_cctraj(self):
        if (not self.solved):
            raise CCPlannerException('Query not solved.')
            return

        # Generate CCTrajectory components
        self.generate_final_lie_traj()
        self.generate_final_translation_traj()
        self.generate_final_timestamps()
        self.generate_final_bimanual_wpts()
        
        self.cctraj = CCTrajectory(self.lie_traj, self.translation_traj, self.bimanual_wpts, self.timestamps)


class CCPlanner(object):
    '''
    Requirements:
    - two identical robots
    '''
    
    def __init__(self, manip_obj, robots, debug=False):
        self.obj = manip_obj
        self.robots = robots
        self.manips = []
        self._debug = debug
        for (i, robot) in enumerate(self.robots):
            self.manips.append(robot.GetActiveManipulator())
            robot.SetActiveDOFs(self.manips[i].GetArmIndices())

        self.bimanual_obj_tracker = BimanualObjectTracker(self.robots, manip_obj, debug=self._debug)

        self._active_dofs = self.manips[0].GetArmIndices()
        self._vmax = self.robots[0].GetDOFVelocityLimits()[self._active_dofs]
        self._amax = self.robots[0].GetDOFAccelerationLimits()[self._active_dofs]

        self.env = self.obj.GetEnv()

    def sample_SE3_config(self):
        '''
        sample_SE3_config randomly samples an object transformation.
        This function does not do any feasibility checking since when
        extending a vertex on a tree to this config, we do not use
        this config directly.
        ''' 
        q_rand = lie.RandomQuat()
        p_rand = np.asarray([RNG.uniform(self._query.lower_limits[i], 
                                         self._query.upper_limits[i]) 
                             for i in xrange(3)])
        
        qd_rand = (1e-3) * np.ones(3)
        pd_rand = np.zeros(3)

        return SE3Config(q_rand, p_rand, qd_rand, pd_rand)
    
    def _check_grasping_pose(self):
        '''
        Check if the start and goal grasping pose matches and complete 
        treeend in self._query if check passed. After being called, the 
        planner will have attribute bimanual_T_rel to store grasping pose.
        '''
        # Compute relative transformation from end-effectors to object
        self.bimanual_T_rel = []
        for i in xrange(2):
            self.bimanual_T_rel.append(np.dot(np.linalg.inv(self._query.v_start.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], self._query.v_start.config.q_robots[i])))

        # Compute object SE3_config at goal if not specified
        if self._query.v_goal.config.SE3_config is None:
            T_left_robot_goal = utils.compute_endeffector_transform(self.manips[0], self._query.v_goal.config.q_robots[0])
            T_obj_goal = np.dot(T_left_robot_goal, np.linalg.inv(self.bimanual_T_rel[0]))
            self._query.v_goal.config.SE3_config = SE3Config.from_matrix(T_obj_goal)

        # Check start and goal grasping pose
        bimanual_goal_rel_T = []
        for i in xrange(2):
            bimanual_goal_rel_T.append(np.dot(np.linalg.inv(self._query.v_goal.config.SE3_config.T), utils.compute_endeffector_transform(self.manips[i], self._query.v_goal.config.q_robots[i])))

        if not np.isclose(self.bimanual_T_rel, bimanual_goal_rel_T, atol=1e-3).all():
            raise CCPlannerException('Start and goal grasping pose not matching.')

        # Complete treeend in the query 
        self._query.treeend = CCTree(self._query.v_goal, BW)


    def solve(self, query, timeout=10):
        self._query = query
        if self._query.solved:
            self._output_info('This query has already been solved.', 'green')
            return True

        self._check_grasping_pose()

        t = 0.0
        prev_iter = self._query.iteration_count
        
        t_begin = time.time()
        if (self._connect() == REACHED):
            self._query.iteration_count += 1
            t_end = time.time()
            self._query.running_time += (t_end - t_begin)
            
            self._output_info('Path found', 'green')
            self._output_info('Total number of iterations : {0}'.format(self._query.iteration_count), 'green')
            self._output_info('Total running time : {0} sec.'.format(self._query.running_time), 'green')
            self._query.solved = True
            self._query.generate_final_cctraj()
            return True

        elasped_time = time.time() - t_begin
        t += elasped_time
        self._query.running_time += elasped_time

        while (t < timeout):
            self._query.iteration_count += 1
            self._output_debug('Iteration : {0}'.format(self._query.iteration_count), 'blue')
            t_begin = time.time()

            SE3_config = self.sample_SE3_config()
            if (self._extend(SE3_config) != TRAPPED):
                self._output_debug('Tree start : {0}; Tree end : {1}'.format(len(self._query.treestart.vertices), len(self._query.treeend.vertices)), 'green')

                if (self._connect() == REACHED):
                    t_end = time.time()
                    self._query.running_time += (t_end - t_begin)
                    self._output_info('Path found', 'green')
                    self._output_info('Total number of iterations : {0}'.format(self._query.iteration_count), 'green')
                    self._output_info('Total running time : {0} sec.'.format(self._query.running_time), 'green')
                    self._query.solved = True
                    self._query.generate_final_cctraj()
                    return True
                
            elasped_time = time.time() - t_begin
            t += elasped_time
            self._query.running_time += elasped_time

        self._output_info('Timeout {0}s reached after {1} iterations'.format(timeout, self._query.iteration_count - prev_iter), 'red')
        return False


    def _extend(self, SE3_config):
        if True:#(np.mod(self._query.iteration_count - 1, 2) == FW):
            return self._extend_fw(SE3_config)
        else:
            return self._extend_bw(SE3_config)


    def _extend_fw(self, SE3_config):
        status = TRAPPED
        nnindices = self._nearest_neighbor_indices(SE3_config, FW)
        for index in nnindices:
            v_near = self._query.treestart[index]
            
            # quaternion
            q_beg  = v_near.config.SE3_config.q
            qd_beg = v_near.config.SE3_config.qd
            
            # translation
            p_beg  = v_near.config.SE3_config.p
            pd_beg = v_near.config.SE3_config.pd

            # Check if SE3_config is too far from v_near.SE3_config
            SE3_dist = utils.SE3_distance(SE3_config.T, v_near.config.SE3_config.T, 1.0 / np.pi, 1.0)
            if SE3_dist <= self._query.step_size:
                q_end = SE3_config.q
                p_end = SE3_config.p
                status = REACHED
            else:
                q_end = q_beg + self._query.step_size * (SE3_config.q - q_beg) / np.sqrt(SE3_dist)
                q_end /= np.linalg.norm(q_end)
                p_end = p_beg + self._query.step_size * (SE3_config.p - p_beg) / np.sqrt(SE3_dist)
                status = ADVANCED

            qd_end = SE3_config.qd
            pd_end = SE3_config.pd

            new_SE3_config = SE3Config(q_end, p_end, qd_end, pd_end)

            # Check collision (SE3_config)
            if not self.is_collision_free_SE3_config(new_SE3_config):
                self._output_debug('TRAPPED : SE(3) config in collision', bold=False)
                status = TRAPPED
                continue

            # Check reachability (SE3_config)
            if not self.check_SE3_config_reachability(new_SE3_config):
                self._output_debug('TRAPPED : SE(3) config not reachable', bold=False)
                status = TRAPPED
                continue
            
            # Interpolate a SE3 trajectory for the object
            R_beg = rotationMatrixFromQuat(q_beg)
            R_end = rotationMatrixFromQuat(q_end)
            rot_traj = lie.InterpolateSO3(R_beg,
                                         rotationMatrixFromQuat(q_end),
                                         qd_beg, qd_end, 
                                         self._query.interpolation_duration)
            translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_end, pd_end, self._query.interpolation_duration)

            # Check translational limit
            # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
            if not utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str):
                self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
                status = TRAPPED
                continue

            translation_traj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(translation_traj_str)

            # Check collision (object trajectory)
            if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg):
                self._output_debug('TRAPPED : SE(3) trajectory in collision', bold=False)
                status = TRAPPED
                continue
            
            # Check reachability (object trajectory)
            passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(rot_traj, translation_traj, [R_beg, R_end], v_near.config.q_robots)
            if not passed:
                self._output_debug('TRAPPED : SE(3) trajectory not reachable', bold=False)
                status = TRAPPED
                continue

            # Now this trajectory is alright.
            self._output_debug('Successful : SE(3) trajectory generated', color='green', bold=False)
            new_q_robots = [wpts[-1][0:6] for wpts in bimanual_wpts] 
            new_config = CCConfig(new_q_robots, new_SE3_config)
            new_vertex = CCVertex(new_config)
            self._query.treestart.add_vertex(new_vertex, v_near.index, rot_traj, translation_traj, bimanual_wpts, timestamps)
            return status
        return status
            

    def _extend_bw(self, SE3_config):
        status = TRAPPED
        return status


    def _connect(self):
        if True:#(np.mod(self._query.iteration_count - 1, 2) == FW):
            # Treestart has just been extended
            return self._connect_fw()
        else:
            # Treeend has just been extended
            return self._connect_bw()


    def _connect_fw(self):
        '''
        _connect_fw tries to connect the newly added vertex on treestart
        (v_test) to other vertices on treeend (v_near).
        '''
        v_test = self._query.treestart.vertices[-1]
        nnindices = self._nearest_neighbor_indices(v_test.config.SE3_config, BW)
        status = TRAPPED
        for index in nnindices:
            v_near = self._query.treeend[index]
            
            # quaternion
            q_beg  = v_test.config.SE3_config.q
            qd_beg = v_test.config.SE3_config.qd
            
            q_end  = v_near.config.SE3_config.q
            qd_end = v_near.config.SE3_config.qd
            
            # translation
            p_beg  = v_test.config.SE3_config.p
            pd_beg = v_test.config.SE3_config.pd

            p_end  = v_near.config.SE3_config.p
            pd_end = v_near.config.SE3_config.pd
            
            # Interpolate the object trajectory
            R_beg = rotationMatrixFromQuat(q_beg)
            R_end = rotationMatrixFromQuat(q_end)
            rot_traj = lie.InterpolateSO3(R_beg,
                                         rotationMatrixFromQuat(q_end),
                                         qd_beg, qd_end, 
                                         self._query.interpolation_duration)
            translation_traj_str = utils.traj_str_3rd_degree(p_beg, p_end, pd_end, pd_end, self._query.interpolation_duration)

            # Check translational limit
            # NB: Can skip this step, since it's not likely the traj will exceed the limits given that p_beg and p_end are within limits
            if not utils.check_translation_traj_str_limits(self._query.upper_limits, self._query.lower_limits, translation_traj_str):
                self._output_debug('TRAPPED : SE(3) trajectory exceeds translational limit', bold=False)
                continue

            translation_traj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(translation_traj_str)

            # Check collision (object trajectory)
            if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, R_beg):
                self._output_debug('TRAPPED : SE(3) trajectory in collision', bold=False)
                continue
            
            # Check reachability (object trajectory)
            passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(rot_traj, translation_traj, [R_beg, R_end], v_test.config.q_robots)
            if not passed:
                self._output_debug('TRAPPED : SE(3) trajectory not reachable', bold=False)
                continue

            # Check similarity of terminal IK solutions
            eps = 1e-3
            for i in xrange(2):
                self._output_debug('{0}'.format(v_near.config.q_robots[i]), bold=False)
                self._output_debug('{0}'.format(bimanual_wpts[i][-1][0:6]), bold=False)
                passed = utils.distance(v_near.config.q_robots[i], bimanual_wpts[i][-1][0:6]) < eps
                if not passed:
                    break
            if not passed:
                self._output_debug('TRAPPED : IK solution discrepancy (robot {0})'.format(i), bold=False)
                continue

            # Now the connection is successful
            self._query.treeend.vertices.append(v_near)
            self._query.connecting_rot_traj         = rot_traj
            self._query.connecting_translation_traj = translation_traj
            self._query.connecting_bimanual_wpts    = bimanual_wpts
            self._query.connecting_timestamps       = timestamps
            status = REACHED
            return status
        return status        


    def _connect_bw(self):
        status = TRAPPED
        return status

    
    def is_collision_free_SE3_config(self, SE3_config):
        self._enable_robots_collision(False)
        self.obj.SetTransform(SE3_config.T)
        is_free = not self.env.CheckCollision(self.obj)
        self._enable_robots_collision()

        return is_free
    
    def is_collision_free_SE3_traj(self, rot_traj, translation_traj, R_beg):
        T = np.eye(4)
        with self.env:
            self._enable_robots_collision(False)

            for t in np.append(np.arange(0, translation_traj.duration, self._query.discr_timestep), translation_traj.duration):
                T[0:3, 0:3] = lie.EvalRotation(R_beg, rot_traj, t)
                T[0:3, 3] = translation_traj.Eval(t)

                self.obj.SetTransform(T)
                in_collision = self.env.CheckCollision(self.obj)
                if in_collision:
                    self._enable_robots_collision()
                    return False
            
            self._enable_robots_collision()

        return True        

    
    def check_SE3_config_reachability(self, SE3_config):
        '''
        check_SE3_config_reachability checks whether both robots can
        grasp the object (at SE3_config.T).
        '''
        with self.env:
            self.obj.SetTransform(SE3_config.T)      
            self._enable_robots_collision(True)

            for i in xrange(2):
                T_gripper = np.dot(SE3_config.T, self.bimanual_T_rel[i])
                sol = self.manips[i].FindIKSolution(T_gripper, IK_CHECK_COLLISION)
                if sol is None:
                    self._enable_robots_collision()
                    return False

            self._enable_robots_collision()

        return True


    def check_SE3_traj_reachability(self, rot_traj, translation_traj, rot_mat_list, ref_sols):
        '''
        check_SE3_traj_reachability checks whether the two robots can
        follow the se3 traj. This function returns status, bimanual_wpts, and timestamps.
        '''
        lie_traj = lie.LieTraj(rot_mat_list, [rot_traj])

        passed, bimanual_wpts, timestamps = self.bimanual_obj_tracker.plan(lie_traj, translation_traj, self.bimanual_T_rel, ref_sols, timestep=self._query.discr_timestep)
        if not passed:
            return False, [], []

        return True, bimanual_wpts, timestamps

    def _nearest_neighbor_indices(self, SE3_config, treetype):
        '''
        _nearest_neighbor_indices returns indices of self.nn nearest
        neighbors of SE3_config on the tree specified by treetype.
        '''
        if (treetype == FW):
            tree = self._query.treestart
        else:
            tree = self._query.treeend
        nv = len(tree)
            
        distance_list = [utils.SE3_distance(SE3_config.T, v.config.SE3_config.T, 1.0 / np.pi, 1.0) 
                        for v in tree.vertices]
        distance_heap = heap.Heap(distance_list)
                
        if (self._query.nn == -1):
            # to consider all vertices in the tree as nearest neighbors
            nn = nv
        else:
            nn = min(self._query.nn, nv)
        nnindices = [distance_heap.ExtractMin()[0] for i in range(nn)]
        return nnindices


    def visualize_cctraj(self, cctraj, speed=1.0):
        timestamps = cctraj.timestamps
        lie_traj   = cctraj.lie_traj
        translation_traj = cctraj.translation_traj
        left_wpts  = cctraj.bimanual_wpts[0]
        right_wpts = cctraj.bimanual_wpts[1]

        sampling_step = timestamps[1] - timestamps[0]
        refresh_step  = sampling_step / speed

        T_obj = np.eye(4)
        for (q_left, q_right, t) in zip(left_wpts, right_wpts, timestamps):
            T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
            T_obj[0:3, 3] = translation_traj.Eval(t)
            self.obj.SetTransform(T_obj)
            self.robots[0].SetActiveDOFValues(q_left)
            self.robots[1].SetActiveDOFValues(q_right)
            time.sleep(refresh_step)

            
    def shortcut(self, query, maxiter=50):
        '''    
        Shortcut query.cctraj and replace it with the new one.
        '''
        # Shortcutting parameters
        min_shortcut_duration = 0.1
        min_n_timesteps = int(min_shortcut_duration / query.discr_timestep)

        # Statistics
        in_collision_count   = 0
        not_reachable_count  = 0
        not_continuous_count = 0
        not_shorter_count    = 0
        successful_count     = 0

        duration = query.cctraj.lie_traj.duration

        new_lie_traj         = query.cctraj.lie_traj
        new_translation_traj = query.cctraj.translation_traj
        new_timestamps       = query.cctraj.timestamps[:]
        new_left_wpts        = query.cctraj.bimanual_wpts[0][:]
        new_right_wpts       = query.cctraj.bimanual_wpts[1][:]

        # Create an accumulated distance list
        accumulated_dist = utils.generate_accumulated_SE3_dist_list(new_lie_traj, new_translation_traj, query.discr_timestep)        

        for i in xrange(maxiter):
            if (duration < min_shortcut_duration):
                self._output_info('Trajectory duration shorter than minimum shortcut duration.', 'yellow')
                break
            
            self._output_debug('Iteration {0}'.format(i + 1), 'blue')

            # Sample two time instants
            timestamps_indices = range(len(new_timestamps))
            t0_index = RNG.choice(timestamps_indices[:-min_n_timesteps])
            t1_index = RNG.choice(timestamps_indices[t0_index + min_n_timesteps:])
            t0 = new_timestamps[t0_index]
            t1 = new_timestamps[t1_index]

            self._output_debug('t0_index = {0}, t0 = {1}'.format(t0_index, t0), bold=False)
            self._output_debug('t1_index = {0}, t1 = {1}'.format(t1_index, t1), bold=False)

            # Interpolate a new SE(3) trajectory segment
            R0 = new_lie_traj.EvalRotation(t0)
            R1 = new_lie_traj.EvalRotation(t1)
            rot_traj = lie.InterpolateSO3(R0, R1,
                                         new_lie_traj.EvalOmega(t0),
                                         new_lie_traj.EvalOmega(t1),
                                         query.interpolation_duration)

            translation_traj_str = utils.traj_str_3rd_degree(new_translation_traj.Eval(t0), new_translation_traj.Eval(t1),
             new_translation_traj.Evald(t0), new_translation_traj.Evald(t1), query.interpolation_duration)
            translation_traj = TOPP.Trajectory.PiecewisePolynomialTrajectory.FromString(translation_traj_str)

            # Check collision (object trajectory)
            if not self.is_collision_free_SE3_traj(rot_traj, translation_traj, R0):          
                in_collision_count += 1
                continue

            # Check reachability (object trajectory)
            passed, bimanual_wpts, timestamps = self.check_SE3_traj_reachability(rot_traj, translation_traj, [R0, R1], [new_left_wpts[t0_index], new_right_wpts[t0_index]])

            if not passed:
                not_reachable_count += 1
                continue

            # Check continuity between newly generated bimanual_wpts and original one
            eps = 1e-3
            if not (utils.distance(bimanual_wpts[0][0], new_left_wpts[t0_index]) < eps and 
                    utils.distance(bimanual_wpts[1][0], new_right_wpts[t0_index]) < eps):
                not_continuous_count += 1
                continue

            # Check SE(3) trajectory length      
            accumulated_dist = utils.generate_accumulated_SE3_dist_list(new_lie_traj, new_translation_traj, query.discr_timestep)

            rot_mat_list = [R0, R1]
            rot_trajslist = [rot_traj]
            lie_traj = lie.LieTraj(rot_mat_list, rot_trajslist)
            new_accumulated_dist = utils.generate_accumulated_SE3_dist_list(lie_traj, translation_traj, query.discr_timestep)

            if new_accumulated_dist[-1] >= (accumulated_dist[t1_index] - accumulated_dist[t0_index]):
                not_shorter_count += 1
                continue

            # Now the new trajectory passes all tests
            # Replace all the old trajectory segments with the new ones

            new_lie_traj         = utils.replace_lie_traj_segment(new_lie_traj, lie_traj.trajlist[0], t0, t1)            
            new_translation_traj = utils.replace_traj_segment(new_translation_traj, translation_traj, t0, t1)

            first_timestamp_chunk = new_timestamps[:t0_index + 1]
            last_timestamp_chunk_offset = new_timestamps[t1_index]
            last_timestamp_chunk = [t - last_timestamp_chunk_offset for t in new_timestamps[t1_index:]]

            new_timestamps = utils.merge_timestamps_list([first_timestamp_chunk, timestamps, last_timestamp_chunk])
            new_left_wpts  = utils.merge_wpts_list([new_left_wpts[:t0_index + 1], bimanual_wpts[0], new_left_wpts[t1_index:]])            
            new_right_wpts = utils.merge_wpts_list([new_right_wpts[:t0_index + 1], bimanual_wpts[1], new_right_wpts[t1_index:]])
            
            self._output_info('Shortcutting successful.', 'green')
            successful_count += 1

        self._output_debug('successful_count = {0}, in_collision_count = {1}, not_shorter_count = {2}, not_reachable_count = {3}, not_continuous_count = {4}'.format(successful_count, in_collision_count, not_shorter_count, not_reachable_count, not_continuous_count), 'yellow')

        query.cctraj = CCTrajectory(new_lie_traj, new_translation_traj, [new_left_wpts, new_right_wpts], new_timestamps)
        
    def _enable_robots_collision(self, enable=True):
        for robot in self.robots:
            robot.Enable(enable)

    def _output_debug(self, msg, color=None, bold=True):
        if self._debug:
            if color is None:
                formatted_msg = msg
            else:
                formatted_msg = colorize(msg, color, bold)
            func_name = traceback.extract_stack(None, 2)[0][2]
            print '[CCPlanner::' + func_name + '] ' + formatted_msg

    def _output_info(self, msg, color=None, bold=True):
        if color is None:
            formatted_msg = msg
        else:
            formatted_msg = colorize(msg, color, bold)
        func_name = traceback.extract_stack(None, 2)[0][2]
        print '[CCPlanner::' + func_name + '] ' + formatted_msg

################################################################################
#                               Object Tracker
################################################################################
class BimanualObjectTracker(object):
    '''
    Class of object tracker for both two robots in a bimanual set-up.

    NB: Require two identical robots.
    TODO: Make it general for mutiple robots of different types.
    '''
    
    def __init__(self, robots, obj, ndof=6, debug=False):
        self.robots = robots
        self.manips = [robot.GetActiveManipulator() for robot in robots]
        self.obj    = obj
        self.env    = obj.GetEnv()
        self.ndof   = ndof

        self._debug   = debug
        self._nrobots = len(robots)
        self._vmax    = robots[0].GetDOFVelocityLimits()[0:self.ndof]
        self._jmax    = robots[0].GetDOFLimits()[1][0:self.ndof]
        self._maxiter = 10
        self._weight  = 10.
        self._tol     = 0.5e-3
        self._gain    = 10.
        self._dt      = 0.1 # time step for ik solver    
        
    def plan(self, lie_traj, translation_traj, bimanual_T_rel, q_robots_init, timestep=0.01):
        '''
        Plan a trajectory for the two robots to follow the object
        trajectory specified by lie_traj and translation_traj.

        Parameters
        ----------
        lie_traj
        translation_traj
        bimanual_T_rel: list
            Relative transformations from end-effectors of the two robots to object
        q_robots_init : list
            Initial configurations of the two robots
        timestep : float, optional
            Time resolution for tracking.

        Returns
        -------
        plannerstatus : bool
        bimanual_wpts : list
            A waypoints list containing vectors q
        timestamps : list
            A list containing timestamps of wpts

        NB: Waypoints returned only contains q info; qd is not involved
        '''
        T_obj = np.eye(4)
        T_obj[0:3, 0:3] = lie_traj.EvalRotation(0)
        T_obj[0:3, 3] = translation_traj.Eval(0)
        self.obj.SetTransform(T_obj)
        duration = lie_traj.duration

        bimanual_T_gripper = []
        for i in xrange(self._nrobots):
            bimanual_T_gripper.append(np.dot(T_obj, bimanual_T_rel[i]))
        
        # Trajectory tracking loop
        bimanual_wpts = [[], []] 
        timestamps = []    

        for i in xrange(self._nrobots):
            bimanual_wpts[i].append(q_robots_init[i])
        timestamps.append(0.)
        
        self._jd_max = self._vmax * timestep
        q_robots_prev = q_robots_init
        for t in np.append(np.arange(timestep, duration, timestep), duration):
            T_obj[0:3, 0:3] = lie_traj.EvalRotation(t)
            T_obj[0:3, 3] = translation_traj.Eval(t)
            
            bimanual_T_gripper = []
            for i in xrange(self._nrobots):
                bimanual_T_gripper.append(np.dot(T_obj, bimanual_T_rel[i]))
            
            q_robots_next = []
            for i in xrange(self._nrobots):
                q_sol = self._compute_IK(i, bimanual_T_gripper[i], q_robots_prev[i])
                if q_sol is None:
                    return False, [], []
                q_robots_next.append(q_sol)

            # Check feasibility
            if not self._is_feasible_bimanual_config(q_robots_next, q_robots_prev, T_obj):
                return False, [], []

            for i in xrange(self._nrobots):
                bimanual_wpts[i].append(q_robots_next[i])
            timestamps.append(t)
            q_robots_prev = q_robots_next
        
        return True, bimanual_wpts, timestamps

    def _is_feasible_bimanual_config(self, q_robots, q_robots_prev, T_obj):
        # Check robot DOF position and velocity limits
        for i in xrange(self._nrobots):
            for j in xrange(self.ndof):
                if abs(q_robots[i][j]) > self._jmax[i] or abs(q_robots[i][j] - q_robots_prev[i][j]) > self._jd_max[j]:
                    return False

        # Update environment for collision checking
        self.obj.SetTransform(T_obj)
        for i in xrange(self._nrobots):
            self.robots[i].SetActiveDOFValues(q_robots[i])

        # Check collision
        for robot in self.robots:
            if self.env.CheckCollision(robot) or robot.CheckSelfCollision():
                return False

        return True

    def _compute_IK(self, robot_index, T, q):
        '''
        Return an IK solution for a robot reaching a end-effector 
        transformation T using differential IK.
        q is initial joint configuration.

        target_pose is a 7-vector, where the first 4 elements are from
        the quarternion of the rotation and the other 3 are from the
        translation vector.
        This implementation follows Stephane's pymanoid library.
        '''
        R = T[0:3, 0:3]
        p = T[0:3, 3]
        target_pose = np.hstack([quatFromRotationMatrix(R), p])
        if target_pose[0] < 0:
            target_pose[0:4] *= -1.

        cur_objective = 5000. # some arbitrary number
        i = 0 # iteration counter
        reached = False
        while i < self._maxiter:
            prev_objective = cur_objective
            cur_objective = self._compute_objective(robot_index, target_pose, q)
            if abs(cur_objective - prev_objective) < self._tol and cur_objective < self._tol:
                # Local minimum reached
                reached = True
                break

            i += 1
            q_delta = self._compute_q_delta(robot_index, target_pose, q)
            
            q_delta = np.maximum(np.minimum(q_delta, self._vmax), -self._vmax)
            q = q + (q_delta * self._dt)
            q = np.maximum(np.minimum(q, self._jmax), -self._jmax)
            
        if not reached:
            self._output_info('Max iteration ({0}) exceeded.'.format(self._maxiter), 'red')
            return None

        return q

    def _compute_objective(self, robot_index, target_pose, q):
        error = self._compute_error(robot_index, target_pose, q)
        obj = self._weight * np.dot(error, error)
        return obj


    def _compute_error(self, robot_index, target_pose, q):
        with self.robots[robot_index]:
            self.robots[robot_index].SetActiveDOFValues(q)
            cur_pose = self.manips[robot_index].GetTransformPose()
        if cur_pose[0] < 0:
            cur_pose[0:4] *= -1.
        error = target_pose - cur_pose
        
        return error

    def _compute_q_delta(self, robot_index, target_pose, q):
        with self.robots[robot_index]:
            self.robots[robot_index].SetActiveDOFValues(q)
            # Jacobian
            J_trans = self.manips[robot_index].CalculateJacobian()
            J_quat = self.manips[robot_index].CalculateRotationJacobian()
            
            cur_pose = self.manips[robot_index].GetTransformPose()
        if cur_pose[0] < 0:
            cur_pose[0:4] *= -1.
            J_quat *= -1.
        
        # Full Jacobian
        J = np.vstack([J_quat, J_trans])

        weight = 10.0

        # J is a [7x6] matrix, need to use pinv()
        q_delta = weight * np.dot(np.linalg.pinv(J), (target_pose - cur_pose))
        return q_delta

    def _output_debug(self, msg, color=None, bold=True):
        if self._debug:
            if color is None:
                formatted_msg = msg
            else:
                formatted_msg = colorize(msg, color, bold)
            func_name = traceback.extract_stack(None, 2)[0][2]
            print '[BimanualObjectTracker::' + func_name + '] ' + formatted_msg

    def _output_info(self, msg, color=None, bold=True):
        if color is None:
            formatted_msg = msg
        else:
            formatted_msg = colorize(msg, color, bold)
        func_name = traceback.extract_stack(None, 2)[0][2]
        print '[BimanualObjectTracker::' + func_name + '] ' + formatted_msg


class CCPlannerException(Exception):
    '''
    Base class for exceptions for cc planners
    '''
    pass
