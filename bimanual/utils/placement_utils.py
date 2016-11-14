"""
Utility functions related to regrasp placement used for motion planners in ikea_planner package.
"""
from pymanip.planningutils import myobject

def create_placement_object(obj, env, T_rest=None):
  with env:
    bodies = env.GetBodies()
    robots = env.GetRobots()
    manips = [robot.GetActiveManipulator() for robot in robots]
    for body in bodies:
      if body.GetName() != obj.GetName():
        env.Remove(body)
    pobj = myobject.MyObject(obj)
    for body in bodies:
      if body.GetName() != obj.GetName():
        env.Add(body)
    for manip, robot in zip(manips, robots):
      robot.SetActiveManipulator(manip.GetName())
  if T_rest is not None:
    pobj.SetRestingSurfaceTransform(T_rest)
  return pobj