"""
Loggers.
"""
import traceback
import rospy

class TextColors(object):
  """
  Can be used as alternative of C{rospy}.
  """
  RED    = '\033[31m'
  GREEN  = '\033[32m'
  ORANGE = '\033[33m'
  ENDC   = '\033[0m'

  # Log level
  DEBUG = 0
  INFO  = 1
  WARN  = 2
  ERROR = 3
  FATAL = 4

  def __init__(self, log_level=INFO):
    self.log_level = log_level
  
  def red(self, msg):
    """
    Return a color message formatted in B{red}
    @type  msg: string
    @param msg: the message to be printed.
    """
    return self.RED + msg + self.ENDC
  
  def green(self, msg):
    """
    Return a color message formatted in B{green}
    @type  msg: string
    @param msg: the message to be printed.
    """
    return self.GREEN + msg + self.ENDC
  
  def yellow(self, msg):
    """
    Return a color message formatted in B{yellow}
    @type  msg: string
    @param msg: the message to be printed.
    """
    return self.ORANGE + msg + self.ENDC
  
  def logdebug(self, msg):
    """
    Print the message in green with a header '[DEBUG]'. 
    Alternative to C{rospy.logdebug}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= self.DEBUG:
      func_name = '[' + traceback.extract_stack(None, 2)[0][2] + '] '
      print(self.green('[DEBUG] ' + func_name + msg))
    
  def loginfo(self, msg):
    """
    Print the message with a header '[INFO]'. 
    Alternative to C{rospy.loginfo}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= self.INFO:
      func_name = '[' + traceback.extract_stack(None, 2)[0][2] + '] '
      print('[INFO] ' + func_name + msg)
  
  def logwarn(self, msg):
    """
    Print the message in yellow with a header '[WARN]'. 
    Alternative to C{rospy.logwarn}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= self.WARN:
      func_name = '[' + traceback.extract_stack(None, 2)[0][2] + '] '
      print(self.yellow('[WARN] ' + func_name + msg))
  
  def logerr(self, msg):
    """
    Print the message in red with a header '[ERROR]'. 
    Alternative to C{rospy.logerr}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= self.ERROR:
      func_name = '[' + traceback.extract_stack(None, 2)[0][2] + '] '
      print(self.red('[ERROR] ' + func_name + msg))
  
  def logfatal(self, msg):
    """
    Print the message in red with a header '[FATAL]'. 
    Alternative to C{rospy.logfatal}.
    @type  msg: string
    @param msg: the message to be printed.
    """
    if self.log_level <= self.FATAL:
      func_name = '[' + traceback.extract_stack(None, 2)[0][2] + '] '
      print(self.red('[FATAL] ' + func_name + msg))
  
  def set_log_level(self, level):
    """
    Sets the log level. Possible values are:
      - DEBUG:  0
      - INFO:   1
      - WARN:   2
      - ERROR:  3
      - FATAL:  4
    @type  level: int
    @param level: the new log level
    """
    self.log_level = level