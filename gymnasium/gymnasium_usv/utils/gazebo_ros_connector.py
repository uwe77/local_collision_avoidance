import rospy, os
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from geometry_msgs.msg import Vector3
from std_srvs.srv import Empty as EmptySrv


class GazeboROSConnector:
    def __init__(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        os.system('gz physics -u 0')
