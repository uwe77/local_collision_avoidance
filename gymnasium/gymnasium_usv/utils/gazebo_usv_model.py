import rospy
import numpy as np
from geometry_msgs.msg import Pose, Twist
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan
from .gazebo_base_model import GazeboBaseModel
from scipy.spatial.transform import Rotation as R


class GazeboUSVModel(GazeboBaseModel):
    
    def __init__(self, name="js", init_pose=Pose()):
        super(GazeboUSVModel, self).__init__(name, init_pose)

        self.__pub_cmd_vel = rospy.Publisher(f"/{name}/cmd_vel", Twist, queue_size=1)
        self.__sub_contact = rospy.Subscriber(f"/{name}/sensors/collision", ContactsState, self.__contact_callback)
        self.__sub_laser = rospy.Subscriber(f"/{name}/gazebo/scan", LaserScan, self.__laser_callback)
        self.alpha_gamma = 0.25
        self.action = np.zeros(2)
        self.last_action = np.zeros(2)
        self.last_alpha = np.zeros(2)
        self.contact = False
        self.laser = None
        self.pose = np.zeros(3) # x, y, yaw
        self.local_vel = np.zeros(2)
        self.global_vel = np.zeros(2)

    def reset(self, pose=None):
        self.contact = False
        self.laser = None
        self.pose = np.zeros(3)
        self.local_vel = np.zeros(2)
        self.global_vel = np.zeros(2)
        self.action = np.zeros(2)

        return super(GazeboUSVModel, self).reset(pose)

    def step(self, action = np.zeros(2)):
        
        self.last_alpha = self.alpha_gamma * action + (1 - self.alpha_gamma) * self.last_alpha

        self.action = np.array([x if abs(x) >= 1e-3 else 0 for x in self.last_alpha])
        cmd_vel = Twist()
        cmd_vel.linear.x = self.action[0]
        cmd_vel.angular.z = self.action[1]
        self.__pub_cmd_vel.publish(cmd_vel)

        return self.update_state()

    def update_state(self):
        state = self.get_state()
        self.pose[0] = state.pose.position.x
        self.pose[1] = state.pose.position.y
        self.pose[2] = R.from_quat([
            state.pose.orientation.x, state.pose.orientation.y, 
            state.pose.orientation.z, state.pose.orientation.w]).as_euler('zyx')[0]
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi

        self.global_vel[0] = state.twist.linear.x
        self.global_vel[1] = state.twist.linear.y

        self.local_vel[0] = np.cos(self.pose[2]) * self.global_vel[0] + np.sin(self.pose[2]) * self.global_vel[1]
        self.local_vel[1] = state.twist.angular.z

        return state
    

    def __contact_callback(self, msg):
        if msg.states != []:
            self.contact = True

    def __laser_callback(self, msg):
        self.laser = msg