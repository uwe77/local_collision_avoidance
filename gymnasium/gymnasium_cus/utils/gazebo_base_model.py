import rospy
import numpy as np
from geometry_msgs.msg import Vector3, Pose
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Wrench, Point
from gazebo_msgs.srv import ApplyBodyWrench


class GazeboBaseModel:

    def __init__(self, name="model", init_pose=Pose()):
        self.model_state = ModelState()
        self.model_state.model_name = name
        self.model_state.pose = init_pose
        self.name = name
        self.init_pose = init_pose

        self.__apply_body_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        self.__set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.__get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def __wait_for_service_safe(self, service_name, timeout=5.0, max_retries=3):
        """Wait for service with timeout and retry."""
        for i in range(max_retries):
            try:
                rospy.wait_for_service(service_name, timeout=timeout)
                return True
            except rospy.ROSException as e:
                rospy.logwarn(f"[{service_name}] not available (attempt {i+1}/{max_retries}): {e}")
        rospy.logerr(f"[{service_name}] failed to be available after {max_retries} attempts.")
        return False

    def reset(self, pose=None):
        if pose is None:
            pose = self.init_pose
        self.set_pose(pose)
        return pose

    def set_pose(self, pose=Pose()):
        service_name = '/gazebo/set_model_state'
        if self.__wait_for_service_safe(service_name):
            try:
                self.model_state.pose = pose
                self.__set_model_state(self.model_state)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call to {service_name} failed: {e}")

    def get_state(self):
        service_name = '/gazebo/get_model_state'
        if self.__wait_for_service_safe(service_name):
            try:
                return self.__get_model_state(self.name, "world")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call to {service_name} failed: {e}")
        return None

    def apply_force(self, force=Vector3(), torque=Vector3(), point=Point()):
        service_name = '/gazebo/apply_body_wrench'
        if self.__wait_for_service_safe(service_name):
            try:
                wrench = Wrench()
                wrench.force = force
                wrench.torque = torque
                self.__apply_body_wrench(
                    body_name=f"{self.name}::base_link",
                    reference_frame="world",
                    reference_point=point,
                    wrench=wrench,
                    start_time=rospy.Time(0.0),
                    duration=rospy.Duration(0.04)
                )
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call to {service_name} failed: {e}")
