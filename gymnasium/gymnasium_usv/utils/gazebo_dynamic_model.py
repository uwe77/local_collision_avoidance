import rospy
import numpy as np
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion
from .gazebo_base_model import GazeboBaseModel
from scipy.spatial.transform import Rotation as R
from .adaptive_pd_controller import AdaptivePDWithBias


class GazeboDynamicModel(GazeboBaseModel):
    
    def __init__(self, name="redball", init_pose=Pose(), max_speed=0.0):
        super(GazeboDynamicModel, self).__init__(name, init_pose)

        self.pose = np.zeros(3) # x, y, yaw
        self.local_vel = np.zeros(2)
        self.global_vel = np.zeros(2)
        self.max_speed = max_speed
        if max_speed > 0.0:
            self.x_pid = AdaptivePDWithBias(
                kp=1.0, kd=0.1, kff=1.0, init_input_bias=max_speed, 
                output_limits=(-np.inf, np.inf),
                adapt_lr=0.01, bias_lr=0.001, gain_bounds=(0.1, 20)
            )

    def reset(self, pose=None, target_vel=0.0):
        self.contact = False
        self.laser = None
        self.local_vel = np.zeros(2)
        self.global_vel = np.zeros(2)
        self.target_vel = target_vel
        if pose is None:
            pose = self.init_pose
            return super(GazeboDynamicModel, self).reset()
        q = R.from_euler('zyx', [pose[2], 0, 0]).as_quat()
        pose = Pose(
            position=Point(x=pose[0], y=pose[1], z=0.0),
            orientation=Quaternion(
                x=q[0], y=q[1], z=q[2], w=q[3]
            )
        )
        return super(GazeboDynamicModel, self).reset(pose)        

    def step(self, dt=0.04):
        # Apply PD control to compute the force and torque
        if self.max_speed < 0.0:
            return self.update_state()
        force = Vector3()
        force.x = self.x_pid.compute(self.target_vel, self.local_vel[0], dt, adapt_mode=True)
        force.y = 0.0
        force.z = 0.0
        self.apply_force(force=force, torque=Vector3(), point=Point(), dt=dt)
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

        self.local_vel[0] =   np.cos(self.pose[2]) * self.global_vel[0] + np.sin(self.pose[2]) * self.global_vel[1]
        self.local_vel[1] = - np.sin(self.pose[2]) * self.global_vel[0] + np.cos(self.pose[2]) * self.global_vel[1]

        return state