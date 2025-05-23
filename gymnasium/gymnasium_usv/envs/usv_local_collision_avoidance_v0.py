import gymnasium as gym
from gymnasium import spaces
import rospy
import time
import numpy as np
import math
import random, sys, os
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped, Point
from std_msgs.msg import Int64, Header, UInt8
from sensor_msgs.msg import LaserScan

from gymnasium_usv.utils import (
    GazeboUSVModel,
    GazeboBaseModel,
    GazeboROSConnector
)

class USVLocalCollisionAvoidanceV0(gym.Env):

    def __init__(
        self,
        *,
        usv_name: str = "js",
        enable_obstacle: bool = False,
        obstacle_max_speed: float = 5.0,
        reset_range: float = 200.0,
        render_mode: str = None,    # 新增，讓 Gymnasium 的 render_mode 能被接收
        **kwargs                     # 吞下其他所有多餘參數
    ):
        super().__init__()

        self.info = {
            'usv_name': usv_name,
            'reset_range': reset_range,
            'current_step': 0,
            'max_steps': 4096,
            'max_laser_dis': 100,
            'max_track_dis': 30,
            'max_vel': np.inf,
            'laser_shape': (241, ),
            'track_shape': (3, ),
            'vel_shape': (2, ),
            'action_shape': (2, ),
            'goal_pose': np.array([0,0,0]),
            'goal_range': 8, # meters
            'safe_laser_range': 50,
            'collision_laser_range': 15,
        }
        self.last_data = {
            'dist_to_goal': None,
            'total_dist': 0,
            'init_dist_to_goal': None,
        }
        self.obs = None
        self.termination = False
        self.truncation = False
        self.reward = 0

        rospy.init_node("usv_local_collision_avoidance_v0", anonymous=True)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        
        self.usv = GazeboUSVModel(usv_name)
        self.gazebo = GazeboROSConnector()
        self.gazebo.unpause_physics()
        
        # self.__reset_usv_pose()
        self.__reset_usv_and_goal()

        self.observation_space = gym.spaces.Dict({
            "laser": gym.spaces.Box(
                low=0,
                high=self.info['max_laser_dis'], 
                shape=self.info['laser_shape'], dtype=np.float64),
            "track": gym.spaces.Box(
                low=np.array([0, -np.pi, -np.pi]),
                high=np.array([self.info['max_track_dis'], np.pi, np.pi]),
                shape=self.info['track_shape'], dtype=np.float64),
            "vel": gym.spaces.Box(
                low=-self.info['max_vel'], 
                high=self.info['max_vel'],
                shape=self.info['vel_shape'], dtype=np.float64)
        })
        self.action_space = spaces.Box(
            low=np.array([0, -1]), 
            high=np.array([1, 1]), 
            shape=self.info['action_shape'], dtype=np.float32)
        
        print("USV Local Collision Avoidance V0 Environment Initialized")
        print("Observation Space: ", self.observation_space)
        print("Action Space: ", self.action_space)
        self.reset()
        # self.__reset_goal(0, 0, random.uniform(-np.pi, np.pi))


    def step(self, action):
        self.info['current_step'] += 1
        self.gazebo.unpause_physics()
        self.usv.step(action)
        self.get_observation()
        self.reward = self.get_reward(action)
        
        if self.info['current_step'] >= self.info['max_steps']:
            self.truncation = True

        self.gazebo.pause_physics()
        output = "\rstep:{:4d}, reward:{}".format(
            self.info['current_step'],
            " {:4.2f}".format(self.reward) if self.reward >= 0 else "{:4.2f}".format(self.reward),
        )
        sys.stdout.write(output)
        sys.stdout.flush()

        return self.obs, self.reward, self.termination, self.truncation, self.info
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        print()
        self.gazebo.unpause_physics()
        self.info['current_step'] = 0
        self.reward = 0
        self.termination = False
        self.truncation = False
        self.__reset_usv_and_goal()
        self.usv.update_state()
        self.last_data['init_dist_to_goal'] = np.linalg.norm(self.info['goal_pose'][:2] - self.usv.pose[:2])
        self.last_data['dist_to_goal'] = self.last_data['init_dist_to_goal']
        self.last_data['total_dist'] = 0
        self.get_observation()
        self.gazebo.pause_physics()
        return self.obs, self.info
    
    def get_observation(self):
        posi_diff = self.info['goal_pose'][:2] - self.usv.pose[:2]
        angle = np.arctan2(posi_diff[1], posi_diff[0])-self.usv.pose[2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        dist_diff = np.linalg.norm(posi_diff)
        
        angle_diff = self.info['goal_pose'][2] - self.usv.pose[2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        track = np.array([np.clip(dist_diff, 0, self.info['max_track_dis']), angle, angle_diff])
        
        vel = self.usv.local_vel

        scan = np.array(self.usv.laser.ranges)
        # scan = self.__scan_encoder(self.usv.laser)
        if scan is None:
            scan = np.full(self.info['laser_shape'][1], self.info['max_laser_dis'])
        scan = np.clip(scan, 0, self.info['max_laser_dis'])

        self.obs = {"laser": scan, "track": track, "vel": vel}
        return self.obs

    def get_reward(self, action):
        laser = self.obs['laser']
        # track = self.obs['track']
        vel = self.obs['vel']
        # constants
        wr = self.info['safe_laser_range']
        r_usv = self.info['collision_laser_range']  # vessel radius approximation 
        omega_w = 1.0
        omega_g = 1.0
        omega_a = 0.5
        omega1 = 1.0
        omega2 = 1.0
        vmax = vel[0] if vel[0] > 0 else 1.0

        # 1) Warning zone reward
        lmin = np.min(laser)
        rw = omega_w * (lmin - r_usv - wr)/self.info['max_laser_dis']
        if lmin < self.info['collision_laser_range']:
            self.termination = True
            rw = -0.4*self.info['max_steps']

        # 2) Goal reaching reward
        posi_diff = self.info['goal_pose'][:2] - self.usv.pose[:2]
        dist_diff = np.linalg.norm(posi_diff)
        prev_rho = self.last_data['dist_to_goal']
        rho = dist_diff
        rg = omega_g if (prev_rho - rho) > 0 else -omega_g
        self.last_data['dist_to_goal'] = rho
        if rho < self.info['goal_range']:
            self.termination = True
            rg = 10*self.info['max_steps']


        # 3) Action continuity reward (use observed yaw rate from vel)
        # yaw_rate_obs = vel[1]
        # prev_yaw = self.last_data.get('prev_yaw_rate', 0.0)
        # ra = -omega_a if yaw_rate_obs * prev_yaw < 0 else omega_a
        # self.last_data['prev_yaw_rate'] = yaw_rate_obs
        prev_action = self.last_data.get('prev_action', np.zeros(2))
        ra = -omega_a if np.linalg.norm(action-prev_action) > 0.5 else omega_a
        # ra = -omega_a if np.any(prev_action * action < 0) else omega_a
        self.last_data['prev_action'] = action


        # 4) COLREGs compliance reward
        idx_min = np.argmin(laser)
        scan_msg = self.usv.laser
        theta = scan_msg.angle_min + idx_min * scan_msg.angle_increment
        psi = self.usv.pose[2]
        speed = abs(vel[0])
        if -math.radians(5) <= theta < math.radians(112.5):
            rc = omega1 * (math.pi/2 - psi) * math.exp(-speed / vmax)
        elif math.radians(247.5) < theta < math.radians(355):
            rc = omega2 * psi * math.exp(-speed / vmax) if psi <= math.pi/2 else 0.0
        else:
            rc = 0.0
        return rw + rg + ra + rc

    def close(self):
        rospy.signal_shutdown('WTF')

    def __reset_goal(self, x, y, yaw):
        yaw = (yaw+np.pi) % (2*np.pi) - np.pi
        self.info['goal_pose'] = np.array([x, y, yaw])
        goal_pose = PoseStamped()
        goal_pose.header = Header()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.info['goal_pose'][0]
        goal_pose.pose.position.y = self.info['goal_pose'][1]
        goal_pose.pose.position.z = 0
        goal_pose.pose.orientation.x = 0
        goal_pose.pose.orientation.y = 0
        goal_pose.pose.orientation.z = np.sin(self.info['goal_pose'][2]/2)
        goal_pose.pose.orientation.w = np.cos(self.info['goal_pose'][2]/2)
        self.pub_goal.publish(goal_pose)

    def __reset_usv_pose(self, x, y, yaw):
        quat = R.from_euler('z', yaw).as_quat()
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        self.usv.set_pose(pose)

    def __reset_usv_and_goal(self):
        self.__reset_goal(0, 0, random.uniform(-np.pi, np.pi))
        angle = random.uniform(-np.pi, np.pi)
        self.__reset_usv_pose(
            self.info['reset_range']*np.cos(angle), 
            self.info['reset_range']*np.sin(angle),
            random.uniform(-np.pi, np.pi)
            )