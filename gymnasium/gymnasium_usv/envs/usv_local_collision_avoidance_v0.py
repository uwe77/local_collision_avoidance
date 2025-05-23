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
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from types import Optional

from gymnasium_usv.utils import (
    GazeboUSVModel,
    GazeboBaseModel,
    GazeboROSConnector
)

class USVLocalCollisionAvoidanceV0(gym.Env):

    def __init__(
        self,
        usv_name: str = "js",
        enable_obstacle: bool = False,
        obstacle_max_speed: float = 5.0,
        reset_range: float = 200.0,
    ):
        super().__init__()

        self.info = {
            'usv_name': usv_name,
            'reset_range': reset_range,
            'current_step': 0,
            'max_steps': 4096,
            'max_reward': 100,
            'max_laser_dis': 100,
            'max_track_dis': 30,
            'max_vel': np.inf,
            'laser_shape': (241, ),
            'track_shape': (3, ),
            'vel_shape': (2, ),
            'action_shape': (2, ),
            'goal_pose': np.array([0,0,0]),
            'goal_range': 8, # meters
            'safe_laser_range': 30,
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
            low=np.array([0.0, -1]), 
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
            " {:4.2f}".format(self.reward*self.info['max_reward']) if self.reward >= 0 else "{:4.2f}".format(self.reward*self.info['max_reward']),
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

        track = np.array([np.clip(dist_diff, self.info['max_track_dis']), angle, angle_diff])
        
        vel = self.usv.local_vel

        scan = np.array(self.usv.laser.ranges)
        # scan = self.__scan_encoder(self.usv.laser)
        if scan is None:
            scan = np.full(self.info['laser_shape'][1], self.info['max_laser_dis'])
        scan = np.clip(scan, 0, self.info['max_laser_dis'])

        self.obs = {"laser": scan, "track": track, "vel": vel}
        return self.obs

    def get_reward(self, action):
        """
        Modified reward function with:
        - Yaw velocity penalty (to reduce frequent rotations).
        - Potential-based reward shaping (to escape local optima).
        """

        # Discount factor for potential shaping
        gamma = 0.99

        # Step penalty (if you want to penalize longer episodes, you can set this > 0)
        step_penalty = 0.0

        # -------------------------------------------------------------------------
        # 1) Compute standard variables
        # -------------------------------------------------------------------------
        vec_to_goal = self.info['goal_pose'][:2] - self.usv.pose[:2]
        dist_to_goal = np.linalg.norm(vec_to_goal)

        # Heading difference to goal in [-pi, pi]
        aline_to_goal = np.arctan2(vec_to_goal[1], vec_to_goal[0]) - self.usv.pose[2]
        aline_to_goal = (aline_to_goal + np.pi) % (2 * np.pi) - np.pi

        # Distance difference from last step (positive if we moved closer)
        dist_diff = self.last_data['dist_to_goal'] - dist_to_goal

        # Track how much total distance we have traveled (if needed)
        self.last_data['total_dist'] += abs(dist_diff)

        # -------------------------------------------------------------------------
        # 2) Base navigation reward (your existing logic)
        # -------------------------------------------------------------------------
        nav_reward = 0.005 * np.exp(-10 * (aline_to_goal / np.pi) ** 2)

        # Reward forward progress when moving closer to goal with forward velocity
        if dist_diff > 0.0 and self.usv.local_vel[0] >= 0:
            nav_reward += 0.005 * np.exp(-250 * (self.usv.local_vel[1]) ** 2) \
                            * np.clip(self.usv.local_vel[0], 0, 5) / 5
        else:
            # Small penalty if going backward or not improving distance
            nav_reward += -0.005 * abs(0.2 * self.usv.local_vel[0])

        # -------------------------------------------------------------------------
        # 3) Goal check (terminal reward)
        # -------------------------------------------------------------------------
        if dist_to_goal < self.info['goal_range']:
            self.termination = True
            nav_reward += self.info['max_reward']

        # -------------------------------------------------------------------------
        # 4) Collision penalty (your existing logic)
        # -------------------------------------------------------------------------
        compressed_scan = self.__scan_encoder(self.usv.laser)
        safe_range = self.info['max_laser_dis'] - self.info['safe_laser_range']

        # We'll accumulate a penalty for each sector that is under safe range
        # Optionally, weight by how relevant that sector is, e.g. if it's the front
        # sector and we're moving forward, penalize more. This is a simple example:
        obstacle_penalty = 0.0

        forward_speed = self.usv.local_vel[0]      # linear speed (forward/backward)
        yaw_speed = self.usv.local_vel[1]*2         # turning speed (positive=left turn, negative=right)

        # Example direction weighting:
        #   - front sector: penalize if moving forward quickly
        #   - front_right: penalize if moving forward & turning right
        #   - right: penalize if turning right
        #   - back_right: penalize if moving backward & turning right
        #   - back: penalize if moving backward
        #   - back_left: penalize if moving backward & turning left
        #   - left: penalize if turning left
        #   - front_left: penalize if moving forward & turning left
        #
        # You can keep this as simple or detailed as you want. Below is one possible mapping:

        dir_factors = np.zeros(8)
        # front (0)
        dir_factors[0] = max(0, forward_speed)
        # front_right (1)
        dir_factors[1] = max(0, forward_speed) + max(0, -yaw_speed)
        # right (2)
        dir_factors[2] = max(0, -yaw_speed)
        # back_right (3)
        dir_factors[3] = max(0, -np.clip(forward_speed,-1,1)) + max(0, -yaw_speed)
        # back (4)
        dir_factors[4] = max(0, -np.clip(forward_speed,-1,1))
        # back_left (5)
        dir_factors[5] = max(0, -np.clip(forward_speed,-1,1)) + max(0, yaw_speed)
        # left (6)
        dir_factors[6] = max(0, yaw_speed)
        # front_left (7)
        dir_factors[7] = max(0, forward_speed) + max(0, yaw_speed)

        obstacle_penalty = np.zeros(compressed_scan.shape[0])
        # if one of a data in commpressed_scan is less than safe_range
        if np.any(compressed_scan < safe_range):
            for i, dist in enumerate(compressed_scan):
            # if dist < safe_range:
                # Closer → bigger penalty
                proximity_ratio = 1.0 - (dist / safe_range)  # in [0, 1]
                # Multiply by the direction factor
                penalty = proximity_ratio * dir_factors[i]
                obstacle_penalty[i] = penalty

        obstacle_penalty = np.mean(obstacle_penalty)
        
        # We still keep a "hard collision check" if truly collided or extremely close
        collision_penalty = 0.0
        min_laser_dis = np.min(compressed_scan)

        # If truly collided or extremely close
        if min_laser_dis < self.info['safe_laser_range'] or self.usv.contact is True:
            # If truly collided after some steps
            if self.info['current_step'] > 10:
                collision_penalty = np.clip(abs(self.usv.local_vel[0]),1, np.inf) * self.info['max_reward']
                nav_reward = 0
                self.termination = True
        # collision_penalty = 0

        # if min_laser_dis < self.info['safe_laser_range'] or self.usv.contact is True:
        #     # If truly collided after some steps
        #     if self.info['current_step'] > 10:
        #         collision_penalty = abs(self.usv.local_vel[0]) * self.info['max_reward']
        #         nav_reward = 0
        #         self.termination = True
        # else:
        #     # If not colliding, small "bonus" for safer distance
        #     collision_penalty = 0.01 * np.exp(-5 * (min_laser_dis / self.info['max_laser_dis']) ** 2)

        # -------------------------------------------------------------------------
        # 5) Over-deviation: if agent gets too far from initial distance
        # -------------------------------------------------------------------------
        if dist_to_goal > self.last_data['init_dist_to_goal'] * 1.5:
            self.termination = True

        # -------------------------------------------------------------------------
        # 6) Yaw velocity penalty (new)
        #    Penalize large angular velocity to discourage frequent rotation
        # -------------------------------------------------------------------------
        yaw_vel = self.usv.local_vel[1]   # angular velocity around z-axis
        yaw_penalty = 0.01 * ((2*yaw_vel) ** 2)
        

        # -------------------------------------------------------------------------
        # 7) Combine unshaped reward
        # -------------------------------------------------------------------------
        nav_weight = 1.0
        # collision_weight = 0.1 * np.clip(dist_to_goal, 0, self.info['max_track_dis']) / self.info['max_track_dis']
        collision_weight = 0.1
        obstacle_weight = 0.01
        step_weight = 0.5

        # Base reward (before potential shaping)
        base_reward = (
            nav_weight * nav_reward
            - obstacle_weight * obstacle_penalty
            - collision_weight * collision_penalty
            - step_weight * step_penalty
            - yaw_penalty
        )

        # Update stored distance for next step
        self.last_data['dist_to_goal'] = dist_to_goal

        # Final reward
        proximity_reward = base_reward-self.reward
        self.reward = base_reward*0.7 + proximity_reward*0.3
        return self.reward

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