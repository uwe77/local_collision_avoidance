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
from rosgraph_msgs.msg import Clock


from gymnasium_usv.utils import (
    GazeboUSVModel,
    GazeboDynamicModel,
    GazeboROSConnector
)
def navigate(heading_diff: float,
             goal_dist: float,
             max_track_dist: float)-> np.ndarray:
    """
    Compute forward velocity and yaw rate commands.

    Args:
        heading_diff (float): heading error to goal, in radians ([-π, +π]).
        max_yaw_rate (float): maximum allowed yaw rate (rad/s).
        goal_dist (float): distance to goal (m).

    Returns:
        forward_vel_cmd (float): commanded forward velocity (m/s).
        yaw_rate_cmd (float): commanded yaw rate (rad/s).
    """

    # --- YAW CONTROL (always active) ---
    default_speed = 0.5
    # tou = -2*np.log(default_speed)
    Kp_yaw = 1.0/np.pi  # tune as needed
    yaw_rate_cmd = Kp_yaw * heading_diff
    # saturate to limits
    yaw_rate_cmd = np.clip(yaw_rate_cmd, -1, 1)

    # if heading error > 90°, rotate in place only
    if abs(heading_diff) > np.pi / 2:
        return default_speed, yaw_rate_cmd

    # --- FORWARD CONTROL (only when roughly facing goal) ---
    forward_vel = goal_dist/ max_track_dist
    # reduce speed when heading_diff nonzero
    forward_vel *= max(0.0, 1)
    # saturate to limits
    forward_vel_cmd = np.clip(forward_vel, -1.0, 1.0)
    action = np.zeros(2)
    action[0] = forward_vel_cmd
    action[1] = yaw_rate_cmd
    return action

class USVLocalCollisionAvoidanceV0(gym.Env):

    def __init__(
        self,
        *,
        usv_name: str = "js",
        max_steps: int = 512,
        obstacle_max_speed: float = 10.0,
        obstacle_numbers: int = 4,
        obstacle_name: str = 'redball',
        reset_range: float = 200.0,
        step_hz: float = 10.0,
        pid_hz: float = 100.0,
        render_mode: str = None,    # 新增，讓 Gymnasium 的 render_mode 能被接收
        **kwargs                     # 吞下其他所有多餘參數
    ):
        super().__init__()

        self.info = {
            'usv_name': usv_name,
            'reset_range': reset_range,
            'reset_weight': 0.5,  # 重置時，x 軸的隨機範圍
            'obstacle_numbers': obstacle_numbers,
            'obstacle_max_speed': obstacle_max_speed,  # 障礙物最大速度
            'obstacle_name': obstacle_name,  # 障礙物名稱
            'step_dt': rospy.Duration(1.0 / step_hz),  # how much sim‐time each step should advance
            'pid_dt': rospy.Duration(1.0 / pid_hz),    # how much sim‐time each PID step should advance
            'current_step': 0,
            'max_steps': max_steps,
            'max_laser_dis': 100,
            'max_track_dis': 50.0,
            'max_vel': np.inf,
            'laser_shape': (241, ),
            'track_shape': (2, ),
            'vel_shape': (2, ),
            'action_shape': (2, ),
            'goal_pose': np.array([0,0,0]),
            'goal_range': 15, # meters
            'safe_laser_range': 50,
            'collision_laser_range': 15,
            'clock': rospy.Time(0),  # 用於模擬時間
        }
        self.last_data = {
            'dist_to_goal': None,
            'total_dist': 0,
            'init_dist_to_goal': None,
            'shaping': 0,
            'obs': None,
            'acc': None,
            'vel': None,
        }
        self.obs = None
        self.termination = False
        self.truncation = False
        self.reward = 0

        rospy.init_node("usv_local_collision_avoidance_v0", anonymous=True)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        rospy.Subscriber('/clock', Clock, self._clock_cb)
        rospy.wait_for_message('/clock', Clock)
        self.usv = GazeboUSVModel(usv_name)
        self.obstacles = []
        if self.info['obstacle_numbers'] > 0:
            init_range = 1000
            d_angle = 2 * np.pi / self.info['obstacle_numbers']
            for i in range(self.info['obstacle_numbers']):
                obstacle = GazeboDynamicModel(
                    name=f"{self.info['obstacle_name']}_{i}",
                    init_pose=Pose(
                        position=Point(
                            x=init_range * np.cos(i * d_angle),
                            y=init_range * np.sin(i * d_angle),
                            z=0.0
                        )
                    ),
                    max_speed=self.info['obstacle_max_speed']
                )
                self.obstacles.append(obstacle)
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
                low=np.array([-self.info['max_track_dis'], -self.info['max_track_dis']]),
                high=np.array([self.info['max_track_dis'], self.info['max_track_dis']]),
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
        # ========================= Apply Action =========================
        sigmoid = lambda x: 1 / (1 + np.exp(-5*x)) # x = -1~1
        d_range = self.info['max_laser_dis'] - self.info['safe_laser_range']
        center_range = self.info['safe_laser_range'] + d_range / 2
        lmin = self.obs['laser'].min()
        laser_weight = np.clip(sigmoid((lmin-center_range) / d_range), 0, 1)
        track_weight = np.clip(np.linalg.norm(self.obs['track']) / self.info['max_track_dis'], 0 , 1)
        action_weight = laser_weight * track_weight

        self.gazebo.unpause_physics()
        
        # self.usv.step(action, dt=self.info['step_dt'].to_sec())
        nav_action = navigate(
            heading_diff=np.arctan2(self.obs['track'][1], self.obs['track'][0]),
            goal_dist=np.linalg.norm(self.obs['track']),
            max_track_dist=self.info['max_track_dis']
        )
        action[0] = nav_action[0] * action_weight + action[0] * (1 - action_weight)
        action[1] = nav_action[1] * action_weight + action[1] * (1 - action_weight)
        
        self.info['current_step'] += 1
        start = self.info['clock']
        
        self.usv.step(action, dt=self.info['pid_dt'].to_sec())
        if self.info['obstacle_numbers'] > 0:
            for obstacle in self.obstacles:
                obstacle.step(dt=self.info['pid_dt'].to_sec())
        
        while self.info['clock'] < start + self.info['step_dt']:
            rospy.sleep(self.info['pid_dt'].to_sec())
            self.usv.step(action, dt=self.info['pid_dt'].to_sec())
            if self.info['obstacle_numbers'] > 0:
                for obstacle in self.obstacles:
                    obstacle.step(dt=self.info['pid_dt'].to_sec())

        # ========================= Get Observation and Reward =========================
        self.get_observation()
        reward = self.get_reward(action)
        self.last_data['last_action'] = self.last_data['action']
        self.last_data['action'] = action.copy()
        if self.info['current_step'] >= self.info['max_steps']:
            self.truncation = True
            reward = -self.info['max_steps']/5
            
        self.reward = reward / self.info['max_steps']

        self.gazebo.pause_physics()
        
        output = "\rstep:{:4d}, action_weight:{}, track:[{},{}] vel:[{},{}], reward:{}".format(
            self.info['current_step'],
            " {:4.2f}".format(action_weight),
            " {:4.2f}".format(self.obs['track'][0]) if self.obs['track'][0] >= 0 else "{:4.2f}".format(self.obs['track'][0]),
            " {:4.2f}".format(self.obs['track'][1]) if self.obs['track'][1] >= 0 else "{:4.2f}".format(self.obs['track'][1]),
            " {:4.2f}".format(self.obs['vel'][0]) if self.obs['vel'][0] >= 0 else "{:4.2f}".format(self.obs['vel'][0]),
            " {:4.2f}".format(self.obs['vel'][1]) if self.obs['vel'][1] >= 0 else "{:4.2f}".format(self.obs['vel'][1]),
            " {:4.2f}".format(reward) if reward >= 0 else "{:4.2f}".format(reward),
        )
        sys.stdout.write(output)
        sys.stdout.flush()

        self.get_observation()
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
        self.__reset_obstacles()
        rospy.sleep(1.0)
        self.usv.update_state()
        self.last_data['shaping'] = 0.0
        self.last_data['action'] = np.zeros(self.info['action_shape'])
        self.last_data['last_action'] = np.zeros(self.info['action_shape'])
        self.last_data['init_dist_to_goal'] = np.linalg.norm(self.info['goal_pose'][:2] - self.usv.pose[:2])
        self.last_data['dist_to_goal'] = self.last_data['init_dist_to_goal']
        self.last_data['total_dist'] = 0
        # self.last_data['laser'] = np.array(self.usv.laser.ranges)
        self.get_observation()
        self.gazebo.pause_physics()
        return self.obs, self.info
    
    def get_observation(self):
        posi_diff = self.info['goal_pose'][:2] - self.usv.pose[:2]
        angle = np.arctan2(posi_diff[1], posi_diff[0])-self.usv.pose[2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        dist_diff = np.linalg.norm(posi_diff)
        self.last_data['total_dist'] += abs(dist_diff-self.last_data['dist_to_goal'])
        self.last_data['dist_to_goal'] = dist_diff
        # angle_diff = self.info['goal_pose'][2] - self.usv.pose[2]
        # angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        if dist_diff > self.info['max_track_dis']:
            track = self.info['max_track_dis']*np.array([np.cos(angle), np.sin(angle)])
        else:
            track = dist_diff * np.array([np.cos(angle), np.sin(angle)])
        # track = np.array([np.clip(dist_diff, 0, self.info['max_track_dis']), angle])
        
        vel = self.usv.local_vel

        scan = np.array(self.usv.laser.ranges)
        # scan = self.__scan_encoder(self.usv.laser)
        if scan is None:
            scan = np.full(self.info['laser_shape'][1], self.info['max_laser_dis'])
        scan = np.clip(scan, 0, self.info['max_laser_dis'])

        self.obs = {"laser": scan, "track": track, "vel": vel}
        return self.obs

    def get_reward(self, action):
        sigmoid = lambda x: 1 / (1 + np.exp(-x)) # x = -1~1
        operator = lambda x: 1 if x > 0 else -1
        laser = self.obs['laser']
        track = self.obs['track']  # [dist_diff, angle, angle_diff]
        vel = self.obs['vel']
        lmin = laser.min()

        # === 基本量 ===
        weight_p = 1.0
        weight_v = 1.0
        weight_a = 0.5
        weight_w = 1.0
        weight_c = 3.0
        # dist_t = track[0]
        # heading = track[1]  # difference between heading and target
        dist_t = np.linalg.norm(track)
        heading = np.arctan2(track[1], track[0])

        goal_range = self.info['goal_range']
        safe_range = self.info['safe_laser_range']
        collision_range = self.info['collision_laser_range']
        max_track_dist = self.info['max_track_dis']

        # =============== nav ==============
        # === 進展獎勵 ===
        potential_reward = weight_p*(1+np.cos(heading))/2

        # === 前進獎勵 ===
        forward_reward = weight_v * (2*sigmoid(vel[0])-1)

        # === 減速懲罰（靠近時） ===
        slow_penalty = -abs(forward_reward) if dist_t < 2 * goal_range else 0.0

        # === 加速度懲罰 ===
        last_d_action = self.last_data['action']-self.last_data['last_action']
        d_acttion = action - self.last_data['action']
        acc_penalty = -0.5 if last_d_action[0] * d_acttion[0] < 0 else 0.0
        acc_penalty += -0.5 if last_d_action[1] * d_acttion[1] < 0 else 0.0
        acc_penalty *= weight_a
        # ==================================

        # =========== collision avoidance =============
        # === laser warning ===
        d_range = self.info['max_laser_dis'] - self.info['safe_laser_range']
        center_range = self.info['safe_laser_range'] + d_range / 2
        min_laser_penalty = weight_w * (1-np.clip(sigmoid(-10*(lmin-center_range) / d_range), 0, 1))
        # === COLREGs 2.5.1 ===
        colregs_reward = 0.0
        laser_angles = np.linspace(-np.pi/2, np.pi/2, len(laser))
        sector_mask = (laser < safe_range)  # consider only lasers in range
        sector_angles = laser_angles[sector_mask]
        sector_ranges = laser[sector_mask]

        if len(sector_angles) > 0:
            danger_angle = sector_angles[np.argmin(sector_ranges)]
            danger_angle = (danger_angle + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
            # If danger is to the right (starboard), yaw_rate should be positive (turn left)
            # If danger is to the left (port), yaw_rate should be nagative (turn right)
            # If danger is ahead, yaw_rate should be nagative (go right)
            if np.pi/36 < abs(danger_angle) < np.pi/2:
                # print("Danger to the right, turning left")
                colregs_reward += weight_c*(np.sign(-danger_angle) * np.cos(danger_angle) * (2*sigmoid(10*action[1]) -1))
            elif abs(danger_angle) <= np.pi/36 :
                colregs_reward += weight_c * np.cos(danger_angle) * (2*sigmoid(-10*action[1]) -1)

        # === 距離獎勵（包含終止條件） ===
        if dist_t < goal_range:
            self.termination = True
            return self.info['max_steps']
        elif self.usv.contact or lmin < collision_range:
            self.termination = True
            return -max(abs(vel[0]),1)*self.info['max_steps']/2

        # === 總 shaping reward ===
        # shaping = progress_reward + heading_reward + forward_reward + distance_reward + slow_penalty
        shaping = potential_reward + forward_reward + slow_penalty + acc_penalty + min_laser_penalty + colregs_reward

        # === reward 使用 shaping 差值 ===
        # if hasattr(self, 'prev_shaping'):
            # reward = (shaping - self.prev_shaping) + potential_reward + forward_reward + slow_penalty
            # reward = (shaping - self.last_data['shaping']) + potential_reward + forward_reward + slow_penalty + acc_penalty
        reward = (shaping - self.last_data['shaping']) + forward_reward + min_laser_penalty
        # else:
            # reward = potential_reward + forward_reward + slow_penalty
            # reward = 0.0
        self.last_data['shaping'] = shaping
        
        return float(reward)

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
        self.__reset_usv_pose(
            self.info['reset_range']*random.uniform(self.info['reset_weight'], 1),
            0.0,
            random.uniform(-np.pi, np.pi)
            )
        
    def _clock_cb(self, msg: Clock):
        # update our copy of sim-time
        self.info['clock'] = msg.clock

    def __reset_obstacles(self):
        """
        Reset the positions of all obstacles to random locations within a specified range.
        """
        usv_max_range = self.info['reset_range'] + self.info['safe_laser_range']
        usv_min_range = self.info['reset_range'] * self.info['reset_weight'] - self.info['safe_laser_range']
        for obstacle in self.obstacles:
            pos_range = random.uniform(0, usv_min_range)
            angle = random.uniform(-1, 1) * np.pi/2
            target_range = random.uniform(usv_min_range, usv_max_range)
            # heading vector: x = target_range-pose[0], y = target_range-pose[1]
            pose = np.array([
                pos_range * np.cos(angle),
                pos_range * np.sin(angle), 
                0.0
            ])
            obstacle_angle = np.arctan2(-pose[1], target_range - pose[0]) + random.uniform(-np.pi/6, np.pi/6)
            pose[2] = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            speed = random.uniform(0, self.info['obstacle_max_speed'])
            if pos_range <= self.info['safe_laser_range']*2:
                speed = self.info['obstacle_max_speed']
            obstacle.reset(pose, target_vel=speed)