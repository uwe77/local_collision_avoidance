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
        step_hz: float = 10.0,
        render_mode: str = None,    # 新增，讓 Gymnasium 的 render_mode 能被接收
        **kwargs                     # 吞下其他所有多餘參數
    ):
        super().__init__()

        self.info = {
            'usv_name': usv_name,
            'reset_range': reset_range,
            'step_dt': rospy.Duration(1.0 / step_hz),  # how much sim‐time each step should advance
            'current_step': 0,
            'max_steps': 512,
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
            # 'laser': None,
            'track': None,
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
                low=np.array([0, -np.pi]),
                high=np.array([self.info['max_track_dis'], np.pi]),
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
        self.gazebo.unpause_physics()
        self.usv.step(action)
        self.info['current_step'] += 1
        start = self.info['clock']
        while self.info['clock'] < start + self.info['step_dt']:
            rospy.sleep(0.0001)
        # print(f"HZ: {1.0/(self.info['clock']- start).to_sec():.2f}")
        self.get_observation()
        self.reward = self.get_reward(action)
        
        if self.info['current_step'] >= self.info['max_steps']:
            self.truncation = True

        self.gazebo.pause_physics()
        output = "\rstep:{:4d}, track:[{},{}] vel:[{},{}], reward:{}".format(
            self.info['current_step'],
            " {:4.2f}".format(self.obs['track'][0]) if self.obs['track'][0] >= 0 else "{:4.2f}".format(self.obs['track'][0]),
            " {:4.2f}".format(self.obs['track'][1]) if self.obs['track'][1] >= 0 else "{:4.2f}".format(self.obs['track'][1]),
            " {:4.2f}".format(self.obs['vel'][0]) if self.obs['vel'][0] >= 0 else "{:4.2f}".format(self.obs['vel'][0]),
            " {:4.2f}".format(self.obs['vel'][1]) if self.obs['vel'][1] >= 0 else "{:4.2f}".format(self.obs['vel'][1]),
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
        rospy.sleep(1.0)
        self.usv.update_state()
        self.last_data['init_dist_to_goal'] = np.linalg.norm(self.info['goal_pose'][:2] - self.usv.pose[:2])
        self.last_data['dist_to_goal'] = self.last_data['init_dist_to_goal']
        self.last_data['total_dist'] = 0
        # self.last_data['laser'] = np.array(self.usv.laser.ranges)
        self.last_data['track'] = np.zeros(self.info['track_shape'])
        self.last_data['vel'] = np.zeros(self.info['vel_shape'])
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

        track = np.array([np.clip(dist_diff, 0, self.info['max_track_dis']), angle])
        
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
        track = self.obs['track']   # [dist_diff, angle_diff]
        vel   = self.obs['vel']     # [v_forward, yaw_rate]

        # 常數
        wr    = self.info['safe_laser_range']
        r_usv = self.info['collision_laser_range']
        max_d = self.info['max_track_dis']
        max_v = max(1.0, vel[0])
        
        # 1) 到目標距離、速度、航向差的 shaping
        dist = track[0]  # 已 clip 到 [0, max_d]
        ang  = track[1]  # in [-π, π]
        lmin = laser.min()

        # (a) 距離懲罰：與目標距離成正比
        term_dist = -10.0 * (dist / max_d)

        # (b) 速度懲罰：前進速率越大，離終點越快，但也要保持穩定
        term_vel  = -10.0 * (abs(vel[0]) / max_v)

        # (c) 航向懲罰：航向偏差越大，懲罰越重
        term_ang  = -100.0 * (abs(ang) / np.pi)

        # (d) 對齊獎勵：action 與目標方向夾角越小，獎勵越大
        align = action[0] * np.cos(ang) + action[1] * np.sin(ang)
        term_align = +10.0 * align

        # (e) 障礙警告：靠近障礙時輕微懲罰
        if lmin < wr:
            term_warn = -10.0 * (1.0 - (lmin - r_usv) / (wr - r_usv))
        else:
            term_warn = 0.0

        # (f)) COLREGs compliance reward
        idx_min = np.argmin(laser)
        scan_msg = self.usv.laser
        theta = scan_msg.angle_min + idx_min * scan_msg.angle_increment
        psi = self.usv.pose[2]
        speed = abs(vel[0])
        if -math.radians(5) <= theta < math.radians(112.5):
            term_colregs = 10 * (math.pi/2 - psi) * math.exp(-speed / max_v)
        elif math.radians(247.5) < theta < math.radians(355):
            term_colregs = 10 * psi * math.exp(-speed / max_v) if psi <= math.pi/2 else 0.0
        else:
            term_colregs = 0.0

        shaping = term_dist + term_ang + term_vel + term_align + term_warn + term_colregs

        # 2) reward = shaping 差值
        if hasattr(self, 'prev_shaping'):
            reward = shaping - self.prev_shaping
        else:
            reward = 0.0
        self.prev_shaping = shaping

        # 3) 油料/動作改變成本
        #    油料成本：油門平方項
        # reward -= 0.3 * (action[0]**2)
        # #    轉向成本：yaw command 平方項
        # reward -= 0.03 * (action[1]**2)

        # 4) 終端狀態覆寫
        # collision
        if lmin < r_usv:
            self.termination = True
            reward = -100.0

        # 到達目標
        if dist < self.info['goal_range']:
            self.termination = True
            reward = +100.0

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
            self.info['reset_range']*random.uniform(0.6, 1),
            0.0,
            random.uniform(-np.pi, np.pi)
            )
        
    def _clock_cb(self, msg: Clock):
        # update our copy of sim-time
        self.info['clock'] = msg.clock