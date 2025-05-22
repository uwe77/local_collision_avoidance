import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import rospy
import time
import numpy as np
import math
import random, sys, os
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64, Header
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Wrench, Point
from gazebo_msgs.srv import ApplyBodyWrench


class JSMultipleMissionsV0(gym.Env):

    def __init__(self):
        # params
        self.gym_params = {
            'missions': ['nav', 'dp', 'collision'],
            'mission': None,
            'goal_range': 30.0,
            'reset': False,
            'goal_pose': np.array([0,0,0]), # [x, y, yaw]
            'active_range': 100,
            'contact': False,
            'current_step': 0,
            'total_step': 0,
            'max_steps': 4096,
            'state': None,
            'reward': 0,
            'termination': False,
            'truncation': False,
            'info': None,
            'max_reward': 1.0,
            'epi': 0,
        }
        self.veh_params = {
            'max_action': 1.0,
            'action_clip_range': 1e-3,
            'max_laser_dis': 50.0,
            'max_track_dis': 30.0,
            'max_vel': np.inf,
            'laser_msg': None,
            'laser_stack': None,
            'track_stack': None,
            'vel_stack': None,
            'last_pose': None, # [x, y, yaw]
            'last_time': None,
            'last_goal_dist': None,
            'last_goal_angle': None,
            'time_diff': 0,
            'action': None,
        }
        # ROS
        rospy.init_node('js_multiple_missions_v0', anonymous=True)
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')

        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy(
            '/gazebo/unpause_physics', Empty)
        self.apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # publisher subscriber
        self.pub_twist = rospy.Publisher('/js/cmd_vel', Twist, queue_size=1)
        self.sub_contact = rospy.Subscriber('/js/sensors/collision', ContactsState, self.cb_contact, queue_size = 1)
        self.sub_laser = rospy.Subscriber('/js/gazebo/scan', LaserScan, self.cb_laser, queue_size=1)
        # unpause physics
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.pause_gym(False)

        # set real time factor
        os.system('gz physics -u 0')

        # gym
        # state info
        self.gym_params['info'] = {'laser_shape': (4, 241),
                     'track_shape': (10, 3),
                     'vel_shape': (10, 1),
                     'action_shape': (2,)}
        
        self.observation_space = gym.spaces.Dict({
            "laser": gym.spaces.Box(low=0, high=self.veh_params['max_laser_dis'], 
                                    shape=self.gym_params['info']['laser_shape'], dtype=np.float64),
            "track": gym.spaces.Box(low=-self.veh_params['max_track_dis'], high=self.veh_params['max_track_dis'], 
                                    shape=(self.gym_params['info']['track_shape'][0]*self.gym_params['info']['track_shape'][1],), dtype=np.float64),
            "vel": gym.spaces.Box(low=-self.veh_params['max_vel'], high=self.veh_params['max_vel'],
                                  shape=(self.gym_params['info']['vel_shape'][0]*self.gym_params['info']['vel_shape'][1],), dtype=np.float64)
        })
        self.action_space = spaces.Box(low=np.array(
            [-self.veh_params['max_action'], -self.veh_params['max_action']]), 
            high=np.array([self.veh_params['max_action'], self.veh_params['max_action']]), 
            shape=self.gym_params['info']['action_shape'], dtype=np.float32)
        
        print("gym env: js-multiple-missions-v0")
        print("obs_dim: ", self.observation_space)
        print("act_dim: ", self.action_space.shape)

        self.polyform = obstacle('polyform_red', np.array([-1000, 0, 0.1]))
        

    def step(self, action):
        self.pause_gym(False)
        self.gym_params['current_step'] += 1

        action[0] = 0 if abs(action[0]) < self.veh_params['action_clip_range'] else action[0]
        action[1] = 0 if abs(action[1]) < self.veh_params['action_clip_range'] else action[1]
        self.veh_params['action'] = action
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.angular.z = action[1]
        self.pub_twist.publish(cmd_vel)

        if self.gym_params['current_step'] % 200 == 1:
            self.gym_params['waves_force'] = [np.random.choice([-1, 1]) * np.random.normal(5, 100),
                                              np.random.choice([-1, 1]) * np.random.normal(5, 100)]

        self.apply_force_veh(self.gym_params['waves_force'][0], self.gym_params['waves_force'][1])

        state = self.get_observation()

        goal_dist = np.linalg.norm(self.gym_params['goal_pose'][:2] - self.veh_params['last_pose'][:2])
        goal_angle = self.gym_params['goal_pose'][2] - self.veh_params['last_pose'][2]
        
        if goal_angle >= np.pi:
            goal_angle -= 2*np.pi
        elif goal_angle <= -np.pi:
            goal_angle += 2*np.pi

        if self.veh_params['last_goal_dist'] is None:
            self.veh_params['last_goal_dist'] = goal_dist
        if self.veh_params['last_goal_angle'] is None:
            self.veh_params['last_goal_angle'] = goal_angle

        reward = self.get_reward()
        self.veh_params['last_goal_dist'] = goal_dist
        self.veh_params['last_goal_angle'] = goal_angle

        if self.gym_params['current_step'] >= self.gym_params['max_steps']:
            self.gym_params['truncation'] = True
        if self.veh_params['last_goal_dist'] > self.gym_params['active_range'] + 5:
            self.gym_params['termination'] = True
            self.gym_params['mission'] = 'OutOfBound'
            
        goal_state = f" \033[33m{self.gym_params['mission']}\033[0m"
        if self.veh_params['last_goal_dist'] < 4 and np.absolute(self.veh_params['last_goal_angle']) < 0.25:
            goal_state = "\033[33mgoal!!       \033[0m"

        output = "\rstep:{:4d}, force on veh: [x:{}, y:{}], reward:{}, state:{}".format(
            self.gym_params['current_step'],
            " {:4.2f}".format(self.gym_params['waves_force'][0]) if self.gym_params['waves_force'][0] >= 0 else "{:4.2f}".format(self.gym_params['waves_force'][0]),
            " {:4.2f}".format(self.gym_params['waves_force'][1]) if self.gym_params['waves_force'][1] >= 0 else "{:4.2f}".format(self.gym_params['waves_force'][1]),
            " {:4.2f}".format(reward) if reward >= 0 else "{:4.2f}".format(reward),
            goal_state
        )
        reward /= self.gym_params['max_steps']
        sys.stdout.write(output)
        sys.stdout.flush()

        self.pause_gym(True)
        return state, reward, self.gym_params['termination'], self.gym_params['truncation'], self.gym_params['info']

    def reset(self, seed=None, options=None):
        print()
        self.pause_gym(False)
        ########################### reset gym params ###########################
        self.gym_params['current_step'] = 0
        self.gym_params['termination'] = False
        self.gym_params['truncation'] = False
        self.gym_params['reset'] = True
        self.gym_params['contact'] = False

        self.veh_params['laser_stack'] = None
        self.veh_params['track_stack'] = None
        self.veh_params['vel_stack'] = None
        self.veh_params['last_pose'] = None
        self.veh_params['last_time'] = None

        ########################### reset goal pose ###########################
        self.gym_params['goal_pose'] = np.array([0, 0, random.uniform(-np.pi, np.pi)])
        goal_pose = PoseStamped()
        goal_pose.header = Header()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = self.gym_params['goal_pose'][0]
        goal_pose.pose.position.y = self.gym_params['goal_pose'][1]
        goal_pose.pose.position.z = 0
        goal_pose.pose.orientation.x = 0
        goal_pose.pose.orientation.y = 0
        goal_pose.pose.orientation.z = np.sin(self.gym_params['goal_pose'][2]/2)
        goal_pose.pose.orientation.w = np.cos(self.gym_params['goal_pose'][2]/2)
        self.pub_goal.publish(goal_pose)

        ########################### reset veh pose ############################
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            radius = np.random.uniform(10, self.gym_params['active_range'])
            angle = np.random.uniform(-np.pi, np.pi)
            state_msg = ModelState()
            state_msg.model_name = 'js'
            state_msg.pose.position.x = np.cos(angle)*radius
            state_msg.pose.position.y = np.sin(angle)*radius
            state_msg.pose.position.z = 0.0
            angle = np.random.uniform(-np.pi, np.pi)
            r = R.from_euler('z', angle)
            quat = r.as_quat()
            state_msg.pose.orientation.x = quat[0]
            state_msg.pose.orientation.y = quat[1]
            state_msg.pose.orientation.z = quat[2]
            state_msg.pose.orientation.w = quat[3]
            self.reset_model(state_msg)
        except(rospy.ServiceException) as e:
            print(e)

        state = self.get_observation()
        self.pause_gym(True)

        return state, self.gym_params['info']
    
    def nav_reward(self):
        goal_diff = self.gym_params['goal_pose'][:2] - self.veh_params['last_pose'][:2]
        goal_dist = np.linalg.norm(goal_diff)
        goal_diff_ang = np.arctan2(goal_diff[1], goal_diff[0]) - self.veh_params['last_pose'][2]
        goal_diff_ang = (goal_diff_ang + np.pi) % (2 * np.pi) - np.pi

        dist_reward = np.clip(2 - goal_dist / self.gym_params['goal_range'], 0, 1)
        angle_reward = np.clip(1 - abs(goal_diff_ang) / (np.pi / 2), 0, 1)
        action_penalty = -self.veh_params['action'][0] if self.veh_params['action'][0] < 0 else 0
        vel_penalty = min(1, abs(self.veh_params['vel_stack'][-1]) / 5.0) if goal_dist <= self.veh_params['max_track_dis'] else 0 # 5.0 is max velocity
        # progress = np.clip(self.veh_params['last_goal_dist'] - goal_dist, -1, 1)
        progress = -1 if self.veh_params['last_goal_dist'] - goal_dist < 0.01 else 1
        goal_reward = 1 if goal_dist < 0.5 else 0
        if goal_reward == 1:
            self.gym_params['termination'] = True
        # reward = (
        #     0.8 * dist_reward * angle_reward
        #     + 0.2 * progress 
        #     + 2.0 * action_penalty
        # )
        reward = (
            goal_reward 
            + 0.01 * progress
            - 0.01 * action_penalty
            - 0.01 * vel_penalty
        )
        return float(reward)

    def dp_reward(self, dp_range=4.0):
        goal_diff = self.gym_params['goal_pose'][:2] - self.veh_params['last_pose'][:2]
        goal_dist = np.linalg.norm(goal_diff)
        goal_ang = self.gym_params['goal_pose'][2] - self.veh_params['last_pose'][2]
        goal_ang = (goal_ang + np.pi) % (2 * np.pi) - np.pi
        
        dist_reward = max(0, 1.0 - goal_dist / dp_range)
        # angle_reward = max(0, 1.0 - abs(goal_ang) / (np.pi / dp_range))
        angle_reward = math.exp(-4*(goal_ang/math.pi)**2)
        action_reward = self.veh_params['action'][0] if self.veh_params['action'][0] < 0 else 0
        vel_penalty = min(1, abs(self.veh_params['vel_stack'][-1]) / 5.0) # 5.0 is max velocity
        
        reward = (
            1.5-dp_range/self.gym_params['goal_range'] # Bonus for keeping in DP
            + angle_reward * dist_reward
            + 0.5 * action_reward
            - 1.0 * vel_penalty
        )
        return float(reward)

    def collision_reward(self, safe_dist = 10.0):
        most_close_dist = min(self.veh_params['laser_stack'][-1])
        if most_close_dist-safe_dist < 0 or self.gym_params['contact'] is True:
            reward = -3
            self.gym_params['termination'] = True
            self.gym_params['mission'] = 'collision'
        else:
            reward = np.clip((most_close_dist-self.veh_params['max_laser_dis'])/self.veh_params['max_laser_dis'], -1, 0)
        return float(reward)

    def get_reward(self):
        dp_range = 10.0
        safe_dist = 10.0
        nav_r = self.nav_reward()
        dp_r = self.dp_reward(dp_range)
        coll_r = self.collision_reward(safe_dist)

        self.gym_params['mission'] = 'nav'
        reward = nav_r
        # if self.gym_params['mission'] == 'collision':
        #     reward = coll_r
        #     self.gym_params['reward'] = reward
        #     return reward
        
        # if self.veh_params['last_goal_dist'] > self.gym_params['goal_range']:
        #     self.gym_params['mission'] = 'nav'
        #     reward = nav_r
        # elif self.veh_params['last_goal_dist'] > dp_range:
        #     self.gym_params['mission'] = 'nav'  # Transition zone
        #     weight = (self.veh_params['last_goal_dist'] - dp_range) / (self.gym_params['goal_range'] - dp_range)  # Smooth transition
        #     reward = weight * nav_r + (1 - weight) * dp_r
        # else:
        #     self.gym_params['mission'] = 'dp'
        #     reward = dp_r
        reward += coll_r
        
        self.gym_params['reward'] = reward
        return reward

    def get_observation(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model('js', '')
        except (rospy.ServiceException) as e:
            print(e)
        yaw = R.from_quat([agent.pose.orientation.x, agent.pose.orientation.y, 
                           agent.pose.orientation.z, agent.pose.orientation.w]).as_euler('zyx')[0]
        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, yaw])
        time = rospy.get_rostime()

        velocity = 0
        if self.veh_params['last_time'] is not None and self.veh_params['last_pose'] is not None and self.gym_params['reset'] is False:
            self.veh_params['time_diff'] = (time.to_nsec()-self.veh_params['last_time'].to_nsec())/1000000000
            if self.veh_params['time_diff'] == 0:
                self.veh_params['time_diff'] = 0.1
            distance = math.sqrt((new_pos[0]-self.veh_params['last_pose'][0])**2+(new_pos[1]-self.veh_params['last_pose'][1])**2)
            velocity = distance/self.veh_params['time_diff']
        
        self.veh_params['last_time'] = time
        self.veh_params['last_pose'] = new_pos

        self.gym_params['reset'] = False
        # caculate angle diff
        diff = self.gym_params['goal_pose'][:2] - self.veh_params['last_pose'][:2]
        angle = np.arctan2(diff[1], diff[0]) - self.veh_params['last_pose'][2]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        diff = np.array(diff)
        diff_norm = np.linalg.norm(diff)
        diff_clipped = np.array([np.cos(angle), np.sin(angle)]) * np.clip(diff_norm, 0, self.veh_params['max_track_dis'])
        diff = diff_clipped

        diff_angle = self.gym_params['goal_pose'][2] - self.veh_params['last_pose'][2]
        if diff_angle >= np.pi:
            diff_angle -= 2*np.pi
        elif diff_angle <= -np.pi:
            diff_angle += 2*np.pi

        track_pos = np.append(diff, diff_angle)

        if self.veh_params['track_stack'] is None:
            self.veh_params['track_stack'] = np.tile(track_pos, (self.gym_params['info']['track_shape'][0], 1))
        else:
            self.veh_params['track_stack'][:-1] = self.veh_params['track_stack'][1:]
            self.veh_params['track_stack'][-1] = track_pos
        
        velocity = np.array([velocity])
        if self.veh_params['vel_stack'] is None:
            self.veh_params['vel_stack'] = np.tile(float(velocity), self.gym_params['info']['vel_shape'])
        else:
            self.veh_params['vel_stack'][:-1] = self.veh_params['vel_stack'][1:]
            self.veh_params['vel_stack'][-1] = float(velocity)

        scan = self.scan_once()

        if self.veh_params['laser_stack'] is None:
            self.veh_params['laser_stack'] = np.tile(scan, (self.gym_params['info']['laser_shape'][0], 1))
        else:
            self.veh_params['laser_stack'][:-1] = self.veh_params['laser_stack'][1:]
            self.veh_params['laser_stack'][-1] = scan
        
        laser = self.veh_params['laser_stack']
        track = self.veh_params['track_stack'].reshape(-1)
        vel = self.veh_params['vel_stack'].reshape(-1)
        self.gym_params['state'] = {"laser": laser, "track": track, "vel": vel}

        return self.gym_params['state']

    def pause_gym(self, pause=True):
        try:
            if pause:
                self.pause_physics()
            else:
                self.unpause_physics()
        except rospy.ServiceException as e:
            print(e)

    def scan_once(self):
        if self.veh_params['laser_msg'] is None:
            return np.full(self.gym_params['info']['laser_shape'][1], self.veh_params['max_laser_dis'])
        ranges = np.clip(np.array(self.veh_params['laser_msg']), 0, self.veh_params['max_laser_dis'])

        return ranges

    def cb_contact(self, msg):
        self.gym_params['contact'] = False
        if msg.states != []: 
            self.gym_params['contact'] = True
    
    def cb_laser(self, msg):
        self.veh_params['laser_msg'] = msg.ranges

    def apply_force_veh(self, force_x, force_y):
        force = Wrench()
        force.force = Vector3(x=force_x, y=force_y, z=0.0)
        force.torque = Vector3(x=0.0, y=0.0, z=0.0)

        body_name_str = "js" + "::" + "js/base_link"
        self.apply_wrench(body_name = body_name_str,
                        reference_frame = 'world', 
                        reference_point = Point(x=0.0, y=0.0, z=0.0),
                        wrench = force,
                        start_time = rospy.Time(0.0),
                        duration = rospy.Duration(0.04))
        
    def close(self):
        rospy.signal_shutdown('WTF')


class obstacle():
    def __init__(self, name=None, init_pose=np.array([-100, 0, 0.1])) -> None:
        self.name = name
        self.pose = np.zeros(3)
        self.force = np.zeros(3)
        self.init_pose = init_pose
    
    def reset(self):
        self.set_pose(self.init_pose)
        # r = np.random.uniform(0, np.pi)
        # dir = r + np.random.normal(0, math.pi/60) + np.pi
        # dir = dir % (2*np.pi)
        # self.force = np.random.uniform(80, 100)*np.array([math.cos(dir), math.sin(dir), 0])

    def apply_force(self, force=np.zeros(3), torque=np.zeros(3)):
        f = Wrench()
        f.force = Vector3(x=force[0], y=force[1], z=force[2])
        f.torque = Vector3(x=torque[0], y=torque[1], z=torque[2])
        body_name_str = f"{self.name}" + "::" + "base_link"
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        apply_wrench(body_name = body_name_str,
                        reference_frame = 'world', 
                        reference_point = Point(x=0.0, y=0.0, z=0.0),
                        wrench = f,
                        start_time = rospy.Time(0.0),
                        duration = rospy.Duration(0.04))
        
    def update_pose(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        get_model = rospy.ServiceProxy(
            '/gazebo/get_model_state', GetModelState)
        try:
            agent = get_model(self.name, '')
        except (rospy.ServiceException) as e:
            print(e)
        self.pose = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])
        
    def set_pose(self, pose=np.zeros(3)):
        def get_initail_obstacle_state(n, pose):
            state_msg = ModelState()
            state_msg.model_name = n
            state_msg.pose.position.x = pose[0]
            state_msg.pose.position.y = pose[1]
            state_msg.pose.position.z = pose[2]
            r = R.from_euler('z', 0)
            quat = r.as_quat()
            state_msg.pose.orientation.x = quat[0]
            state_msg.pose.orientation.y = quat[1]
            state_msg.pose.orientation.z = quat[2]
            state_msg.pose.orientation.w = quat[3]
            return state_msg
        reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        reset_model(get_initail_obstacle_state(self.name, pose))

    def apply_task(self):
        self.apply_force(self.force)