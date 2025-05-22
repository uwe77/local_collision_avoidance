import rospy
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from geometry_msgs.msg import Vector3
from std_srvs.srv import Empty as EmptySrv


class GazeboROSConnector:
    def __init__(self):
        self.pause_service = rospy.ServiceProxy("/gazebo/pause_physics", EmptySrv)
        self.unpause_service = rospy.ServiceProxy("/gazebo/unpause_physics", EmptySrv)
        self.set_physics_service = rospy.ServiceProxy("/gazebo/set_physics_properties", SetPhysicsProperties)

    def pause(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause_service()

    def unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause_service()

    def set_physic_properties(self):
        rospy.wait_for_service("/gazebo/set_physics_properties")
        req = SetPhysicsPropertiesRequest()
        req.time_step = 0.001
        req.max_update_rate = 10000
        req.gravity = Vector3(x=0.0, y=0.0, z=-9.81)
        ode_config = ODEPhysics()
        ode_config.auto_disable_bodies = False
        ode_config.sor_pgs_precon_iters = 0
        ode_config.sor_pgs_iters = 50
        ode_config.sor_pgs_w = 1.3
        ode_config.sor_pgs_rms_error_tol = 0.0
        ode_config.contact_surface_layer = 0.001
        ode_config.contact_max_correcting_vel = 100.0
        ode_config.cfm = 0.0
        ode_config.erp = 0.2
        ode_config.max_contacts = 20
        req.ode_config = ode_config
        self.set_physics_service(req)
