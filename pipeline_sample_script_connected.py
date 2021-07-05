import os

import gym
import numpy as np
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from shadow_hand_contact_sensor.msg import shadow_hand_contact_force
from shadow_hand_contact_sensor.msg import shadowhand_link_pose
from std_msgs.msg import Float64
from std_srvs.srv import Empty

from agent.ppo_agent import PPOAgent
from common.senses import Sensation
from common.wrappers import TransformationWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# BUILD DUMMY MODEL
agent = PPOAgent.from_agent_state(1624884304, "best")
print(f"Agent {agent.agent_id} successfully loaded.")
model, _, _ = agent.build_models(agent.joint.get_weights(), batch_size=1, sequence_length=1)


# SIMULATION DUMMY

FINGERTIP_SITE_NAMES = [  # rostopic names of distal bones (in place of the fingertips)
    'rh_ffdistal',
    'rh_mfdistal',
    'rh_rfdistal',
    'rh_lfdistal',
    'rh_thdistal',
]


class NRPDummy:
    """Dummy class representing the role of the NRP/Gazebo simulation."""

    def __init__(self, dt: float):
        """Initialize simulator.

        Args:
            dt (object):    we should be able to control the temporal difference between timesteps; however it is not so
                            important that we can do this in a continuous manner; for reference: MuJoCo has a predefined
                            length of a timestep and we control how many of these timesteps correspond to a timestep
                            from the perspective of our model. E.g.: If we say one timestep from the model's perspective
                            corresponds to 20 steps in MuJoCo, then the same action provided to this class will be
                            applied for 20 steps in MuJoCo.
        """
        self.dt = dt

        # Shadow Hand Joint names
        self.shadow_hand_joint_dic = {
            "wrist": ["rh_WRJ1", "rh_WRJ0"],
            "index": ["rh_FFJ3", "rh_FFJ2", "rh_FFJ1", "rh_FFJ0"],
            "middle": ["rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_MFJ0"],
            "ring": ["rh_RFJ3", "rh_RFJ2", "rh_RFJ1", "rh_RFJ0"],
            "pinky": ["rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1", "rh_LFJ0"],
            "thumb": ["rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1", "rh_THJ0"]
        }

        # self.finger_control_ros_service = [rospy.ServiceProxy("/shadowhand_motor/" + joint_name + "/set_target", SetJointStates) for finger_name in self.shadow_hand_joint_dic.keys() for joint_name in self.shadow_hand_joint_dic[finger_name]]
        self.finger_control_ros_service = [
            rospy.Publisher("/shadowhand_motor/" + joint_name + "/cmd_pos", Float64, queue_size=10) for finger_name in
            self.shadow_hand_joint_dic.keys() for joint_name in self.shadow_hand_joint_dic[finger_name]]

        self.test_counter = 0

        self.vision = []

    def apply_action(self, action: np.ndarray):
        """Dummy method that acts in place of applying an action in the simulation. When we provide an action here, the
        simulation should try to apply this action for the duration of a timestep. AFTER THAT the simulation should
        pause and wait for the next action, no matter whether or not the action was finished. If for instance the model
        gives a motor command that, in the timespan of a single timestep, cannot be achieved, then thats fine. The model
        should then have learned to continue providing the same motor command in future timesteps until the desired
        state is reached or until it desires to give a new target."""

        """ Adapted behaviour from Mujoco:
            The first finger joint 1 and joint 0 are coupled 
            The middle finger joint 1 and joint 0 are coupled 
            The ring finger joint 1 and joint 0 are coupled 
            The little finger joint 1 and joint 0 are coupled 

        'robot0:A_WRJ1', 'robot0:A_WRJ0', 'robot0:A_FFJ3', 'robot0:A_FFJ2', 'robot0:A_FFJ1', 
        'robot0:A_MFJ3', 'robot0:A_MFJ2', 'robot0:A_MFJ1', 
        'robot0:A_RFJ3', 'robot0:A_RFJ2', 'robot0:A_RFJ1', 
        'robot0:A_LFJ4', 'robot0:A_LFJ3', 'robot0:A_LFJ2', 'robot0:A_LFJ1', 
        'robot0:A_THJ4', 'robot0:A_THJ3', 'robot0:A_THJ2', 'robot0:A_THJ1', 'robot0:A_THJ0'
       """
        rate = rospy.Rate(100)
        action = np.insert(action, 5, action[4])
        action = np.insert(action, 9, action[8])
        action = np.insert(action, 13, action[12])
        action = np.insert(action, 18, action[17])

        # Erdi test starts here
        # action = np.zeros(24)
        # action[2:] = np.random.uniform(0, 1.57,size=action[2:].shape)
        # self.test_counter +=1
        # print(self.test_counter)

        # if (self.test_counter == 1) :
        #     action[7] = 0.5
        # elif (self.test_counter == 2) :
        #     action[7] = 1.57
        # elif (self.test_counter ==3):
        #     action[7] = 0
        # else:
        #     print("There is nothing to test counter")

        # Erdi test ends here

        for i, act in enumerate(self.finger_control_ros_service):
            act.publish(action[i])
            while (act.get_num_connections() == 0):
                rate.sleep()

    def get_state(self):
        """Dummy method that acts in place of reading the actual data from the NRP."""
        # print("Get state")
        # time.sleep(0.3)
        # rospy.Subscriber("/shadow_hand/camera/image_raw", Image, self.camera_callback)
        camera_pixels = rospy.wait_for_message('/shadow_hand/camera/image_raw', Image)
        vision = np.frombuffer(np.array(camera_pixels.data), np.uint8).reshape(200, 200, 3)

        shadow_hand_contact_data = rospy.wait_for_message('/shadow_hand_visual_tag_contact_sensor', shadow_hand_contact_force)
        touch_sensor_values = np.array([contact_force.z for contact_force in shadow_hand_contact_data.force_array])
        somatosensation = touch_sensor_values

        joint_state_data = rospy.wait_for_message("/shadowhand_motor/joint_states", JointState)
        fingertip_position_data = rospy.wait_for_message("/shadowhand_motor/link_positions", shadowhand_link_pose)

        print(fingertip_position_data.pose_array.poses[0])
        print(type(fingertip_position_data.pose_array.poses[0]))
        print(dir(fingertip_position_data.pose_array.poses[0]))
        print("\n")

        joint_pos = np.array(joint_state_data.position[0:-1])
        joint_vel = np.array(joint_state_data.velocity[0:-1])
        fingertip_position = np.random.randn(15)
        proprioception = np.concatenate((joint_pos, joint_vel, fingertip_position), axis=0)

        return {
            "vision": None,
            "somatosensation": somatosensation,  # touch sensor readings
            "proprioception": proprioception,  # joint positions and velocities
        }

    def set_state(self):
        """Dummy method to set the state of the simulation (hand position, velocities, etc.)"""
        pass

    def reset(self):
        """Dummy method that acts in playe of a full reset of the simulation to initial conditions."""
        # initial_positions = [-0.16514339750464327, -0.31973286565062153,0.14340512546557435,0.32028208333591573,0.7126053607727917,0.6705281001412586,0.000246444303701037,0.3152655251085491,0.7659800313729842,0.7323156897425923,0.00038520700007378114,0.36743546201985233,0.7119514095008576,0.6699446327514138,0.0525442258033891,-0.13615534724474673,0.39872030433433003,0.7415570009679252,0.704096378652974,0.003673823825070126,0.5506291436028695, -0.014515151997119306,-0.0015229223564485414,-0.7894883021600622]
        # self.apply_action(initial_positions)
        gazebo_reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        gazebo_reset_world()
        gazebo_reset_sim = rospy.ServiceProxy("/gazebo/reset_sim", Empty)
        gazebo_reset_sim()


# DUMMY ENVIRONMENT
class NRPEnv(gym.Env):
    """Dummy for the environment wrapper around the NRP simulation that we would preferrably construct based on
    openai gym standards"""

    def __init__(self):
        self.sim = NRPDummy(dt=0.002)
        self.goal = np.random.random((15,))

        self.action_space = gym.spaces.Box(-1, 1, shape=(20,), dtype=np.float32)

    def _get_obs(self):
        sim_state = self.sim.get_state()

        observation = {
            "observation": Sensation(
                **sim_state.copy(),
                goal=self.goal)
        }

        return observation

    def get_fingertip_positions(self):
        """Get positions of all fingertips in euclidean space. Each position is encoded by three floating point numbers,
        as such the output is a 15-D numpy array."""
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = np.random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        self.current_target_finger = finger_idx

        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += np.random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if np.random.uniform() < 0.1:
            goal = self.initial_goal.copy()

        return goal.flatten()

    def step(self, action: np.ndarray):
        # print(action.shape)
        assert len(action) == 20, "Actions must be 20-dimensional vectors."

        # restrict action based on allowed range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # here, the simulation needs to make a step by following the given action till the next timestep
        self.sim.apply_action(action)

        # after applying the action in simulation, we can read the current state, calculate rewards, etc.
        observation = self._get_obs()
        reward = np.random.normal(0, 1)
        done = False
        info = {}

        return observation, reward, done, info

    def reset(self):
        """Reset simulation and get initial state/observation."""
        self.sim.reset()
        return self._get_obs()

    def render(self, mode='human'):
        pass


rospy.init_node('pipe', anonymous=True)

# THE LOOP
env = TransformationWrapper(NRPEnv(), transformers=agent.env.transformers)
state = env.reset()
done = False
while not done:
    state.inject_leading_dims(time=True)
    probabilities = np.squeeze(model(state.dict(), training=False))
    action = agent.distribution.act_deterministic(*probabilities)

    state, _, done, _ = env.step(action)

# for i in range(100):

#     inject_leading_dims(state)
#     next_action = np.squeeze(model(state))
#     state, _, done, _ = env.step(next_action)
#     print(state['proprioception'].reshape(-1)[11])
