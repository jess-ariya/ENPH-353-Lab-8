
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected


    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # cv2.imshow("raw", cv_image)

        # rate = rospy.Rate(2)
        # move = Twist()

        shape = cv_image.shape
        h = shape[0] #240
        w = shape[1] #320

        NUM_BINS = 3
        state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        done = False

        # TODO: Analyze the cv_image and compute the state array and
        # episode termination condition.
        #
        # The state array is a list of 10 elements indicating where in the
        # image the line is:
        # i.e.
        #    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] indicates line is on the left
        #    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] indicates line is in the center
        #
        # The episode termination condition should be triggered when the line
        # is not detected for more than 30 frames. In this case set the done
        # variable to True.
        #
        # You can use the self.timeout variable to keep track of which frames
        # have no line detected.

        # Convert the image to grayscale and apply blur
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Make binary
        threshold = 115 #@param {type: "slider", min : 0, max : 255}
        _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        # Invert
        binary = cv2.bitwise_not(binary)

        # Get indices of white pixels (Road)
        white_pixels = binary.nonzero()

        # Define the number of vertical sections to split the image into
        num_sections = 10

        # Determine the height of each section based on the image height
        # section_height = int(cv_image.shape[0] / num_sections)

        # Determine the width of each section based on the image height
        section_width = int(w / num_sections)

        # Initialize the state array
        #state_array = [0] * num_sections

        if white_pixels[0].size > 0:
            # Compute the centroid of the white pixels
            centroid_x = int(white_pixels[1].mean())
            centroid_y = int(white_pixels[0].mean())
            #print(centroid_x, centroid_y)

            # Compute the bottom half of the image
            bottom_half = int (h / 2) + 5
            bottom_half_pixels = white_pixels[0][white_pixels[0] > bottom_half]
            if bottom_half_pixels.size > 0:
                centroid_y = int(bottom_half_pixels.mean())

            # Determine which section of the image the centroid is in
            section_width = int(cv_image.shape[1] / 10)
            section_idx = int(centroid_x / section_width)
            state[section_idx] = 1

            # Draw a red circle on the center of the line
            center = (centroid_x, centroid_y)
            cv2.circle(cv_image, center, 10, (0, 0, 255), 20)
            #cv2.circle(binary, center, 10, (0,0,255), -1)
            # if centroid_x < w/2:
            #     rate = rospy.Rate(2)
            #     move = Twist()
            #     move.linear.x = 0.1
            #     move.angular.z = 0.1
            # else:
            #     rate = rospy.Rate(2)
            #     move = Twist()
            #     move.linear.x = 0.1
            #     move.angular.z = -0.1

        # Update episode termination condition
        #if white_pixels[0].size == 0:
        if white_pixels[0].size == 0:
            self.timeout += 1
        else:
            self.timeout = 0

        done = (self.timeout > 30)

        # Display image for debugging purposes
        cv2.imshow("Image", cv_image)
        #cv2.imshow("Image", binary)
        cv2.waitKey(2)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action == 0:  # FORWARD
            vel_cmd.linear.x = 0.4
            vel_cmd.angular.z = 0.01
        elif action == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.3
        elif action == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.3

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 2
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
