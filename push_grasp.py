#!/usr/bin/env python
# Imports
import rospy
import sys
# from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray, Pose, Twist
from visualization_msgs.msg import MarkerArray, Marker
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
import numpy as np
import tf
from tf import TransformListener
import copy
from time import sleep
import roslaunch
import math
import pdb
from cv_bridge import CvBridge

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState, RobotTrajectory, CollisionObject
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, \
    _Robotiq2FGripper_robot_input as inputMsg
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy

from pyquaternion import Quaternion

import pdb
from enum import Enum

from controller_manager_msgs.srv import SwitchController, LoadController


# Enums
class State(Enum):
    FIRST_GRAB = 1
    SECOND_GRAB = 2
    FINISHED = 3

GRAB_THRESHOLD = 8  # Newtons
RELEASE_THRESHOLD = 8  # Newtons


# Grasp Class
class GraspExecutor:
    # Initialisation
    def __init__(self):
        # Initialisation
        rospy.init_node('push_grasp', anonymous=True)

        self.tf_listener_ = TransformListener()
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        # Publisher for grasp poses
        self.pose_publisher = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)

        self.state = State.FIRST_GRAB

        self.gripper_data = 0

        # Force Sensor
        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        # Gripper
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input,
                                            self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output,
                                           queue_size=1)
        # ros::Publisher vis_pub = node_handle.advertise<visualization_msgs::Marker>( "visualization_marker", 0 )
        self.vis_pub = rospy.Publisher("vis", MarkerArray, queue_size=1)
        self.latest_force = 0.0

        # Hard-coded joint values
        # self.view_home_joints = [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155]
        self.view_home_joints = [0.07646834850311279, -0.7014802137957972, -2.008395496998922, -1.1388691107379358, 1.5221940279006958, 0.06542113423347473]
        self.move_home_joints = [0.0030537303537130356, -1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]

        # Set default robot states
        self.move_home_robot_state = self.get_robot_state(self.move_home_joints)
        self.view_home_robot_state = self.get_robot_state(self.view_home_joints)

        # Hard-code corner positions
        # Start Top Right (robot perspective) and go around clockwise
        corner_1 = [-0.830, 0.225]
        corner_2 = [-0.405, 0.225]
        corner_3 = [-0.405, -0.130]
        # corner_4 = [-0.830, -0.130]
        corner_4 = [-0.870, 0.080]
        self.corner_pos_list = [corner_1, corner_2, corner_3, corner_4]

        # Keep sleep here to allow scene to load
        rospy.sleep(3)

        # Add front wall
        self.front_wall_name = "front_wall"
        # self.add_front_wall()

        # Add right wall
        self.right_wall_name = "right_wall"
        # self.add_right_wall()

        # Add left wall
        self.left_wall_name = "left_wall"
        # self.add_left_wall()

        # Add left wall
        self.back_wall_name = "back_wall"
        # self.add_back_wall()

        # Add roof
        self.roof_name = "roof"
        self.add_roof()

        # self.switch_controller_server()

        rospy.wait_for_service('/controller_manager/switch_controller')
        
        self.switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.start_controllers = ["twist_controller"]
        self.stop_controllers = ["scaled_pos_joint_traj_controller"]
        self.strictness = 1
        self.start_asap = True
        self.timeout = 5.0

        rospy.wait_for_service('/controller_manager/load_controller')
        load_controller = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        load_controllers_name = "twist_controller"

        self.twist_pub = rospy.Publisher('/twist_controller/command', Twist, queue_size=1)
        # Load twist controller
        ok = load_controller(load_controllers_name)

        # RGB Image
        self.rgb_sub = rospy.Subscriber('/realsense/rgb', sensor_msgs/Image, self.rgb_callback)

    def rgb_callback(self, image):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

    
    def turn_on_twist_controller(self):
        try:
            # Turn on twist controller
            ok = self.switch_controller(self.start_controllers, self.stop_controllers, self.strictness, self.start_asap, self.timeout)
            rospy.loginfo("Twist controller activated")
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def turn_on_joint_traj_controller(self):
        try:
            # Turn on twist controller
            ok = self.switch_controller(self.stop_controllers, self.start_controllers, self.strictness, self.start_asap, self.timeout)
            rospy.loginfo("Joint traj controller activated")
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))

    def add_front_wall(self):
        # Add front wall
        front_wall_pose = PoseStamped()
        front_wall_pose.header.frame_id = "base_link"
        front_wall_pose.pose.orientation.w = 1.0
        front_wall_pose.pose.position.x = -0.405
        front_wall_pose.pose.position.y = 0.0475
        front_wall_pose.pose.position.z = 0.05

        self.scene.add_box(self.front_wall_name, front_wall_pose, size=(0.02, 0.355, 0.1))

    def add_right_wall(self):
        # Add right wall
        right_wall_pose = PoseStamped()
        right_wall_pose.header.frame_id = "base_link"
        right_wall_pose.pose.orientation.w = 1.0
        right_wall_pose.pose.position.x = -0.6175
        right_wall_pose.pose.position.y = 0.225
        right_wall_pose.pose.position.z = 0.05

        self.scene.add_box(self.right_wall_name, right_wall_pose, size=(0.42, 0.02, 0.1))

    def add_back_wall(self):
        # Add back wall
        back_wall_pose = PoseStamped()
        back_wall_pose.header.frame_id = "base_link"
        back_wall_pose.pose.orientation.w = 1.0
        back_wall_pose.pose.position.x = -0.830
        back_wall_pose.pose.position.y = 0.0475
        back_wall_pose.pose.position.z = 0.05

        self.scene.add_box(self.back_wall_name, back_wall_pose, size=(0.02, 0.355, 0.1))

    def add_left_wall(self):
        # Add left wall
        left_wall_pose = PoseStamped()
        left_wall_pose.header.frame_id = "base_link"
        left_wall_pose.pose.orientation.w = 1.0
        left_wall_pose.pose.position.x = -0.6175
        left_wall_pose.pose.position.y = -0.130
        left_wall_pose.pose.position.z = 0.05

        self.scene.add_box(self.left_wall_name, left_wall_pose, size=(0.42, 0.02, 0.1))

    def add_roof(self):
        # Add roof
        roof_pose = PoseStamped()
        roof_pose.header.frame_id = "base_link"
        roof_pose.pose.orientation.w = 1.0
        roof_pose.pose.position.x = -0.75
        roof_pose.pose.position.y = -0.0
        roof_pose.pose.position.z = 0.67

        self.scene.add_box(self.roof_name, roof_pose, size=(2.5, 1.4, 0.01))

    def force_callback(self, wrench_msg):
        # self.latest_force = abs(wrench_msg.wrench.force.z)
        self.latest_force = abs(wrench_msg.wrench.force.y)
        # rospy.loginfo("Force: %f", self.latest_force)

    def gripper_state_callback(self, data):
        # function called when gripper data is received
        self.gripper_data = data

    def find_best_grasp(self):
        # Angle at which grasps are performed
        grasp_angle = 30
        # Initialise values
        final_grasp_pose = 0
        final_grasp_pose_offset = 0
        # Z value
        z_value = 0.06
        # Grasp pose offset distance
        offset_dist = 0.1  # 0.2
        # Grasp pose list
        poses = []

        # Listen to tf
        self.tf_listener_.waitForTransform("/orange0", "/base_link", rospy.Time(), rospy.Duration(4))
        (trans, rot) = self.tf_listener_.lookupTransform('/base_link','/orange0', rospy.Time(0))
        x_pos = trans[0]
        y_pos = trans[1]
        rospy.loginfo("Apple Found!")
        rospy.loginfo("X: %f", x_pos)
        rospy.loginfo("Y: %f", y_pos)

        # # Create pose in camera frame
        p_base = PoseStamped()
        p_base.header.frame_id = "/base_link"

        # Add position to pose
        p_base.pose.position.x = x_pos
        p_base.pose.position.y = y_pos
        # Set z to set value
        p_base.pose.position.z = z_value

        # Find nearest corner
        # nearest_corner = self.find_nearest_corner(p_base) TURN OFF ATM
        nearest_corner = 3

        # Convert positions to 2D
        corner_pos = self.corner_pos_list[nearest_corner]  # [x,y]
        grasp_pos = [p_base.pose.position.x, p_base.pose.position.y]  # [x,y]

        # Find approach angle
        x_angle, offset_pos = self.calculate_approach(grasp_pos, corner_pos, -offset_dist)

        # Find angle of gripper to the ground from hard-coded value
        y_angle = np.deg2rad(grasp_angle)

        # Print results
        # rospy.loginfo("Nearest corner: %d", nearest_corner)
        # rospy.loginfo("Corner Position - X: %f Y:%f", corner_pos[0], corner_pos[1])
        rospy.loginfo("Grasp Position - X: %f Y:%f Z:%f", grasp_pos[0], grasp_pos[1], p_base.pose.position.z)
        # rospy.loginfo("Z-angle: %f", np.rad2deg(z_angle))

        # Generate pose
        p_base = self.generate_push_pose(p_base, y_angle, x_angle)

        # Create offset pose
        p_base_offset = copy.deepcopy(p_base)
        p_base_offset.pose.position.x = offset_pos[0]
        p_base_offset.pose.position.y = offset_pos[1]

        # Add pose to pose list
        poses.append(copy.deepcopy(p_base.pose))

        # Create pose array and add valid grasps
        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"

        # Publish grasp array
        self.pose_publisher.publish(posearray)

        # Set values
        valid_grasp = False
        cancel_flag = "n"

        while not valid_grasp:
            # Check path planning from home state to offset grasp pose
            self.move_group.set_start_state(self.view_home_robot_state)
            self.move_group.set_pose_target(p_base_offset)
            plan_to_offset = self.move_group.plan()

            # Clear target
            self.move_group.clear_pose_targets()

            # If plan to offset is valid
            if plan_to_offset.joint_trajectory.points:
                # Show plan to check with user
                valid_path = self.user_check_path(p_base_offset, plan=plan_to_offset)

                # If user confirms the path
                if valid_path:
                    # Find robot state of offset position
                    next_robot_state = self.get_robot_state(plan_to_offset.joint_trajectory.points[-1].positions)

                    # Create pose in the corner to move towards
                    corner_p_base = copy.deepcopy(p_base_offset.pose)
                    corner_p_base.position.x = corner_pos[0]
                    corner_p_base.position.y = corner_pos[1]

                    # Create plan from offset grasp pos to corner grasp pos
                    self.move_group.set_start_state(next_robot_state)
                    self.move_group.set_pose_target(corner_p_base)
                    plan_to_corner = self.move_group.plan()

                    # Use current poses as final grasp poses
                    final_grasp_pose = p_base
                    final_grasp_pose_offset = p_base_offset
                    rospy.loginfo("Final grasp found!")
                    # Append pose to pose array
                    poses = [poses[-1]]
                    valid_grasp = True
                else:
                    rospy.loginfo("Invalid path to offset pose")
            else:
                rospy.loginfo("Invalid path to offset pose")

            if not valid_grasp:
                cancel_flag = raw_input("Cancel? (y or n)")
                if cancel_flag == "y":
                    break

        if valid_grasp:
            # Create pose array and add valid grasps
            posearray = PoseArray()
            posearray.poses = poses
            posearray.header.frame_id = "base_link"

            # Publish grasp array
            self.pose_publisher.publish(posearray)
        else:
            # Offset value is 0
            plan_to_offset = 0

        return final_grasp_pose, final_grasp_pose_offset, plan_to_offset, plan_to_corner, corner_p_base, valid_grasp

    def find_nearest_corner(self, p_base):
        # Find the nearest corner to the grasp
        grasp_pos = [p_base.pose.position.x, p_base.pose.position.y]
        distance_list = [0, 0, 0, 0]

        # Calculate distance between the grasp point and the corners
        for i in range(len(self.corner_pos_list)):
            corner_pos = self.corner_pos_list[i]
            distance_list[i] = self.dist_two_points(grasp_pos, corner_pos)
            rospy.loginfo("Distance to corner number %d: %f", i, distance_list[i])

        # Nearest corner 
        nearest_corner = distance_list.index(min(distance_list))
        return nearest_corner

    def dist_two_points(self, p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def calculate_approach(self, start_pos, final_pos, distance):
        # Difference in x and y
        x_diff = -(final_pos[0] - start_pos[0])
        y_diff = (final_pos[1] - start_pos[1])
        # Angle of the gripper to the corner (in z-axis)
        x_angle = math.atan2(y_diff, x_diff)

        # Calculate offset position
        v = np.subtract(final_pos, start_pos)
        u = v / np.linalg.norm(v)
        new_pos = np.array(start_pos) + distance * u

        return x_angle, new_pos

    def generate_push_pose(self, current_pose, y_angle, x_angle):
        # Update pose in base frame
        y_angle = -y_angle
        z_angle = np.deg2rad(-90)
        quaternion = tf.transformations.quaternion_from_euler(x_angle, y_angle, z_angle, axes='sxyz')
        # Update values
        current_pose.pose.orientation.x = quaternion[1]
        current_pose.pose.orientation.y = quaternion[2]
        current_pose.pose.orientation.z = quaternion[3]
        current_pose.pose.orientation.w = quaternion[0]

        return current_pose

    def run_motion(self, final_grasp_pose_offset, plan_to_offset, plan_to_corner, corner_pose):
        # self.move_group.set_start_state_to_current_state()
        # self.move_to_joint_position(self.move_home_joints)
        # Move to offset pose
        self.move_to_position(final_grasp_pose_offset, plan_to_offset)
        # self.move_to_position(final_grasp_pose)

        # push grasp
        force_threshold = 42
        # Remove walls
        self.scene.remove_world_object(self.back_wall_name)
        self.scene.remove_world_object(self.left_wall_name)

        # Location of orange
        # cancel_flag = "n"
        # while True:
        #     self.tf_listener_.waitForTransform("/orange0", "/camera_link", rospy.Time(), rospy.Duration(4))
        #     (trans, rot) = self.tf_listener_.lookupTransform('/camera_link', '/orange0', rospy.Time(0))
        #     x_pos_of_obj = trans[0]
        #     rospy.loginfo("X pos of object: %f", x_pos_of_obj)
        #     cancel_flag = raw_input("Cancel? (y or n)")
        #     if cancel_flag == "y":
        #             break

        loop_flag = raw_input("Closed loop? (y or n)")
        if loop_flag == "y":
            self.closed_loop_push_grasp(corner_pose, force_threshold)
        else:
            self.vel_push_grasp(corner_pose, force_threshold)

        # self.push_grasp(corner_pose, force_threshold, plan_to_corner)

        rospy.sleep(1)

        # pose, plan = 
        self.move_to_position(*self.lift_up_pose())

        rospy.sleep(1)
        if self.gripper_data.gOBJ == 3:
            rospy.loginfo("Robot has missed/dropped object!")
            self.move_to_joint_position(self.move_home_joints)
            self.move_to_joint_position(self.view_home_joints)
        else:
            # Go to move home position using joint
            self.move_to_joint_position(self.move_home_joints)
            # self.move_to_joint_position(self.drop_object_joints)
            self.command_gripper(open_gripper_msg())
            # self.move_to_joint_position(self.move_home_joints)
            self.move_to_joint_position(self.view_home_joints)

        rospy.sleep(2)

    def move_to_position(self, grasp_pose, plan=None):
        self.move_group.set_pose_target(grasp_pose)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.execute(plan, wait=True)
        else:
            rospy.loginfo("Path cancelled")

        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def user_check_path(self, grasp_pose, plan=None):
        self.move_group.set_pose_target(grasp_pose)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y or n]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.clear_pose_targets()
            return True
        elif run_flag == "n":
            self.move_group.clear_pose_targets()
            return False

    def push_grasp(self, final_pose, force_threshold, plan_to_corner, plan=None):
        self.move_group.set_pose_target(final_pose)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.execute(plan, wait=False)
            # Check force feedback
            rospy.sleep(0.5)
            # baseline_force = 0
            while self.latest_force < force_threshold:
                # Get position of gripper
                current_pose = self.move_group.get_current_pose()
                current_pose_coords = [current_pose.pose.position.x, current_pose.pose.position.y]
                final_pose_coords = [final_pose.position.x, final_pose.position.y]
                distance_to_corner = self.dist_two_points(current_pose_coords, final_pose_coords)
                # Distance to orange
                self.tf_listener_.waitForTransform("/orange0", "/base_link", rospy.Time(), rospy.Duration(4))
                (trans, rot) = self.tf_listener_.lookupTransform('/base_link','/orange0', rospy.Time(0))
                obj_coords = [trans[0], trans[1]]
                distance_to_obj = self.dist_two_points(current_pose_coords, obj_coords)
                rospy.loginfo("Distance to object: %f", distance_to_obj)
                rospy.loginfo("Force: %f", self.latest_force)

                if distance_to_obj < 0.01:
                    rospy.loginfo("Contact with orange")
                    baseline_force = self.latest_force + 2

                if distance_to_corner < 0.05:
                    rospy.loginfo("Close to corner")
                    break

            rospy.sleep(.1)
            rospy.loginfo("Close gripper")
            self.command_gripper(close_gripper_msg())
            rospy.sleep(.1)
        else:
            rospy.loginfo("Path cancelled")

        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def vel_push_grasp(self, final_pose, force_threshold):
        run_flag = "n"
        current_force = 0
        move_twist = self.plan_linear_twist(final_pose)
        stop_twist = Twist()
        run_flag = raw_input("Ready? (y or n):")

        if run_flag == "y":
            self.turn_on_twist_controller()
            self.twist_pub.publish(move_twist)
        else:
            pass

        while current_force < force_threshold:
            # Get position of gripper
            current_pose = self.move_group.get_current_pose()
            current_pose_coords = [current_pose.pose.position.x, current_pose.pose.position.y]
            final_pose_coords = [final_pose.position.x, final_pose.position.y]
            distance_to_corner = self.dist_two_points(current_pose_coords, final_pose_coords)

            # Distance to orange
            self.tf_listener_.waitForTransform("/orange0", "/base_link", rospy.Time(), rospy.Duration(4))
            (trans, rot) = self.tf_listener_.lookupTransform('/base_link','/orange0', rospy.Time(0))
            obj_coords = [trans[0], trans[1]]
            distance_to_obj = self.dist_two_points(current_pose_coords, obj_coords)
            rospy.loginfo("Distance to object: %f", distance_to_obj)
            # rospy.loginfo("Force: %f", self.latest_force)

            if distance_to_corner < 0.15:
                # rospy.loginfo("Within range")
                current_force = self.latest_force
                rospy.loginfo("Force: %f", self.latest_force)

            if distance_to_corner < 0.05:
                rospy.loginfo("Close to corner")
                break

        rospy.sleep(.1)
        rospy.loginfo("Close gripper")
        self.command_gripper(close_gripper_msg())
        # Stop movement
        self.twist_pub.publish(stop_twist)
        rospy.sleep(.1)

        # Turn traj controller back on
        self.turn_on_joint_traj_controller()

    def closed_loop_push_grasp(self, final_pose, force_threshold):
        run_flag = "n"
        current_force = 0

        run_flag = raw_input("Ready? (y or n):")
        if run_flag == "y":
            self.turn_on_twist_controller()
        else:
            pass

        stop_twist = Twist()

        while current_force < force_threshold:
            # Move towards corner
            move_twist = self.plan_linear_twist(final_pose)
            self.twist_pub.publish(move_twist)

            obj_in_front = True

            while obj_in_front:
                # Get position of gripper
                current_pose = self.move_group.get_current_pose()
                current_pose_coords = [current_pose.pose.position.x, current_pose.pose.position.y]
                final_pose_coords = [final_pose.position.x, final_pose.position.y]
                distance_to_corner = self.dist_two_points(current_pose_coords, final_pose_coords)

                # Distance to orange
                self.tf_listener_.waitForTransform("/orange0", "/base_link", rospy.Time(), rospy.Duration(4))
                (trans, rot) = self.tf_listener_.lookupTransform('/base_link','/orange0', rospy.Time(0))
                obj_coords = [trans[0], trans[1]]
                distance_to_obj = self.dist_two_points(current_pose_coords, obj_coords)
                # rospy.loginfo("Distance to object: %f", distance_to_obj)

                # Location of orange
                self.tf_listener_.waitForTransform("/orange0", "/camera_link", rospy.Time(), rospy.Duration(4))
                (trans, rot) = self.tf_listener_.lookupTransform('/camera_link', '/orange0', rospy.Time(0))
                x_pos_of_obj = trans[0]
                rospy.loginfo("X pos of object: %f", x_pos_of_obj)

                # Check if too far to the right
                if x_pos_of_obj > 0.025:
                    rospy.loginfo("Object to the right")
                    obj_in_front = False
                    self.twist_pub.publish(stop_twist)
                    # rotate until obj_in_front
                    rot_twist = self.plan_rotation_twist("right")
                    self.twist_pub.publish(rot_twist)
                    while x_pos_of_obj > 0.025:
                        # Location of orange
                        self.tf_listener_.waitForTransform("/orange0", "/camera_link", rospy.Time(), rospy.Duration(4))
                        (trans, rot) = self.tf_listener_.lookupTransform('/camera_link', '/orange0', rospy.Time(0))
                        x_pos_of_obj = trans[0]
                        rospy.loginfo("Rotating")
                    self.twist_pub.publish(stop_twist)
                # Check if too far to the left
                elif x_pos_of_obj < -0.025:
                    rospy.loginfo("Object to the left")
                    obj_in_front = False
                    self.twist_pub.publish(stop_twist)
                    # rotate until obj_in_front
                    rot_twist = self.plan_rotation_twist("left")
                    self.twist_pub.publish(rot_twist)
                    while x_pos_of_obj < -0.025:
                        # Location of orange
                        self.tf_listener_.waitForTransform("/orange0", "/camera_link", rospy.Time(), rospy.Duration(4))
                        (trans, rot) = self.tf_listener_.lookupTransform('/camera_link', '/orange0', rospy.Time(0))
                        x_pos_of_obj = trans[0]
                        rospy.loginfo("Rotating")
                    self.twist_pub.publish(stop_twist)

                if distance_to_corner < 0.15:
                    # rospy.loginfo("Within range")
                    current_force = self.latest_force
                    rospy.loginfo("Force: %f", self.latest_force)
                    break

                if distance_to_corner < 0.05:
                    rospy.loginfo("Close to corner")
                    break

            # SOMETHIGS

        rospy.sleep(.1)
        rospy.loginfo("Close gripper")
        self.command_gripper(close_gripper_msg())
        # Stop movement
        self.twist_pub.publish(stop_twist)
        rospy.sleep(.1)

        # Turn traj controller back on
        self.turn_on_joint_traj_controller()

    def plan_linear_twist(self, final_pose):
        current_pose = self.move_group.get_current_pose()
        x_diff = -(final_pose.position.x - current_pose.pose.position.x)
        y_diff = -(final_pose.position.y - current_pose.pose.position.y)

        rospy.loginfo("X diff: %f", x_diff)
        rospy.loginfo("Y diff: %f", x_diff)
        rospy.sleep(5)
        
        move_twist = Twist()
        # TODO: Normalise values
        move_twist.linear.x = x_diff / 10
        move_twist.linear.y = y_diff / 10

        return move_twist

    def plan_rotation_twist(self, direction):
        move_twist = Twist()
        if direction == "left":
            move_twist.angular.z = 0.05
        elif direction == "right":
            move_twist.angular.z = -0.05
        else:
            rospy.loginfo("Invalid direction")

        return move_twist

    def move_to_joint_position(self, joint_array, plan=None):
        self.move_group.set_joint_value_target(joint_array)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.execute(plan, wait=True)

        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def get_drop_pose(self):
        drop = PoseStamped()

        drop.pose.position.x = -0.450
        drop.pose.position.y = -0.400
        drop.pose.position.z = 0.487

        return drop

    def command_gripper(self, grip_msg):
        # publish gripper message to gripper
        self.gripper_pub.publish(grip_msg)

    def lift_up_pose(self):
        # lift gripper up
        lift_dist = 0.05
        new_pose = self.move_group.get_current_pose()
        new_pose.pose.position.z += lift_dist

        self.move_group.set_start_state_to_current_state()
        self.move_group.set_pose_target(new_pose)
        plan_to_lift = self.move_group.plan()

        return new_pose, plan_to_lift

    def get_robot_state(self, joint_list):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                            'wrist_2_joint', 'wrist_3_joint']
        joint_state.position = joint_list
        robot_state = RobotState()
        robot_state.joint_state = joint_state

        return robot_state

    def launch_pcl_process(self, pcl_node):
        pcl_process = self.launcher.launch(pcl_node)
        while not pcl_process.is_alive():
            rospy.sleep(0.1)
        return pcl_process

    def stop_pcl_process(self, pcl_process):
        pcl_process.stop()
        while pcl_process.is_alive():
            rospy.sleep(0.1)

    def makeMarker(self, pose, color, id):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time().now()
        marker.ns = "my_namespace"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0  # // Don't forget to set the alpha!
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        return marker

    def main(self):
        # Set rate
        rate = rospy.Rate(1)
        # Gripper startup sequence
        while not self.gripper_data and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rospy.sleep(1)
        self.command_gripper(reset_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(activate_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(close_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(open_gripper_msg())
        rospy.sleep(.1)
        rospy.loginfo("Gripper active")

        # Go to move home position using joint
        # self.move_to_joint_position(self.move_home_joints)
        # rospy.sleep(0.1)
        # rospy.loginfo("Moved to Home Position")
        self.move_to_joint_position(self.view_home_joints)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to View Position")

        while not rospy.is_shutdown():
            # Boot up pcl
            # TODO: remove preprocess? Feed it into colour segmentation
            # pcl_node = roslaunch.core.Node('push_grasp', 'pcl_preprocess_node.py')
            # pcl_process = self.launch_pcl_process(pcl_node)

            # TODO: rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0
            # pcl_node = roslaunch.core.Node('push_grasp', 'pcl_preprocess_node.py')
            # pcl_process = self.launch_pcl_process(pcl_node)

            # Wait for a valid reading from agile grasp
            # self.agile_state = AgileState.RESET
            # while self.agile_state is not AgileState.READY:
            #     rospy.loginfo("Waiting for agile grasp")
            #     rospy.sleep(2)

            # rospy.loginfo("Grasp detection complete")
            # Stop pcl
            # self.stop_pcl_process(pcl_process)

            # Find best grasp from reading
            # final_grasp_pose, final_grasp_pose_offset, plan_to_offset, plan_to_final, corner_pose = self.find_best_grasp(self.agile_data)
            final_grasp_pose, final_grasp_pose_offset, plan_to_offset, plan_to_final, corner_pose, valid_grasp = self.find_best_grasp()

            # m1 = self.makeMarker(final_grasp_pose, (1, 0, 0), 0)
            # m2 = self.makeMarker(final_grasp_pose_offset, (0, 1, 0), 1)
            # m_array = MarkerArray()
            # m_array.markers = [m1, m2]
            # self.vis_pub.publish(m_array)

            if valid_grasp:
                self.run_motion(final_grasp_pose_offset, plan_to_offset, plan_to_final, corner_pose)
                pass
            else:
                rospy.loginfo("No pose target generated!")

            rospy.loginfo("Task complete!")
            rospy.spin()
            rate.sleep()


if __name__ == '__main__':
    try:
        grasper = GraspExecutor()
        grasper.main()
    except KeyboardInterrupt:
        pass
