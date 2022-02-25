import numpy as np
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
import rospy
import cv2
import math
import sys
import tf
from std_msgs.msg import Header, Float64
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from geometry_msgs.msg import PoseStamped

PYTHON3 = sys.version_info.major == 3


def dist_to_guess(p_base, guess):
    return np.sqrt((p_base.x - guess[0])**2 + (p_base.y - guess[1])**2 + (p_base.z - guess[2])**2)


def vector3ToNumpy(v):
    return np.array([v.x, v.y, v.z])


def move_ur5(move_group, robot, disp_traj_pub, input, plan=None, no_confirm=False):
    if type(input) == list:
        move_group.set_joint_value_target(input)
    else:
        move_group.set_pose_target(input)

    if not plan:
        plan = move_group.plan()

    if no_confirm or check_valid_plan(disp_traj_pub, robot, plan):
        move_group.execute(plan, wait=True)
    else: 
        print("Plan is invalid!")

    move_group.stop()
    move_group.clear_pose_targets()


def show_motion(disp_traj_pub, robot, plan):
    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    disp_traj_pub.publish(display_trajectory)


def check_valid_plan(disp_traj_pub, robot, plan):
    run_flag = "d"

    while run_flag == "d":
        show_motion(disp_traj_pub, robot, plan)
        run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

    return True if run_flag == "y" else False


def find_center(im):
    result = im.copy()
    image = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    # image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    light_orange = np.array([0, 87, 146])
    dark_orange = np.array([16, 255, 255])

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, light_orange, dark_orange)
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow("orig", im)
    cv2.imshow("hsv", image)
    cv2.imshow("mask", mask)
    # cv2.waitKey(1)

    contours = None
    if PYTHON3:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # print(tmp, contours)
        contours = sorted(
            contours, key=lambda el: cv2.contourArea(el), reverse=True
        )

    else:
        _, contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours.sort(key=lambda el: cv2.contourArea(el), reverse=True)

    canvas = result.copy()
    
    if len(contours) > 0:
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(canvas, center, 2, (0, 255, 0), -1)
        else:
            # Return random
            x_pos = 320
            # x_pos = np.random.randint(0,640)
            center = [x_pos,240]
    else:
        # Return random
        x_pos = 320
        # x_pos = np.random.randint(0,640)
        center = [x_pos,240]

    # cv2.waitKey(0)

    return center


def dist_two_points(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def smallestSignedAngleBetween(x, y):
    PI = math.pi
    TAU = 2*PI
    a = (x - y) % TAU
    b = (y - x) % TAU
    return -a if a < b else b


def calculate_approach(start_pos, final_pos, distance):
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


def generate_push_pose(current_pose, y_angle, x_angle):
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


def find_nearest_corner(p_base, corner_pos_list):
    # Find the nearest corner to the grasp
    grasp_pos = [p_base.pose.position.x, p_base.pose.position.y]
    distance_list = [0, 0, 0, 0]

    # Calculate distance between the grasp point and the corners
    for i in range(len(corner_pos_list)):
        corner_pos = corner_pos_list[i]
        distance_list[i] = dist_two_points(grasp_pos, corner_pos)
        rospy.loginfo("Distance to corner number %d: %f", i, distance_list[i])

    # Nearest corner
    nearest_corner = distance_list.index(min(distance_list))
    return nearest_corner


def floatToMsg(data):
    force_msg = Float64()
    force_msg.data = data
    return force_msg


def command_gripper(gripper_pub, grip_msg):
    # publish gripper message to gripper
    gripper_pub.publish(grip_msg)


def get_robot_state(joint_list):
    joint_state = JointState()
    joint_state.header = Header()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                        'wrist_2_joint', 'wrist_3_joint']
    joint_state.position = joint_list
    robot_state = RobotState()
    robot_state.joint_state = joint_state

    return robot_state


def lift_up_plan(move_group):
    # lift gripper up
    lift_dist = 0.20
    new_pose = move_group.get_current_pose()
    new_pose.pose.position.z += lift_dist

    move_group.set_start_state_to_current_state()
    move_group.set_pose_target(new_pose)
    plan_to_lift = move_group.plan()

    return new_pose, plan_to_lift

def move_back_plan(move_group):
    # lift gripper up
    lift_dist = 0.05
    new_pose = move_group.get_current_pose()
    new_pose.pose.position.x += lift_dist

    move_group.set_start_state_to_current_state()
    move_group.set_pose_target(new_pose)
    plan_to_lift = move_group.plan()

    return new_pose, plan_to_lift


def add_front_wall(scene, front_wall_name):
    # Add front wall
    front_wall_pose = PoseStamped()
    front_wall_pose.header.frame_id = "base_link"
    front_wall_pose.pose.orientation.w = 1.0
    front_wall_pose.pose.position.x = -0.405
    front_wall_pose.pose.position.y = 0.0475
    front_wall_pose.pose.position.z = 0.05

    scene.add_box(front_wall_name, front_wall_pose, size=(0.02, 0.355, 0.1))


def add_right_wall(scene, right_wall_name):
    # Add right wall
    right_wall_pose = PoseStamped()
    right_wall_pose.header.frame_id = "base_link"
    right_wall_pose.pose.orientation.w = 1.0
    right_wall_pose.pose.position.x = -0.6175
    right_wall_pose.pose.position.y = 0.230
    right_wall_pose.pose.position.z = 0.05

    scene.add_box(right_wall_name, right_wall_pose, size=(0.42, 0.02, 0.1))


def add_back_wall(scene, back_wall_name):
    # Add back wall
    back_wall_pose = PoseStamped()
    back_wall_pose.header.frame_id = "base_link"
    back_wall_pose.pose.orientation.w = 1.0
    back_wall_pose.pose.position.x = -0.830
    back_wall_pose.pose.position.y = 0.0475
    back_wall_pose.pose.position.z = 0.05

    scene.add_box(back_wall_name, back_wall_pose, size=(0.02, 0.355, 0.1))


def add_left_wall(scene, left_wall_name):
    # Add left wall
    left_wall_pose = PoseStamped()
    left_wall_pose.header.frame_id = "base_link"
    left_wall_pose.pose.orientation.w = 1.0
    left_wall_pose.pose.position.x = -0.6175
    left_wall_pose.pose.position.y = -0.130
    left_wall_pose.pose.position.z = 0.05

    scene.add_box(left_wall_name, left_wall_pose, size=(0.42, 0.02, 0.1))


def add_roof(scene, roof_name):
    # Add roof
    roof_pose = PoseStamped()
    roof_pose.header.frame_id = "base_link"
    roof_pose.pose.orientation.w = 1.0
    roof_pose.pose.position.x = -0.75
    roof_pose.pose.position.y = -0.0
    roof_pose.pose.position.z = 0.8

    scene.add_box(roof_name, roof_pose, size=(2.5, 1.4, 0.01))
