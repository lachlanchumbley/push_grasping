import numpy as np
import moveit_commander
from moveit_msgs.msg import DisplayTrajectory
import rospy
import cv2

import sys

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

    M = cv2.moments(contours[0])
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cv2.circle(canvas, center, 2, (0, 255, 0), -1)

    # cv2.waitKey(0)

    return center