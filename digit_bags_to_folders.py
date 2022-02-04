#!/usr/bin/env python
import rospy
import rosbag
import glob
from grasp_executor.srv import AngleTrack
from std_srvs.srv import Empty
import os
import pandas as pd
import itertools
from cv_bridge import CvBridge, CvBridgeError
import copy
import cv2
from rosbag.bag import ROSBagUnindexedException

FILE_DIR = './'
FOLDER_NAME = 'push_grasp_data_bags'
BAG_DIR = 'push_grasp_data_bags/'
OUTPUT_DIR = 'both_data_unpacked/'

# def digit_data_to_folder(folder, bridge, time_data, image_data, digit_data_0, digit_data_1, track_angle_srv):
#     df = pd.DataFrame()
#     df['true_angle'] = None
#     df['timestep'] = None

#     for i, (time, im, digit_0, digit_1) in enumerate(itertools.izip(time_data, image_data, digit_data_0, digit_data_1)):
#         new_row = pd.Series(dtype='int64')
#         response = track_angle_srv(im)
#         new_row['true_angle'] = response.angle
#         new_row['timestep'] = time.to_nsec()
#         df = df.append(copy.deepcopy(new_row), ignore_index=True)

#         dig_im_0 = bridge.imgmsg_to_cv2(digit_0, desired_encoding='8UC3')
#         dig_im_1 = bridge.imgmsg_to_cv2(digit_1, desired_encoding='8UC3')

#         cv2.imwrite(folder+"/"+str(i)+"_digit_0.jpeg", dig_im_0)
#         cv2.imwrite(folder+"/"+str(i)+"_digit_1.jpeg", dig_im_1)

#     df.to_csv(folder+"/"+"ground_truth.csv", index=False)


def main():
    bridge = CvBridge()
    num_data_points = len(glob.glob(FILE_DIR+OUTPUT_DIR+'*/'))
    bag_list = glob.glob(FILE_DIR+BAG_DIR+'*.bag')
    print(FILE_DIR+BAG_DIR+'*.bag', bag_list)

    for i, bag_dir in enumerate(bag_list):
        try:
            bag = rosbag.Bag(bag_dir)
        except ROSBagUnindexedException:
            print('Unindexed bag'+bag_dir+' with i='+str(i)+' (total list length = '+str(num_data_points)+')')
        # continue

        time_data = [msg.stamp for _, msg, _ in bag.read_messages(topics=['time'])]
        rgb_image = [msg for _, msg, _ in bag.read_messages(topics=['rgb_image'])]
        depth_image = [msg for _, msg, _ in bag.read_messages(topics=['depth_image'])]
        action = [msg for _, msg, _ in bag.read_messages(topics=['action'])]
        reward = [msg for _, msg, _ in bag.read_messages(topics=['reward'])]
        x_force = [msg for _, msg, _ in bag.read_messages(topics=['x_force'])]
        y_force = [msg for _, msg, _ in bag.read_messages(topics=['y_force'])]
        z_force = [msg for _, msg, _ in bag.read_messages(topics=['z_force'])]
        x_pos = [msg for _, msg, _ in bag.read_messages(topics=['x_pos'])]

        print(len(time_data),len(rgb_image),len(depth_image),len(action), len(x_force), len(x_pos))
        
        if len(time_data) == len(rgb_image) == len(depth_image) == len(action) == len(reward):
            # #Folder naming convention is: <name>_<number of df>_<twist>_<gripperTwist>_<eeGroundRot>_<eeAirRot>_<gripperWidth>
            # folder = [FILE_DIR,OUTPUT_DIR,FOLDER_NAME,'_',num_data_points,'_',meta.gripperTwist,'_',meta.eeGroundRot,'_',meta.eeAirRot,'_',meta.gripperWidth]
            # folder = ''.join(list(map(str, folder)))
            # os.mkdir(folder)

            # digit_data_to_folder(folder, bridge, time_data, image_data, digit_data_0, digit_data_1, track_angle_srv)
            for i, (time, rgb, depth, a, r, x_f, y_f, z_f, x) in enumerate(itertools.izip(time_data, rgb_image, depth_image, action, reward, x_force, y_force, z_force, x_pos)):
            # num_data_points += 1
                cv2_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
                cv2_depth = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                cv2.imshow("rgb", cv2_rgb)
                cv2.imshow("depth", cv2_depth)
                print("action", a, "reward", r, "x_force", x_f, "y_force", y_f, "z_force", z_f, "x_pos", x)
                cv2.waitKey(10)
        else:
            print('ERROR: bag ' + bag_dir +' had mismatched data!')

        # reset_angle_srv()

if __name__ == "__main__":
    if not os.path.exists(FILE_DIR + OUTPUT_DIR):
        # Create a new directory because it does not exist 
        os.makedirs(FILE_DIR + OUTPUT_DIR)
        print("The new directory is created!")

    # rospy.wait_for_service('track_angle')
    # try:
    #     track_angle_srv = rospy.ServiceProxy('track_angle', AngleTrack)
    #     rospy.loginfo("Angle tracking service available!")
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s"%e)

    # rospy.wait_for_service('reset_angle_tracking')
    # try:
    #     reset_angle_srv = rospy.ServiceProxy('reset_angle_tracking', Empty)
    #     rospy.loginfo("Reset service available!")
    # except rospy.ServiceException as e:
    #     print("Service call failed: %s"%e)
    

    main()