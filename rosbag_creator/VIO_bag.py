
import numpy as np
import time, sys, os, glob, re
from ros import rosbag
import rospy
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Imu, Image, NavSatFix
import pandas as pd
from PIL import ImageFile
import argparse
import cv2 as cv
import pickle
from cv_bridge import CvBridge
import sys
bridge = CvBridge()


def sort_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    sorted_files = sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))
    sorted_array = []
    for file in sorted_files:
        file_path = os.path.join(folder_path, file)
        sorted_array.append(file_path)
            
    return sorted_array

def split_float_to_secs_nsecs(float_number):
    # Split the number into integer and fractional parts
    str_number = str(float_number)
    integer_part, decimal_part = str_number.split('.')
    decimal_as_nsecs = round(float("0." + decimal_part) * 1000000000)
    # Use the built-in `format` function to ensure we get 9 digits
    decimal_as_nsecs_str = format(int(decimal_as_nsecs), '09')
    # Return the parts as integers
    return int(integer_part), int(decimal_as_nsecs_str)
    
def main():

    argparser = argparse.ArgumentParser(description= 'for genarating raw data')
    argparser.add_argument('--root_path',type=str,default='/media/lde/yuhang/DynaCARLA',help='dataset root path')
    argparser.add_argument('--output_path',type=str,default='/media/lde/yuhang/DynaCARLA/rosbags',help='output path to save ros bag')
    args = argparser.parse_args()
    
    
    map_names = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
    weathers = ['ClearNoon', 'FoggyNoon', 'RainyNight']
    dyna_types = ['dynamic', 'static']

    
    for i in range(8):
        map_name = map_names[i]
        for weather in weathers:
            for dyna_type in dyna_types:
                seq_root = os.path.join(args.root_path, map_name)
                common_data_path = os.path.join(seq_root, 'common_data.pkl')
                cam_root_path = os.path.join(seq_root, weather+'/'+dyna_type)
                bag_name = 'VIO_'+map_name+'_'+weather+'_'+dyna_type+'_.bag'
                bag_path = os.path.join(args.output_path,map_name,'visual',bag_name)
                
                with open(common_data_path, 'rb') as f:
                    common_data = pickle.load(f)
                
                with rosbag.Bag(bag_path, 'w') as bag:
                    print('now processing: '+bag_name)
                    #----------------------imu data--------------------
                    # carla: angular veocity in left hand coordinate!!!
                    print('converting imu data......')
                    for i in range(len(common_data['imu']['acc'])):
                        secs, nano_secs = split_float_to_secs_nsecs(common_data['imu']['timestamp'][i])
                        timestamp = rospy.Time(secs= secs, nsecs=nano_secs)
                        imu_msg = Imu()
                        imu_msg.header.seq = i
                        if i ==1:
                            i = 2
                        imu_msg.header.frame_id = "imu"
                        imu_msg.header.stamp = timestamp
                        imu_msg.linear_acceleration.x = float(common_data['imu']['acc'][i][0])
                        imu_msg.linear_acceleration.y = float(common_data['imu']['acc'][i][1])
                        imu_msg.linear_acceleration.z = float(common_data['imu']['acc'][i][2])
                        imu_msg.angular_velocity.x = float(common_data['imu']['gyro'][i][0])
                        imu_msg.angular_velocity.y = float(common_data['imu']['gyro'][i][1])
                        imu_msg.angular_velocity.z = float(common_data['imu']['gyro'][i][2])
                        bag.write("/imu", imu_msg, timestamp)
                        sys.stdout.write('\r'+str(i)+' of '+str(len(common_data['imu']['acc'])))
                        sys.stdout.flush()
                    
                    #---------------------camera data-------------------
                    print('converting left cam data......')
                    print(os.path.join(cam_root_path, 'stereo_l'))
                    img_paths = sort_files_in_folder(os.path.join(cam_root_path, 'stereo_l'))
                    for i in range(len(img_paths)):
                        img = cv.imread(img_paths[i])
                        # encoding = "bgr8"
                        cv_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        encoding = "mono8"
                        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
                        secs, nano_secs = split_float_to_secs_nsecs(common_data['cam_left']['timestamp'][i])
                        timestamp = rospy.Time(secs= secs, nsecs=nano_secs)
                        image_message.header.stamp = timestamp
                        image_message.header.frame_id = 'cam_left'
                        image_message.header.seq = i
                        bag.write('/cam0_raw', image_message, timestamp)
                        sys.stdout.write('\r'+str(i)+' of '+str(len(img_paths)))
                        sys.stdout.flush()
                
                    print('converting right cam data......')
                    img_paths = sort_files_in_folder(os.path.join(cam_root_path, 'stereo_r'))
                    for i in range(len(img_paths)):
                        img = cv.imread(img_paths[i])
                        img = cv.imread(img_paths[i])
                        # encoding = "bgr8"
                        cv_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        encoding = "mono8"
                        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
                        secs, nano_secs = split_float_to_secs_nsecs(common_data['cam_right']['timestamp'][i])
                        timestamp = rospy.Time(secs= secs, nsecs=nano_secs)
                        image_message.header.stamp = timestamp
                        image_message.header.frame_id = 'cam_right'
                        image_message.header.seq = i
                        bag.write('/cam1_raw', image_message, timestamp)
                        sys.stdout.write('\r'+str(i)+' of '+str(len(img_paths)))
                        sys.stdout.flush()
                        
                bag.close()    

if __name__ == '__main__':
    main()
