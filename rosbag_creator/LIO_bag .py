
import numpy as np
import sys, os,  re
from ros import rosbag
import rospy
import roslib
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import PointCloud2, Imu, PointField
from numpy.lib.recfunctions import unstructured_to_structured
import std_msgs.msg as std_msgs
import argparse
from tqdm import tqdm
import array
import pickle

def split_float_to_secs_nsecs(float_number):
    # Split the number into integer and fractional parts
    str_number = str(float_number)
    integer_part, decimal_part = str_number.split('.')
    decimal_as_nsecs = round(float(f"0.{decimal_part}") * 1_000_000_000)
    # Use the built-in `format` function to ensure we get 9 digits
    decimal_as_nsecs_str = format(decimal_as_nsecs, '09')
    # Return the parts as integers
    return int(integer_part), int(decimal_as_nsecs_str)

def sort_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    sorted_files = sorted(file_list, key=lambda x: int(re.findall(r'\d+', x)[0]))
    sorted_array = []
    for file in sorted_files:
        file_path = os.path.join(folder_path, file)
        sorted_array.append(file_path)
    return sorted_array

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def process_pointcloud(points_in, label, label_to_delete):
    list_points = []
    for i in range(len(label)):
        if label[i] not in label_to_delete:
            list_points.append(points_in[i].tolist())
    return list_points

def creat_pc2_msg( points, header, point_step = None):
    fields = [PointField(name='x', offset=0,
                    datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4,
                        datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8,
                        datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12,
                        datatype=PointField.FLOAT32, count=1)]
    # Check if input is numpy array
    if isinstance(points, np.ndarray):
        # Check if this is an unstructured array
        if points.dtype.names is None:
            assert all(fields[0].datatype == field.datatype for field in fields[1:]), \
                'All fields need to have the same datatype. Pass a structured NumPy array \
                    with multiple dtypes otherwise.'
            # Convert unstructured to structured array
            points = unstructured_to_structured(
                points,
                dtype=dtype_from_fields(fields, point_step))
        else:
            assert points.dtype == dtype_from_fields(fields, point_step), \
                'PointFields and structured NumPy array dtype do not match for all fields! \
                    Check their field order, names and types.'
    else:
        # Cast python objects to structured NumPy array (slow)
        points = np.array(
            # Points need to be tuples in the structured array
            list(map(tuple, points)),
            dtype=dtype_from_fields(fields, point_step))
    # Handle organized clouds
    assert len(points.shape) <= 2, \
        'Too many dimensions for organized cloud! \
            Points can only be organized in max. two dimensional space'
    height = 1
    width = points.shape[0]
    # Check if input points are an organized cloud (2D array of points)
    if len(points.shape) == 2:
        height = points.shape[1]
    # Convert numpy points to array.array
    memory_view = memoryview(points)
    casted = memory_view.cast('B')
    array_array = array.array('B')
    array_array.frombytes(casted)
    list_array = list(array_array)

    cloud = PointCloud2(
        header=header,
        height=height,
        width=width,
        is_dense=False,
        is_bigendian=sys.byteorder != 'little',
        fields=fields,
        point_step=points.dtype.itemsize,
        row_step=(points.dtype.itemsize * width))
    cloud.data = list_array
    return cloud

def dtype_from_fields(fields, point_step = None):
    field_names = []
    field_offsets = []
    field_datatypes = []
    _DATATYPES = {}
    _DATATYPES[PointField.INT8] = np.dtype(np.int8)
    _DATATYPES[PointField.UINT8] = np.dtype(np.uint8)
    _DATATYPES[PointField.INT16] = np.dtype(np.int16)
    _DATATYPES[PointField.UINT16] = np.dtype(np.uint16)
    _DATATYPES[PointField.INT32] = np.dtype(np.int32)
    _DATATYPES[PointField.UINT32] = np.dtype(np.uint32)
    _DATATYPES[PointField.FLOAT32] = np.dtype(np.float32)
    _DATATYPES[PointField.FLOAT64] = np.dtype(np.float64)

    DUMMY_FIELD_PREFIX = 'unnamed_field'
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == '':
            name = f'{DUMMY_FIELD_PREFIX}_{i}'
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f'{name}_{a}'
            else:
                subfield_name = name
            assert subfield_name not in field_names, 'Duplicate field names are not allowed!'
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)
    # Create dtype
    dtype_dict = {
            'names': field_names,
            'formats': field_datatypes,
            'offsets': field_offsets
    }
    if point_step is not None:
        dtype_dict['itemsize'] = point_step
    return np.dtype(dtype_dict)
    
    
def main():

    argparser = argparse.ArgumentParser(description= 'for genarating raw data')
    argparser.add_argument('--root_path',type=str,default='/media/lde/yuhang/DynaCARLA',help='Data index')
    argparser.add_argument('--seg_lidar',action='store_true',default=False,help='If crop lidar point cloud belongs to certain label')
    argparser.add_argument('--seg_lidar_label',metavar='label1_label2_label3_...',type=str,default='4_10',help='lidar labels to crop')    
    argparser.add_argument('--output_path',type=str,default='/media/lde/yuhang/DynaCARLA/rosbags',help='output path to save ros bag')
    args = argparser.parse_args()
    
    args.label_to_delete = [int(x) for x in args.seg_lidar_label.split('_')]
    map_names = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']
    # [dynamic_type, if_use_segmentation]
    presets = [['dynamic', False], ['dynamic', True], ['static', False]]

    for map_name in map_names:
        seq_root = os.path.join(args.root_path, map_name)
        common_data_path = os.path.join(seq_root, 'common_data.pkl')
        for preset in presets:
            if preset[1]:
                bag_name = 'LIOseg_'+map_name+'_'+preset[0]+'_.bag'
            else:
                bag_name = 'LIO_'+map_name+'_'+preset[0]+'_.bag'
            bag_path = os.path.join(args.output_path,map_name,'lidar',bag_name)
            
            with open(common_data_path, 'rb') as f:
                common_data = pickle.load(f)
            
            with rosbag.Bag(bag_path, 'w') as bag:
                
                #----------------------imu data--------------------
                print('converting imu data......')
                for i in tqdm(range(len(common_data['imu']['acc']))):
                    secs, nano_secs = split_float_to_secs_nsecs(common_data['imu']['timestamp'][i])
                    timestamp = rospy.Time(secs=secs, nsecs=nano_secs)
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
                
                print('converting lidar msg...')
                lidar_path = os.path.join(seq_root, 'lidar', preset[0])
                seg_lidar_path = os.path.join(seq_root, 'seg_lidar', preset[0])
                lidar_file_paths = sort_files_in_folder(lidar_path)
                seg_lidar_file_paths = sort_files_in_folder(seg_lidar_path)
                for index in tqdm(range(len(lidar_file_paths))):
                    lidar_pkl = read_pickle(lidar_file_paths[index])
                    label_pkl = read_pickle(seg_lidar_file_paths[index])
                    if preset[1]:
                        lidar_list = process_pointcloud(lidar_pkl, label_pkl, args.label_to_delete)
                    else:
                        lidar_list = lidar_pkl.tolist()
                    secs, nano_secs = split_float_to_secs_nsecs(common_data['lidar']['timestamp'][index])
                    timestamp = rospy.Time(secs=secs, nsecs=nano_secs)
                    header = std_msgs.Header()
                    header.stamp = timestamp
                    header.frame_id = 'laser'
                    header.seq = index
                    lidar_msg = creat_pc2_msg(lidar_list, header)
                    bag.write('/velodyne', lidar_msg, timestamp)
                    
            bag.close()    

if __name__ == '__main__':
    main()
