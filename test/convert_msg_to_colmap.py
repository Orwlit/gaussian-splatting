import msgpack
import random
import os

def read_msgpack_file(path):
    with open(path, 'rb') as file:
        msgpack_data = msgpack.unpackb(file.read())
    return msgpack_data

def generate_random_rgb():
    return [random.randint(0, 255) for _ in range(3)]

def convert_to_colmap_format(json_data):
    # 处理关键帧和地图点
    # 这里需要根据 msg 数据的具体格式来解析
    keyframes = json_data['keyframes']
    landmarks = json_data['landmarks']

    for item in keyframes:
        cam_rot_cw = keyframes[item]['rot_cw']
        cam_trans_cw = keyframes[item]['trans_cw']
        cam_time_stamp = keyframes[item]['ts']
        pass

    
    for item in landmarks:
        pt_xyz = landmarks[item]['pos_w']
        pass
    



def generate_random_rgb():
    """生成随机RGB颜色值。"""
    return [random.randint(0, 255) for _ in range(3)]

def convert_to_colmap(output_dir, keyframes, landmarks):
    # 创建关键帧文件
    with open(f"{output_dir}/colmap_keyframes.txt", 'w') as kf_file:
        for kf_id, kf_data in keyframes.items():
            cam_rot_cw = kf_data['rot_cw']
            cam_trans_cw = kf_data['trans_cw']
            cam_time_stamp = kf_data['ts']
            # 格式化和写入关键帧数据
            # kf_id, time_stamp, q(4), t(3)
            kf_file.write(f"{kf_id} {cam_time_stamp} {cam_rot_cw[0]} {cam_rot_cw[1]} {cam_rot_cw[2]} {cam_rot_cw[3]} {cam_trans_cw[0]} {cam_trans_cw[1]} {cam_trans_cw[2]}\n")

    # 创建三维点文件
    with open(f"{output_dir}/colmap_points3D.txt", 'w') as pt_file:
        for pt_id, pt_data in landmarks.items():
            pt_xyz = pt_data['pos_w']
            rgb = generate_random_rgb()
            # 格式化和写入三维点数据
            # pt_id, x, y, z, r, g, b
            pt_file.write(f"{pt_id} {pt_xyz[0]} {pt_xyz[1]} {pt_xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")



# 使用示例
msgpack_path = 'map.msg'
output_dir = "output"
json_data = read_msgpack_file(msgpack_path)
convert_to_colmap(output_dir=output_dir, keyframes=json_data['keyframes'], landmarks=json_data['landmarks'])
