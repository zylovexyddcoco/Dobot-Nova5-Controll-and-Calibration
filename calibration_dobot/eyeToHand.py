# coding=utf-8
"""
眼在手外 (Eye-to-Hand) 计算脚本
适配 Dobot V3 采集的数据 (pose_eyetohand.txt)
核心逻辑：对机械臂位姿求逆，计算相机相对于基座的变换矩阵
"""

import os
import cv2
import numpy as np
import csv

# 设置打印精度
np.set_printoptions(precision=8, suppress=True)

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵 (Z-Y-X 顺序，对应 Dobot/Realsense 常用习惯)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx 
    return R

def pose_to_homogeneous_matrix(pose):
    x, y, z, rx, ry, rz = pose
    R = euler_angles_to_rotation_matrix(rx, ry, rz)
    t = np.array([x, y, z]).reshape(3, 1)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t[:, 0]
    return H

def inverse_transformation_matrix(T):
    """
    【眼在手外关键步骤】
    求逆变换矩阵。因为在眼在手外标定中，我们需要建立 "Base -> Gripper -> CalibrationTarget" 的链条，
    OpenCV的算法通常需要 Gripper 相对于 Base 的逆（即 Base 相对于 Gripper）或者特定的输入形式。
    保留您原脚本的这一逻辑。
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    T_inv = np.identity(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def save_matrices_to_csv(matrices, file_path):
    rows, cols = matrices[0].shape
    num_matrices = len(matrices)
    combined_matrix = np.zeros((rows, cols * num_matrices))
    for i, matrix in enumerate(matrices):
        combined_matrix[:, i * cols: (i + 1) * cols] = matrix
    
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in combined_matrix:
            csv_writer.writerow(row)
    print(f"中间矩阵文件已保存: {file_path}")

def poses_process_and_save(source_filepath, save_dir):
    """读取位姿 -> 转齐次矩阵 -> 求逆 -> 保存CSV"""
    with open(source_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 解析数据 (x, y, z, rx, ry, rz)
    data_values = [float(i) for line in lines for i in line.split(',')]
    
    matrices = []
    for i in range(0, len(data_values), 6):
        raw_pose = data_values[i:i+6]
        # 1. 转齐次矩阵
        H = pose_to_homogeneous_matrix(raw_pose)
        # 2. 【核心】求逆矩阵 (这是眼在手外与眼在手内最大的区别)
        H_inv = inverse_transformation_matrix(H)
        matrices.append(H_inv)
    
    output_csv = os.path.join(save_dir, 'robotToolPose.csv')
    save_matrices_to_csv(matrices, output_csv)
    return output_csv

def save_calibration_results(R_tsai, t_tsai, R_park, t_park, R_horaud, t_horaud, save_dir):
    methods = [
        ('TSAI', R_tsai, t_tsai, 'calibration_result_tsai.txt'),
        ('PARK', R_park, t_park, 'calibration_result_park.txt'),
        ('HORAUD', R_horaud, t_horaud, 'calibration_result_horaud.txt')
    ]
    
    for method_name, R, t, filename in methods:
        full_path = os.path.join(save_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f"# {method_name} Eye-to-Hand Result (Base to Camera)\n")
            f.write(f"# Rotation Matrix (3x3):\n")
            R_flat = R.flatten()
            for i in range(0, 9, 3):
                f.write(f"{R_flat[i]:.8f} {R_flat[i+1]:.8f} {R_flat[i+2]:.8f}\n")
            f.write(f"# Translation Vector (3x1, Unit: m):\n")
            t_flat = t.flatten()
            f.write(f"{t_flat[0]:.8f} {t_flat[1]:.8f} {t_flat[2]:.8f}\n")
        print(f"[{method_name}] 结果已保存: {full_path}")

def save_camera_params(mtx, dist, save_dir):
    np.savetxt(os.path.join(save_dir, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(save_dir, 'distortion_coefficients.txt'), dist)
    print("相机参数已保存")

def compute_T(images_path, corner_point_long, corner_point_short, corner_point_size, save_dir, pose_csv_path):
    print(f"参数配置: 角点 {corner_point_long}x{corner_point_short}, 尺寸 {corner_point_size}m")
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    # 准备物体点
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2)
    objp = corner_point_size * objp

    obj_points = []
    img_points = []
    
    valid_count = 0
    # 自动遍历图片 (0.jpg 到 99.jpg)
    for i in range(100):
        # 兼容之前的采集脚本 (.jpg) 和旧脚本 (.png)
        p_jpg = os.path.join(images_path, f"{i}.jpg")
        p_png = os.path.join(images_path, f"{i}.png")
        
        target = None
        if os.path.exists(p_jpg): target = p_jpg
        elif os.path.exists(p_png): target = p_png
        
        if target:
            img = cv2.imread(target)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
            
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                img_points.append(corners2)
                
                # 简单显示一下进度
                cv2.drawChessboardCorners(img, (corner_point_long, corner_point_short), corners2, ret)
                cv2.imshow('Processing...', img)
                cv2.waitKey(50) # 不阻塞，快速过
                valid_count += 1
            else:
                print(f"警告: {target} 未识别到角点")
    
    cv2.destroyAllWindows()
    print(f"有效图片数量: {valid_count}")
    
    if valid_count < 3:
        print("错误: 图片数量过少，无法标定")
        return None, None

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)
    save_camera_params(mtx, dist, save_dir)

    # 读取经过求逆处理的机器人位姿
    tool_pose_inv = np.loadtxt(pose_csv_path, delimiter=',')
    
    # 数据对齐检查
    pose_count = tool_pose_inv.shape[1] // 4
    if pose_count < valid_count:
        print(f"严重错误: 位姿数量 ({pose_count}) 少于图片数量 ({valid_count})")
        valid_count = pose_count
        rvecs = rvecs[:valid_count]
        tvecs = tvecs[:valid_count]

    R_tool_inv = []
    t_tool_inv = []
    for i in range(valid_count):
        # 这里的 tool_pose_inv 已经是 T_tool_base (即 T_base_tool 的逆)
        R_tool_inv.append(tool_pose_inv[0:3, 4*i : 4*i+3])
        t_tool_inv.append(tool_pose_inv[0:3, 4*i+3])

    # 转换旋转向量
    rotation_matrices = []
    for rvec in rvecs:
        rot_mat, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(rot_mat)

    print("-" * 30)
    print("开始计算手眼矩阵 (Eye-to-Hand)...")

    # 1. TSAI
    R_tsai, t_tsai = cv2.calibrateHandEye(R_tool_inv, t_tool_inv, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_TSAI)
    print("TSAI Result (Base -> Camera):\n", t_tsai.T)

    # 2. PARK
    R_park, t_park = cv2.calibrateHandEye(R_tool_inv, t_tool_inv, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
    print("PARK Result:\n", t_park.T)

    # 3. HORAUD
    R_horaud, t_horaud = cv2.calibrateHandEye(R_tool_inv, t_tool_inv, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
    print("HORAUD Result:\n", t_horaud.T)

    # 保存结果
    save_calibration_results(R_tsai, t_tsai, R_park, t_park, R_horaud, t_horaud, save_dir)
    return R_tsai, t_tsai

if __name__ == '__main__':
    # ========== 路径配置 ==========
    base_dir = os.path.dirname(__file__)
    
    # 输入: Dobot采集脚本生成的文件夹和txt
    images_path = os.path.join(base_dir, "img_eyetohand_tool0_bizhang")
    pose_file_txt = os.path.join(base_dir, "pose_eyetohand_tool0_bizhang.txt")

    # 输出: 结果文件夹
    save_dir = os.path.join(base_dir, "result_eye_to_hand_tool0_bizhang")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 标定板参数 (保持与 eyeInHand 一致)
    corner_point_long = 11
    corner_point_short = 8
    corner_point_size = 0.015 # 0.015m = 15mm

    print(f"图片路径: {images_path}")
    print(f"位姿文件: {pose_file_txt}")

    # 1. 处理位姿 (求逆)
    pose_csv_path = poses_process_and_save(pose_file_txt, save_dir)
    
    # 2. 计算手眼关系
    compute_T(
        images_path,
        corner_point_long,
        corner_point_short,
        corner_point_size,
        save_dir,
        pose_csv_path
    )
    