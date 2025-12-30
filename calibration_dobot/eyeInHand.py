# coding=utf-8
"""
眼在手内计算脚本 (适配 Dobot V3 数据)
功能：读取 img/ 下的图片和 pose.txt，计算手眼矩阵，并将结果保存到 result_eye_in_hand 文件夹
"""

import os
import cv2
import numpy as np
import csv

# 设置打印精度
np.set_printoptions(precision=8, suppress=True)

def euler_angles_to_rotation_matrix(rx, ry, rz):
    # 计算旋转矩阵
    # 注意：这里使用的是 R = Rz @ Ry @ Rx (对应 Intrinsic Z-Y-X 或 Extrinsic X-Y-Z)
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
    print(f"中间文件已保存: {file_path}")

# 打开文本文件并处理位姿
def poses_save_csv(source_filepath, save_dir):
    with open(source_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 扁平化数据处理
    data_values = [float(i) for line in lines for i in line.split(',')]
    
    matrices = []
    # 每6个数据为一组 (x,y,z,rx,ry,rz)
    for i in range(0, len(data_values), 6):
        matrices.append(pose_to_homogeneous_matrix(data_values[i:i+6]))
    
    # 将结果保存到指定文件夹
    output_csv = os.path.join(save_dir, 'robotToolPose.csv')
    save_matrices_to_csv(matrices, output_csv)
    return output_csv

def save_camera_params(mtx, dist, save_dir):
    """保存相机内参矩阵和畸变系数到指定文件夹"""
    # 保存内参矩阵
    mtx_file = os.path.join(save_dir, 'camera_matrix.txt')
    with open(mtx_file, 'w', encoding='utf-8') as f:
        for i in range(3):
            f.write(f"{mtx[i, 0]:.8f} {mtx[i, 1]:.8f} {mtx[i, 2]:.8f}\n")
    print(f"相机内参已保存: {mtx_file}")
    
    # 保存畸变系数
    dist_file = os.path.join(save_dir, 'distortion_coefficients.txt')
    with open(dist_file, 'w', encoding='utf-8') as f:
        dist_flat = dist.flatten()
        for coeff in dist_flat:
            f.write(f"{coeff:.8f}\n")
    print(f"畸变系数已保存: {dist_file}")

def save_calibration_results(R_tsai, t_tsai, R_park, t_park, R_horaud, t_horaud, save_dir):
    """保存三种方法的标定结果到指定文件夹"""
    methods = [
        ('TSAI', R_tsai, t_tsai, 'calibration_result_tsai.txt'),
        ('PARK', R_park, t_park, 'calibration_result_park.txt'),
        ('HORAUD', R_horaud, t_horaud, 'calibration_result_horaud.txt')
    ]
    
    for method_name, R, t, filename in methods:
        full_path = os.path.join(save_dir, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(f"# {method_name} Result (Camera to TCP)\n")
            f.write(f"# Rotation Matrix (3x3):\n")
            R_flat = R.flatten()
            for i in range(0, 9, 3):
                f.write(f"{R_flat[i]:.8f} {R_flat[i+1]:.8f} {R_flat[i+2]:.8f}\n")
            
            f.write(f"# Translation Vector (3x1, Unit: m):\n")
            t_flat = t.flatten()
            f.write(f"{t_flat[0]:.8f} {t_flat[1]:.8f} {t_flat[2]:.8f}\n")
        
        print(f"[{method_name}] 结果已保存: {full_path}")

def compute_T(images_path, corner_point_long, corner_point_short, corner_point_size, save_dir, pose_csv_path):
    print(f"标定板角点: {corner_point_long}x{corner_point_short}, 格子尺寸: {corner_point_size}m")
    
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    # 准备物体点 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((corner_point_long * corner_point_short, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corner_point_long, 0:corner_point_short].T.reshape(-1, 2)
    objp = corner_point_size * objp # 转为真实物理尺寸(米)

    obj_points = []     # 3D points
    img_points = []     # 2D points

    # 遍历读取图片
    # 注意：这里假设图片命名是 0.jpg, 1.jpg ...
    # 为了更健壮，可以遍历文件夹
    valid_images_count = 0
    
    # 尝试读取 0 到 99 张图片
    for i in range(100): 
        # 兼容两种常见格式
        image_path_jpg = os.path.join(images_path, f"{i}.jpg")
        image_path_png = os.path.join(images_path, f"{i}.png")
        
        target_path = None
        if os.path.exists(image_path_jpg):
            target_path = image_path_jpg
        elif os.path.exists(image_path_png):
            target_path = image_path_png
            
        if target_path:
            img = cv2.imread(target_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            
            # 寻找角点
            ret, corners = cv2.findChessboardCorners(gray, (corner_point_long, corner_point_short), None)
            
            if ret:
                obj_points.append(objp)
                # 亚像素优化
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                img_points.append(corners2)
                
                # 仅在调试时显示，批量处理时建议注释掉以免阻塞
                cv2.drawChessboardCorners(img, (corner_point_long, corner_point_short), corners2, ret)
                cv2.imshow(f'Processing...', img)
                cv2.waitKey(100) # 显示100ms
                valid_images_count += 1
            else:
                print(f"警告: 图片 {target_path} 未检测到角点，将被忽略。")
    
    cv2.destroyAllWindows()
    print(f"共加载有效图片: {valid_images_count} 张")

    if valid_images_count == 0:
        print("错误: 未找到任何有效图片，程序终止。")
        return None, None

    # 相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print("内参矩阵:\n", mtx)
    print("畸变系数:\n", dist)
    
    # 保存相机参数
    save_camera_params(mtx, dist, save_dir)

    # 读取机器人末端位姿 (从生成的CSV文件)
    tool_pose = np.loadtxt(pose_csv_path, delimiter=',')
    
    # 检查数据对齐
    if tool_pose.shape[1] // 4 < valid_images_count:
        print(f"严重错误: 位姿数量 ({tool_pose.shape[1] // 4}) 少于图片数量 ({valid_images_count})")
        print("请检查 pose.txt 是否有遗漏，或重新采集。")
        # 这里为了防止报错，截断处理，但在实际标定中这是很危险的
        valid_images_count = min(tool_pose.shape[1] // 4, valid_images_count)
        rvecs = rvecs[:valid_images_count]
        tvecs = tvecs[:valid_images_count]

    R_tool = []
    t_tool = []
    # 注意：这里假设 robotToolPose.csv 的列是按图片顺序排列的
    for i in range(valid_images_count):
        R_tool.append(tool_pose[0:3, 4*i : 4*i+3])
        t_tool.append(tool_pose[0:3, 4*i+3])

    # 旋转向量 -> 旋转矩阵
    rotation_matrices = []
    for rvec in rvecs:
        rot_mat, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(rot_mat)

    # --- 手眼标定计算 ---
    print("-" * 30)
    
    # 1. TSAI
    R_tsai, t_tsai = cv2.calibrateHandEye(R_tool, t_tool, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_TSAI)
    print("TSAI Result:\nR:\n", R_tsai, "\nt:\n", t_tsai)

    # 2. PARK
    R_park, t_park = cv2.calibrateHandEye(R_tool, t_tool, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_PARK)
    print("PARK Result:\n", t_park.T)

    # 3. HORAUD
    R_horaud, t_horaud = cv2.calibrateHandEye(R_tool, t_tool, rotation_matrices, tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
    print("HORAUD Result:\n", t_horaud.T)

    # 保存所有结果
    save_calibration_results(R_tsai, t_tsai, R_park, t_park, R_horaud, t_horaud, save_dir)

    return R_tsai, t_tsai

if __name__ == '__main__':
    # ========== 配置路径 ==========
    base_dir = os.path.dirname(__file__)
    
    # 输入：图片文件夹和位姿txt
    images_path = os.path.join(base_dir, "img_eyeinhand_tool2") 
    pose_file_txt = os.path.join(base_dir, "pose_eyeinhand_tool2.txt")
    
    # 输出：结果保存文件夹
    save_dir = os.path.join(base_dir, "result_eye_in_hand_tool2")
    
    # 标定板参数
    corner_point_long = 11      # 长边角点数 (内角点)
    corner_point_short = 8      # 短边角点数
    corner_point_size = 0.015   # 15mm = 0.015m

    # ========== 执行流程 ==========
    # 1. 确保结果目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建结果文件夹: {save_dir}")

    print(f"开始处理... \n图片路径: {images_path}\n位姿文件: {pose_file_txt}")

    # 2. 将 txt 位姿转换为 csv 矩阵格式，并保存到结果文件夹
    pose_csv_path = poses_save_csv(pose_file_txt, save_dir)
    
    # 3. 计算手眼矩阵
    R_result, t_result = compute_T(
        images_path, 
        corner_point_long, 
        corner_point_short, 
        corner_point_size, 
        save_dir,
        pose_csv_path
    )

    print("\n" + "="*50)
    print("标定完成！")
    print(f"所有结果文件已保存至: {save_dir}")
    print("建议优先使用 TSAI 或 PARK 方法的结果。")
    print("="*50)