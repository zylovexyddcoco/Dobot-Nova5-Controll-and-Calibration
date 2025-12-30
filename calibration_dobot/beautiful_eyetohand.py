#! /usr/bin/env python
# coding=utf-8
"""
眼在手外 (Eye-to-Hand) 验证脚本 - 完整版
场景：相机固定在外部，机械臂(TCP)去移动到 Tag 的位置进行验证
逻辑：T_tag_base = T_base_cam (标定结果) @ T_tag_cam (实时检测)
"""

import os
import cv2
import pyrealsense2 as rs
import numpy as np
import time
import math
import socket
import apriltag

# ========== 1. 基础数学工具 ==========

def euler_to_mat(rx, ry, rz):
    """欧拉角(rad) 转 旋转矩阵 (ZYX顺序)"""
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def mat_to_euler(R):
    """旋转矩阵 转 欧拉角(rad) (ZYX顺序)"""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy < 1e-6:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0
    else:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    return rx, ry, rz

def pose_to_mat(pose):
    """[x,y,z,rx,ry,rz] -> 4x4 T矩阵"""
    T = np.eye(4)
    T[:3, :3] = euler_to_mat(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T

def mat_to_pose(T):
    """4x4 T矩阵 -> [x,y,z,rx,ry,rz]"""
    rx, ry, rz = mat_to_euler(T[:3, :3])
    return [T[0, 3], T[1, 3], T[2, 3], rx, ry, rz]

# ========== 2. 可视化工具 ==========

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length):
    """在 Tag 中心绘制 XYZ 坐标轴"""
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X 红
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y 绿
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z 蓝
    return img

# ========== 3. Dobot 通信类 ==========

class DobotClient:
    def __init__(self, ip):
        self.ip = ip
        self.port_dash = 29999
        self.port_move = 30003
        self.sock_dash = None
        self.sock_move = None

    def connect(self):
        try:
            self.sock_dash = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock_dash.settimeout(5)
            self.sock_dash.connect((self.ip, self.port_dash))
            
            self.sock_move = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock_move.settimeout(5)
            self.sock_move.connect((self.ip, self.port_move))
            print(f"Dobot 连接成功: {self.ip}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def close(self):
        if self.sock_dash:
            self.send_dash("StopDrag()")
            self.send_dash("Tool(3)")
            self.sock_dash.close()
        if self.sock_move: self.sock_move.close()

    def send_dash(self, cmd):
        if not self.sock_dash: return None
        try:
            self.sock_dash.send(str.encode(cmd + '\n'))
            return self.sock_dash.recv(1024).decode("utf-8")
        except: return None

    def send_move(self, cmd):
        if not self.sock_move: return
        try:
            self.sock_move.send(str.encode(cmd + '\n'))
        except Exception as e: print(f"Motion指令失败: {e}")

    def mov_j(self, x, y, z, rx, ry, rz):
        """发送 MovJ (单位: mm, degree)"""
        cmd = f"MovJ({x:.4f},{y:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f})"
        print(f">> [Motion] 验证移动: {cmd}")
        self.send_move(cmd)
        self.send_move("Sync()")
        time.sleep(0.5)

    def set_drag(self, enable):
        """控制拖拽模式"""
        if enable:
            print(">>> 开启拖拽...")
            self.send_dash("StartDrag()")
        else:
            print(">>> 锁定机械臂...")
            self.send_dash("StopDrag()")

# ========== 4. 辅助加载函数 ==========

def get_realsense_intrinsics(profile):
    """获取官方内参"""
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    dist = np.array(intr.coeffs)
    return mtx, dist

def load_hand_eye_result(result_dir):
    """加载 result_eye_to_hand 文件夹下的 TSAI 结果"""
    calib_file = os.path.join(result_dir, 'calibration_result_tsai.txt')
    try:
        with open(calib_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if not l.startswith('#')]
        R = np.array([[float(x) for x in l.split()] for l in lines[:3]])
        t = np.array([float(x) for x in lines[3].split()]).reshape(3, 1)
        return R, t
    except Exception as e:
        print(f"标定文件加载失败: {e}")
        return None, None

# ========== 5. 主程序 ==========

def main():
    # --- 配置 ---
    robot_ip = "192.168.5.1" # 请修改为您的IP
    tag_size = 0.03           # Tag 边长 (米)
    
    # 路径：指向眼在手外计算结果
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result_eye_to_hand_tool0_bizhang")
    
    # 1. 加载标定结果 T_base_cam
    R_base_cam, t_base_cam = load_hand_eye_result(result_dir)
    if R_base_cam is None: 
        print(f"错误：未在 {result_dir} 找到标定结果，请先运行 eyeToHand_calc.py")
        return

    # 构造变换矩阵 (Base -> Cam)
    T_base_cam = np.eye(4)
    T_base_cam[:3, :3] = R_base_cam
    T_base_cam[:3, 3] = t_base_cam.flatten()
    print("标定数据加载完毕 (Camera in Base Frame)。")

    # 2. 连接机器人
    bot = DobotClient(robot_ip)
    if not bot.connect(): return
    
    bot.send_dash("EnableRobot()")
    
    # 【关键设置】: 验证时，你要用哪个工具去碰 Tag？
    # 假设你在软件里把【夹爪尖端】设为了 Tool 1
    # 那么这里激活 Tool 1，机械臂就会把夹爪尖端送到 Tag 的坐标去
    target_tool_index = 0
    bot.send_dash(f"Tool({target_tool_index})")
    time.sleep(1)

    # 3. 开启相机 (官方内参)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    mtx, dist = get_realsense_intrinsics(profile)
    
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
    is_drag_mode = False 

    print("\n" + "="*50)
    print("=== Dobot 眼在手外 (Eye-to-Hand) 验证 ===")
    print(f" 使用工具: Tool {target_tool_index}")
    print(" [空格] 执行 MovJ (去碰 Tag)")
    print(" [ d  ] 切换 拖拽/锁定")
    print(" [ESC ] 退出")
    print("="*50 + "\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            
            target_pose_dobot = None # (mm, deg)
            
            for d in detections:
                # 绘制边框
                cv2.polylines(img, [d.corners.astype(int)], True, (255, 255, 255), 2)
                
                # --- PnP 解算 ---
                obj_pts = np.array([[-tag_size/2, -tag_size/2, 0], [tag_size/2, -tag_size/2, 0], 
                                    [tag_size/2, tag_size/2, 0], [-tag_size/2, tag_size/2, 0]], dtype=np.float32)
                ret, rvec, tvec = cv2.solvePnP(obj_pts, d.corners.astype(np.float32), mtx, dist)
                
                if ret:
                    # 绘制坐标轴 (红X 绿Y 蓝Z)
                    draw_axes(img, mtx, dist, rvec, tvec, length=tag_size)
                    
                    if not is_drag_mode:
                        # 1. T_tag_cam (Tag 在相机坐标系)
                        T_tag_cam = np.eye(4)
                        T_tag_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
                        T_tag_cam[:3, 3] = tvec.flatten()
                        
                        # 2. 计算绝对坐标 T_tag_base
                        # Eye-to-Hand 公式: T_tag_base = T_base_cam @ T_tag_cam
                        # (注意：此处完全不需要读取机械臂当前位姿，因为相机是固定的！)
                        T_tag_base = T_base_cam @ T_tag_cam
                        
                        # 3. 转换回 Dobot 单位 (mm, deg)
                        tgt_m_rad = mat_to_pose(T_tag_base)
                        
                        target_pose_dobot = [
                            tgt_m_rad[0] * 1000.0,
                            tgt_m_rad[1] * 1000.0,
                            tgt_m_rad[2] * 1000.0,
                            math.degrees(tgt_m_rad[3]),
                            math.degrees(tgt_m_rad[4]),
                            math.degrees(tgt_m_rad[5])
                        ]
                        
                        info = f"Base XYZ: {target_pose_dobot[0]:.0f},{target_pose_dobot[1]:.0f},{target_pose_dobot[2]:.0f}"
                        cv2.putText(img, info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # UI 状态
            if is_drag_mode:
                cv2.putText(img, "[ DRAG MODE ]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.putText(img, "[ LOCKED ]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Eye-to-Hand Verify", img)
            key = cv2.waitKey(1) & 0xFF
            
            # --- 交互 ---
            if key == ord('d'):
                is_drag_mode = not is_drag_mode
                bot.set_drag(is_drag_mode)
                time.sleep(0.2)
            
            elif key == ord(' '):
                if is_drag_mode:
                    print(">>> 请先锁定机械臂 (按 'd')")
                elif target_pose_dobot:
                    print("-" * 40)
                    print(f"目标绝对坐标: {target_pose_dobot}")
                    # 直接去 Tag 位置
                    # 机械臂会驱动 Tool 3(TCP) 去重合 Tag
                    bot.mov_j(*target_pose_dobot)
                else:
                    print("未检测到 Tag")
            
            elif key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        bot.close()

if __name__ == '__main__':
    main()