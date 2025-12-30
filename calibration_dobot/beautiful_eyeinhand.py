#! /usr/bin/env python
# coding=utf-8
"""
基于Apriltag的抓取验证脚本 (Dobot V3 - 6D 对齐 + 拖拽救援版)
新增功能：
1. 按 'd' 键：在 [拖拽模式] 和 [锁定模式] 之间切换。
2. 拖拽期间禁止执行自动对齐。
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
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def mat_to_euler(R):
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
    T = np.eye(4)
    T[:3, :3] = euler_to_mat(pose[3], pose[4], pose[5])
    T[:3, 3] = pose[:3]
    return T

def mat_to_pose(T):
    rx, ry, rz = mat_to_euler(T[:3, :3])
    return [T[0, 3], T[1, 3], T[2, 3], rx, ry, rz]

# ========== 2. 可视化工具 ==========

def draw_axes(img, camera_matrix, dist_coeffs, rvec, tvec, length):
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3) # X 红
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3) # Y 绿
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3) # Z 蓝
    return img

# ========== 3. Dobot 通信类 (新增拖拽控制) ==========

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
            # 安全退出：关闭拖拽，回Tool 0
            self.send_dash("StopDrag()")
            self.send_dash("Tool(0)")
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

    def get_pose(self):
        res = self.send_dash("GetPose()")
        if res and res.startswith("0") and '{' in res:
            content = res.split('{')[1].split('}')[0]
            return [float(v) for v in content.split(',')]
        return None

    def mov_j(self, x, y, z, rx, ry, rz):
        cmd = f"MovJ({x:.4f},{y:.4f},{z:.4f},{rx:.4f},{ry:.4f},{rz:.4f})"
        print(f">> [Motion] 6D对齐: {cmd}")
        self.send_move(cmd)
        self.send_move("Sync()")
        time.sleep(0.5)

    def set_drag(self, enable):
        """控制拖拽模式开关"""
        if enable:
            print(">>> 正在开启拖拽模式...")
            res = self.send_dash("StartDrag()")
        else:
            print(">>> 正在锁定机械臂...")
            res = self.send_dash("StopDrag()")
        return res

# ========== 4. 主逻辑 ==========

def load_calibration_data(result_dir):
    try:
        mtx = np.loadtxt(os.path.join(result_dir, 'camera_matrix.txt'))
        dist = np.loadtxt(os.path.join(result_dir, 'distortion_coefficients.txt')).reshape(-1, 1)
        with open(os.path.join(result_dir, 'calibration_result_tsai.txt'), 'r') as f:
            lines = [l.strip() for l in f.readlines() if not l.startswith('#')]
        R = np.array([[float(x) for x in l.split()] for l in lines[:3]])
        t = np.array([float(x) for x in lines[3].split()]).reshape(3, 1)
        return mtx, dist, R, t
    except Exception as e:
        print(f"标定文件加载失败: {e}")
        return None, None, None, None

def main():
    # --- 配置 ---
    robot_ip = "192.168.5.1"
    tag_size = 0.03
    
    base_dir = os.path.dirname(__file__)
    result_dir = os.path.join(base_dir, "result_eye_in_hand_tool2")
    
    # 1. 加载参数
    mtx, dist, R_cam_tcp, t_cam_tcp = load_calibration_data(result_dir)
    if mtx is None: return

    T_cam_tcp = np.eye(4)
    T_cam_tcp[:3, :3] = R_cam_tcp
    T_cam_tcp[:3, 3] = t_cam_tcp.flatten()
    print("标定数据加载完毕。")

    # 2. 连接机器人
    bot = DobotClient(robot_ip)
    if not bot.connect(): return
    
    bot.send_dash("EnableRobot()")
    bot.send_dash("Tool(2)")  
    time.sleep(1)

    # 3. 视觉循环
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(cfg)
    
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

    # === 状态标志位 ===
    is_drag_mode = False 

    print("\n" + "="*40)
    print("=== Dobot 6D 强行对齐 + 拖拽救援 ===")
    print(" [空格] 执行自动对齐 (仅锁定状态可用)")
    print(" [ d  ] 切换 拖拽/锁定 模式")
    print(" [ESC ] 退出程序")
    print("="*40 + "\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)
            
            target_pose_dobot = None 
            
            # --- 视觉处理与绘制 ---
            for d in detections:
                cv2.polylines(img, [d.corners.astype(int)], True, (255, 255, 255), 2)
                
                obj_pts = np.array([[-tag_size/2, -tag_size/2, 0], [tag_size/2, -tag_size/2, 0], 
                                    [tag_size/2, tag_size/2, 0], [-tag_size/2, tag_size/2, 0]], dtype=np.float32)
                ret, rvec, tvec = cv2.solvePnP(obj_pts, d.corners.astype(np.float32), mtx, dist)
                
                if ret:
                    draw_axes(img, mtx, dist, rvec, tvec, length=tag_size)
                    
                    # 仅在非拖拽模式下进行复杂计算（节省性能，虽然影响不大）
                    T_tag_cam = np.eye(4)
                    T_tag_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
                    T_tag_cam[:3, 3] = tvec.flatten()
                    
                    curr = bot.get_pose()
                    if curr:
                        curr_m_rad = [curr[0]/1000.0, curr[1]/1000.0, curr[2]/1000.0, 
                                      math.radians(curr[3]), math.radians(curr[4]), math.radians(curr[5])]
                        T_tcp_base = pose_to_mat(curr_m_rad)
                        T_tag_base = T_tcp_base @ T_cam_tcp @ T_tag_cam
                        tgt_m_rad = mat_to_pose(T_tag_base)
                        
                        target_pose_dobot = [
                            tgt_m_rad[0] * 1000.0, tgt_m_rad[1] * 1000.0, tgt_m_rad[2] * 1000.0,
                            math.degrees(tgt_m_rad[3]), math.degrees(tgt_m_rad[4]), math.degrees(tgt_m_rad[5])
                        ]
                        
                        info = f"TGT: {target_pose_dobot[0]:.0f},{target_pose_dobot[1]:.0f},{target_pose_dobot[2]:.0f}"
                        cv2.putText(img, info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- UI 状态显示 ---
            if is_drag_mode:
                # 拖拽模式：显示醒目的绿色提示
                cv2.putText(img, "[ DRAG MODE ON ] - Manual Adjust", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(img, "Press 'd' to LOCK", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            else:
                # 锁定模式：显示待命提示
                cv2.putText(img, "[ LOCKED ] - Ready for Space", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Dobot 6D Experiment", img)
            key = cv2.waitKey(1) & 0xFF
            
            # --- 交互逻辑 ---
            if key == ord('d'):
                # 切换拖拽状态
                is_drag_mode = not is_drag_mode
                bot.set_drag(is_drag_mode)
                # 防止切换瞬间误触其他键
                time.sleep(0.2) 

            elif key == ord(' '):
                # 只有在非拖拽模式下才允许自动运动
                if is_drag_mode:
                    print(">>> [警告] 请先按 'd' 锁定机械臂，再执行自动对齐！")
                else:
                    if target_pose_dobot:
                        tx, ty, tz = target_pose_dobot[0], target_pose_dobot[1], target_pose_dobot[2]
                        trx, try_, trz = target_pose_dobot[3], target_pose_dobot[4], target_pose_dobot[5]

                        print("-" * 40)
                        print(">>> [EXECUTE] 执行 6D 全姿态对齐")
                        bot.mov_j(tx, ty, tz, trx, try_, trz)
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