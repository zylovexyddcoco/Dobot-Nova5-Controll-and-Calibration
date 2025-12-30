import os
import cv2
import pyrealsense2 as rs
import numpy as np
import time
import socket
import math

# ========== Dobot V3 通信类 ==========
class DobotClient:
    def __init__(self, ip, port=29999):
        self.ip = ip
        self.port = port
        self.socket_dobot = None

    def connect(self):
        try:
            self.socket_dobot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket_dobot.settimeout(5)
            self.socket_dobot.connect((self.ip, self.port))
            print(f"已连接到机械臂控制器: {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def close(self):
        if self.socket_dobot:
            # 退出前强制退出拖拽模式，并恢复到法兰坐标系(Tool 0)
            self.send_command("StopDrag()")
            self.send_command("Tool(0)") 
            self.socket_dobot.close()
            print("连接已断开")

    def send_command(self, string):
        """发送指令并接收反馈"""
        if not self.socket_dobot:
            return None
        try:
            self.socket_dobot.send(str.encode(string + '\n'))
            data = self.socket_dobot.recv(1024).decode("utf-8")
            return data
        except Exception as e:
            print(f"通信错误: {e}")
            return None

    def set_drag_status(self, enable):
        """控制拖拽模式"""
        if enable:
            res = self.send_command("StartDrag()")
            if res and "0" in res.split(',')[0]:
                print(">>> [拖拽模式] 已开启 (您可以手动移动机械臂)")
            else:
                print(f"开启拖拽失败: {res}")
        else:
            res = self.send_command("StopDrag()")
            if res and "0" in res.split(',')[0]:
                print(">>> [锁定模式] 已开启 (机械臂保持静止)")
            else:
                print(f"关闭拖拽失败: {res}")

    def activate_tool(self, tool_index=1):
        """
        激活指定的工具坐标系索引
        """
        # 发送 Tool(index) 告诉控制器使用哪个工具参数
        cmd_use = f"Tool({tool_index})"
        res_use = self.send_command(cmd_use)
        
        if res_use and "0" in res_use.split(',')[0]:
            print(f"已激活工具坐标系索引: Tool {tool_index}")
        else:
            print(f"切换工具坐标系失败: {res_use}")

    def get_pose_converted(self):
        """
        获取位姿并转换为 [米, 弧度] 格式 (适配 eyeInHand.py)
        """
        res = self.send_command("GetPose()")
        if res and res.startswith("0") and '{' in res:
            try:
                content = res.split('{')[1].split('}')[0]
                values = [float(v) for v in content.split(',')]
                if len(values) == 6:
                    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = values
                    # 转换单位：毫米->米，度->弧度
                    return [x_mm/1000.0, y_mm/1000.0, z_mm/1000.0, 
                            math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)]
            except Exception as e:
                print(f"数据解析异常: {e}")
        return None

# ========== 图像与文件保存函数 ==========

def take_photo(color_frame, chess_path):
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imwrite(chess_path, color_image)
    print(f"图片已保存: {chess_path}")

def save_pose(pose_file_path, pose):
    with open(pose_file_path, 'a', encoding='utf-8') as f:
        pose_str = ','.join([f"{x:.6f}" for x in pose])
        f.write(pose_str + '\n')
    print(f"位姿已追加: {pose_str}")

# ========== 主程序 ==========

def main():
    # --- 1. 参数配置 ---
    robot_ip = "192.168.5.1"
    robot_port = 29999
    
    # 既然您已在软件中设置，这里不需要再写 tcp_offset 变量了
    # 只要记住您在软件里用的是 Tool 几 (默认假设是 Tool 1)
    target_tool_index = 0
    
    base_dir = os.path.dirname(__file__)
    photo_dir = os.path.join(base_dir, 'img_eyetohand_tool0_bizhang')
    pose_file_path = os.path.join(base_dir, 'pose_eyetohand_tool0_bizhang.txt')

    if not os.path.exists(photo_dir):
        os.makedirs(photo_dir)

    # --- 2. 连接与初始化 ---
    dobot = DobotClient(robot_ip, robot_port)
    if not dobot.connect():
        return

    dobot.send_command("ClearError()")
    dobot.send_command("EnableRobot()") 
    time.sleep(1) 
    
    # 关键步骤：告诉控制器切换到您设置好的 Tool 1
    # 这样 GetPose 返回的就是经过 TCP 偏移后的坐标了
    dobot.activate_tool(target_tool_index)
    time.sleep(0.5)

    # --- 3. 开启相机 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    print("\n" + "="*50)
    print(" Dobot 手眼标定采集程序 (Eye-in-Hand)")
    print(f" 当前使用工具坐标系: Tool {target_tool_index}")
    print(" [d] 开启拖拽 | [l] 锁定(拍照前必按) | [空格] 保存 | [ESC] 退出")
    print("="*50 + "\n")

    try:
        photo_count = 0
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.putText(color_image, "Keys: 'd'=Drag, 'l'=Lock, Space=Save, ESC=Exit", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration Collect', color_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):
                dobot.set_drag_status(True)
            elif key == ord('l'):
                dobot.set_drag_status(False)
            elif key == ord(' '):
                pose = dobot.get_pose_converted()
                if pose:
                    img_name = os.path.join(photo_dir, f"{photo_count}.jpg")
                    take_photo(color_frame, img_name)
                    save_pose(pose_file_path, pose)
                    photo_count += 1
                    print(f"--> 第 {photo_count} 组数据采集完成\n")
                else:
                    print("!!! 获取位姿失败\n")
            elif key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        dobot.close()

if __name__ == '__main__':
    main()