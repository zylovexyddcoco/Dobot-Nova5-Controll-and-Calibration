#! /usr/bin/env python
# coding=utf-8

import socket
import time

class DobotStartup:
    def __init__(self, ip, port=29999):
        self.ip = ip
        self.port = port
        self.sock = None

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.ip, self.port))
            print(f"[连接成功] {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[连接失败] {e}")
            return False

    def close(self):
        if self.sock: self.sock.close()

    def send_cmd(self, cmd):
        if not self.sock: return None
        try:
            self.sock.send((cmd + "\n").encode('utf-8'))
            return self.sock.recv(1024).decode('utf-8').strip()
        except Exception as e:
            print(f"  通信错误: {e}")
            return None

    def get_robot_mode(self):
        # 获取状态码: 3=未上电, 4=未使能, 5=使能空闲, 9=报警
        res = self.send_cmd("RobotMode()")
        if res and "{" in res:
            try:
                return int(res.split('{')[1].split('}')[0])
            except: pass
        return -1

    def wait_for_power_on(self, timeout=50):
        """循环等待直到状态不再是 3 (未上电)"""
        print(f"  正在等待上电完成 (超时 {timeout}秒)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            mode = self.get_robot_mode()
            if mode != 3 and mode != -1:
                print(f"  上电完成！当前状态: {mode}")
                return True
            time.sleep(1)
            print(f"  ... 等待中 (状态: {mode})")
        print("  ❌ 上电超时！")
        return False

    def wait_for_enable(self, timeout=10):
        """循环等待直到状态变为 5 (使能)"""
        print(f"  正在等待使能完成 (超时 {timeout}秒)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            mode = self.get_robot_mode()
            if mode == 5:
                print("  ✅ 机器人已使能 (Status 5)")
                return True
            elif mode == 9:
                print("  ❌ 机器人进入报错状态 (Status 9)，请检查急停按钮！")
                return False
            time.sleep(0.5)
        print("  ❌ 使能超时！")
        return False

    def run(self):
        print("=== Dobot 启动流程 (智能等待版) ===")
        
        # 1. 清除错误
        print(">>> 1. 清除报警")
        self.send_cmd("ClearError()")
        time.sleep(0.5)

        # 2. 上电逻辑
        print(">>> 2. 检查电源")
        mode = self.get_robot_mode()
        if mode == 3: # 未上电
            print("  检测到未上电，发送 PowerOn()...")
            self.send_cmd("PowerOn()")
            # 关键修改：循环等待直到状态改变，而不是死等10秒
            if not self.wait_for_power_on(): return 
        else:
            print(f"  已上电 (状态 {mode})，跳过 PowerOn")

        # 3. 使能逻辑
        print(">>> 3. 使能机器人")
        self.send_cmd("EnableRobot()")
        if not self.wait_for_enable(): return

        # 4. 设置参数
        print(">>> 4. 重置参数 (速度50%, Tool 0)")
        self.send_cmd("SpeedFactor(50)")
        self.send_cmd("Tool(0)")
        self.send_cmd("User(0)")

        print("\n🎉 初始化全部完成！可以运行标定脚本了。")

if __name__ == "__main__":
    # 请确认 IP 是否正确
    bot = DobotStartup("192.168.5.1") 
    if bot.connect():
        try:
            bot.run()
        finally:
            bot.close()