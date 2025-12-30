import threading
from dobot_api import DobotApiDashboard, DobotApiMove, DobotApiFeedBack, alarmAlarmJsonFile
from time import sleep, time
import re

# 全局变量
current_actual = None
robotMode = 0
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
globalLockValue = threading.Lock()


def ConnectRobot():
    """连接机器人"""
    ip = "192.168.5.1"
    print("正在建立连接...")
    dashboard = DobotApiDashboard(ip, 29999)
    move = DobotApiMove(ip, 30003)
    feedFour = DobotApiFeedBack(ip, 30004)
    print(">.<连接成功>!<")
    return dashboard, move, feedFour


def RunPoint(move: DobotApiMove, point_list: list):
    """运动到目标点"""
    move.MovJ(point_list[0], point_list[1], point_list[2],
              point_list[3], point_list[4], point_list[5])


def GetFeed(feedFour: DobotApiFeedBack):
    """实时获取机器人反馈"""
    global current_actual, robotMode, algorithm_queue, enableStatus_robot, robotErrorState
    
    while True:
        try:
            feedInfo = feedFour.feedBackData()
            if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
                with globalLockValue:
                    robotMode = feedInfo['robot_mode'][0]
                    current_actual = feedInfo["tool_vector_actual"][0]
                    algorithm_queue = feedInfo['run_queued_cmd'][0]
                    enableStatus_robot = feedInfo['enable_status'][0]
                    robotErrorState = feedInfo['error_status'][0]
            sleep(0.01)
        except Exception as e:
            print(f"获取反馈失败: {e}")
            sleep(0.1)


def WaitArrive(point_list, tolerance=3.0, timeout=30.0):
    """等待机器人到达目标点"""
    global current_actual, robotMode, algorithm_queue
    
    print("等待到达...")
    start_time = time()
    sleep(0.3)
    
    while True:
        elapsed_time = time() - start_time
        
        if elapsed_time > timeout:
            print(f"✗ 超时({timeout}秒)")
            return False
        
        with globalLockValue:
            mode = robotMode
            queue = algorithm_queue
            actual = current_actual
        
        # 方法1: 检查坐标距离
        if actual is not None:
            distances = [abs(actual[i] - point_list[i]) for i in range(3)]
            max_diff = max(distances)
            
            if max_diff <= tolerance:
                print(f"✓ 到达目标点 (误差{max_diff:.2f}mm)")
                return True
        
        # 方法2: 检查机器人状态（空闲且队列为空）
        if mode == 5 and queue == 0 and elapsed_time > 0.5:
            print("✓ 到达目标点 (机器人空闲)")
            return True
        
        sleep(0.1)


def ClearRobotError(dashboard: DobotApiDashboard):
    """后台监控并清除机器人错误"""
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()
    
    while True:
        try:
            with globalLockValue:
                error_state = robotErrorState
            
            if error_state:
                numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
                numbers = [int(num) for num in numbers]
                
                if numbers[0] == 0 and len(numbers) > 1:
                    print("\n" + "="*50)
                    print("机器人报警:")
                    
                    for i in numbers[1:]:
                        if i == -2:
                            print(f"  ⚠ 碰撞检测: {i}")
                        else:
                            for item in dataController:
                                if i == item["id"]:
                                    print(f"  ⚠ 控制器 {i}: {item['zh_CN']['description']}")
                                    break
                            for item in dataServo:
                                if i == item["id"]:
                                    print(f"  ⚠ 伺服 {i}: {item['zh_CN']['description']}")
                                    break
                    
                    print("="*50)
                    choose = input("输入1清除错误继续(其他键退出): ")
                    
                    if choose == "1":
                        dashboard.ClearError()
                        sleep(0.5)
                        dashboard.EnableRobot()
                        sleep(0.5)
                        print("✓ 已清除错误")
                    else:
                        exit(0)
                        
        except Exception as e:
            print(f"错误处理异常: {e}")
            
        sleep(5)


def main():
    """主程序"""
    global current_actual
    
    dashboard, move, feedFour = ConnectRobot()
    
    # 启动后台线程
    threading.Thread(target=GetFeed, args=(feedFour,), daemon=True).start()
    threading.Thread(target=ClearRobotError, args=(dashboard,), daemon=True).start()
    
    # 等待反馈数据
    print("等待反馈数据...")
    sleep(2)
    
    with globalLockValue:
        if current_actual is None:
            print("警告: 未收到反馈")
        else:
            print(f"✓ 当前位置: [{current_actual[0]:.2f}, {current_actual[1]:.2f}, {current_actual[2]:.2f}]")
    
    # 使能机器人
    print("\n使能机器人...")
    dashboard.EnableRobot()
    sleep(1)
    
    # 设置运动参数
    dashboard.SpeedFactor(30)
    dashboard.SpeedJ(40)
    dashboard.AccJ(30)
    print("✓ 已设置运动参数(速度30%)")
    
    # 定义往返点位
    point_a = [104.0, -436.0, 463.0, -165.0, 20.0, 121.0]
    point_b = [204.0, -336.0, 663.0, -165.0, 20.0, 121.0]
    
    print("\n" + "="*50)
    print(f"点位A: {point_a[:3]}")
    print(f"点位B: {point_b[:3]}")
    print("="*50)
    
    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n>>> 第 {cycle} 次循环")
            
            # A -> B
            print("→ 运动到点A")
            RunPoint(move, point_a)
            if not WaitArrive(point_a):
                break
            sleep(0.5)
            
            print("→ 运动到点B")
            RunPoint(move, point_b)
            if not WaitArrive(point_b):
                break
            sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n程序异常: {e}")
    finally:
        dashboard.DisableRobot()
        print("程序结束")


if __name__ == '__main__':
    main()