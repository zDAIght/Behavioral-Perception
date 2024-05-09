import sys
import time
from math import pi

import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
import moveit_commander
from std_msgs.msg import String, Float32MultiArray
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from CR_robot import CR_ROBOT, CR_bringup


def pub_joint_msg(robot, pub):
    # 设置要发布的关节名称
    joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    joint_positions = robot.GetAngle()
    joint_positions = [angle * (pi / 180) for angle in joint_positions]
    # rospy.loginfo(joint_positions)
    # rospy.loginfo(time.time())

    joint_state_msg = JointState()
    joint_state_msg.name = [''] * 6
    joint_state_msg.position = [0] * 6
    joint_state_msg.header.stamp = rospy.Time.now()
    for i in range(6):
        joint_state_msg.name[i] = joint_names[i]
        joint_state_msg.position[i] = joint_positions[i]

    # 发布关节状态信息
    pub.publish(joint_state_msg)


def calculate_link_pose(start_point, end_point):
    # 计算连杆的长度（欧式距离）
    distance = np.sqrt(
        (end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2 + (end_point[2] - start_point[2]) ** 2)

    # 两个点的坐标
    point_a = np.array(start_point)
    point_b = np.array(end_point)

    # 计算连接两个点的向量
    link_vector = point_b - point_a

    # 归一化向量作为旋转轴
    rotation_axis = link_vector / np.linalg.norm(link_vector)

    # 计算旋转角度（可以根据需要使用不同的方法来计算）
    # 这里使用arccos来计算角度，然后将角度转换为弧度
    angle_degrees = np.arccos(np.dot(rotation_axis, np.array([1, 0, 1]))) * 180 / np.pi
    angle_radians = np.radians(angle_degrees)

    # 使用旋转轴和角度构造四元数
    rotation = R.from_rotvec(rotation_axis * angle_radians)
    quaternion = rotation.as_quat()

    return distance, quaternion


class MoveitAPIControl:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        # 初始化ROS节点
        rospy.init_node("cr5_moveit_plan")

        # 初始化需要使用move_group控制的机械臂
        self.arm = MoveGroupCommander("cr5_arm")

        # 初始化机械臂控制API
        self.robot_control = CR_ROBOT("192.168.5.1", 30003)
        self.robot_bringup = CR_bringup("192.168.5.1", 29999)

        # 获取终端link的名称
        self.end_effector_link = self.arm.get_end_effector_link()

        # 初始化场景
        self.scene = moveit_commander.PlanningSceneInterface()
        # 清理残留物体
        self.scene.remove_attached_object(self.end_effector_link, 'tool')
        self.scene.remove_world_object('wrist_obstacle')
        self.scene.remove_world_object('elbow_obstacle')
        # 将机械臂末端工具添加到场景中
        tool_size = [0.02, 0.02, 0.17]
        tool = PoseStamped()
        tool.header.frame_id = self.end_effector_link
        tool.pose.position.x = 0
        tool.pose.position.y = 0
        tool.pose.position.z = 0.085
        tool.pose.orientation.w = 1.0
        # 将tool附着到机器人的终端
        self.scene.attach_box(self.end_effector_link, 'tool', tool, tool_size)

        # 设置目标位置所使用的参考坐标系
        self.reference_frame = "base_link"
        self.arm.set_pose_reference_frame("base_link")

        # 设置默认规划器
        self.arm.set_planner_id("BiTRRT")
        # 当运动规划失败后，允许重新规划
        self.arm.allow_replanning(True)

        # 设置位置和姿态的允许误差
        self.arm.set_goal_position_tolerance(0.02)
        self.arm.set_goal_orientation_tolerance(0.02)

        # 设置允许的最大速度和加速度
        self.arm.set_max_acceleration_scaling_factor(1.0)
        self.arm.set_max_velocity_scaling_factor(0.5)

        # 实时发布关节信息
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.rate = rospy.Rate(1.5)

        # 订阅目标点和障碍物信息,以及预设姿态信息
        self.target_pose = PoseStamped()
        self.wrist_obstacle_pose = PoseStamped()
        self.elbow_obstacle_pose = PoseStamped()

        rospy.Subscriber("target_pose", PoseStamped, self.target_callback)
        rospy.Subscriber("obstacle_pose", Float32MultiArray, self.obstacle_callback)
        rospy.Subscriber("target_named", String, self.named_callback)

        while not rospy.is_shutdown():
            pub_joint_msg(self.robot_bringup, self.joint_pub)
            self.rate.sleep()

        rospy.spin()
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def obstacle_callback(self, data):
        obs_poses = data.data
        wrist_pose = [obs_poses[0], obs_poses[1], obs_poses[2]]
        elbow_pose = [obs_poses[3], obs_poses[4], obs_poses[5]]
        elbow_obs_pose = [(wrist_pose[i]+elbow_pose[i])/2 for i in range(3)]

        # elbow_obstacle_length, elbow_obstacle_quaternion = calculate_link_pose(wrist_pose, elbow_pose)
        wrist_obstacle_radius = 0.08
        elbow_obstacle_radius = 0.08

        self.wrist_obstacle_pose.header.frame_id = self.reference_frame
        self.wrist_obstacle_pose.header.stamp = rospy.Time.now()
        self.wrist_obstacle_pose.pose.position.x = wrist_pose[0]
        self.wrist_obstacle_pose.pose.position.y = wrist_pose[1]
        self.wrist_obstacle_pose.pose.position.z = wrist_pose[2]
        self.wrist_obstacle_pose.pose.orientation.x = 0.0
        self.wrist_obstacle_pose.pose.orientation.y = 0.0
        self.wrist_obstacle_pose.pose.orientation.z = 0.0
        self.wrist_obstacle_pose.pose.orientation.w = 1.0

        self.elbow_obstacle_pose.header.frame_id = self.reference_frame
        self.elbow_obstacle_pose.header.stamp = rospy.Time.now()
        self.elbow_obstacle_pose.pose.position.x = elbow_obs_pose[0]
        self.elbow_obstacle_pose.pose.position.y = elbow_obs_pose[1]
        self.elbow_obstacle_pose.pose.position.z = elbow_obs_pose[2]
        # self.elbow_obstacle_pose.pose.orientation.x = elbow_obstacle_quaternion[0]
        # self.elbow_obstacle_pose.pose.orientation.y = elbow_obstacle_quaternion[1]
        # self.elbow_obstacle_pose.pose.orientation.z = elbow_obstacle_quaternion[2]
        # self.elbow_obstacle_pose.pose.orientation.w = elbow_obstacle_quaternion[3]
        self.elbow_obstacle_pose.pose.orientation.x = 0.0
        self.elbow_obstacle_pose.pose.orientation.y = 0.0
        self.elbow_obstacle_pose.pose.orientation.z = 0.0
        self.elbow_obstacle_pose.pose.orientation.w = 1.0

        self.scene.remove_world_object('wrist_obstacle')
        self.scene.remove_world_object('elbow_obstacle')
        time.sleep(0.05)
        self.scene.add_sphere("wrist_obstacle", self.wrist_obstacle_pose, wrist_obstacle_radius)
        self.scene.add_sphere("elbow_obstacle", self.elbow_obstacle_pose, elbow_obstacle_radius)
        # self.scene.add_cylinder("elbow_obstacle", self.elbow_obstacle_pose, elbow_obstacle_length, elbow_obstacle_radius)

    def target_callback(self, data):
        self.target_pose.header.frame_id = self.reference_frame
        self.target_pose.header.stamp = rospy.Time.now()
        self.target_pose.pose = data.pose

        # 设置当前位置为规划的初始位置
        self.arm.set_start_state_to_current_state()
        # 设置机械臂终端运动的目标位姿
        self.arm.set_pose_target(self.target_pose.pose, self.end_effector_link)

        # 请求路径规划计算
        plan = self.arm.plan()
        self.trajectory_plan(plan)
        self.arm.execute(plan[1], wait=True)

    def named_callback(self, data):
        # 设置当前位置为规划的初始位置
        self.arm.set_start_state_to_current_state()
        # 设置机械臂终端运动的目标位姿
        self.arm.set_named_target(data.data)
        # 请求路径规划计算
        plan = self.arm.plan()
        self.trajectory_plan(plan)
        self.arm.execute(plan[1], wait=True)

    def trajectory_plan(self, plan):
        # plan[0]代表规划是否成功(True/False),plan[1]为规划的轨迹集合
        if plan[0]:
            trajectory_all = plan[1]
            points = trajectory_all.joint_trajectory.points
            trajectory = []
            for point in points:
                trajectory.append([i * (180 / pi) for i in point.positions])
            # print(trajectory)
            rospy.logwarn("trajectory points number: " + str(len(trajectory)))

            print("轨迹\n", trajectory, "\n")

            for position in trajectory:
                self.robot_control.ServoJ(position[0], position[1], position[2], position[3], position[4], position[5])
                time.sleep(0.35)
        else:
            rospy.logerr("规划失败")


if __name__ == "__main__":
    try:
        MoveitAPIControl()
    except Exception as e:
        rospy.loginfo("error:%s", e)
