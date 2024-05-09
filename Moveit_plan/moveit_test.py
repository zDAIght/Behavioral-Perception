#!/usr/bin/env python
import time
import rospy
from geometry_msgs.msg import PoseStamped


def publish_pose():
    rospy.init_node('publish_pose_node', anonymous=True)
    obs_pub = rospy.Publisher("obstacle_pose", PoseStamped, queue_size=10)
    tar_pub = rospy.Publisher("target_pose", PoseStamped, queue_size=10)
    time.sleep(1)

    # 创建一个PoseStamped类型的消息并设置其数据
    obstacle_pose = PoseStamped()
    obstacle_pose.header.stamp = rospy.Time.now()
    obstacle_pose.pose.position.x = -0.35
    obstacle_pose.pose.position.y = -0.14
    obstacle_pose.pose.position.z = 0.30
    obstacle_pose.pose.orientation.x = 0.0
    obstacle_pose.pose.orientation.y = 0.0
    obstacle_pose.pose.orientation.z = 0.0
    obstacle_pose.pose.orientation.w = 1.0

    # 创建一个PoseStamped类型的消息并设置其数据
    target_pose = PoseStamped()
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x = -0.50
    target_pose.pose.position.y = -0.14
    target_pose.pose.position.z = 0.45
    target_pose.pose.orientation.x = 0.9062748444355868
    target_pose.pose.orientation.y = 0.4225988475743994
    target_pose.pose.orientation.z = 0.007907262606498518
    target_pose.pose.orientation.w = 0.0036872171233396376

    '''
    pose: 
      position: 
        x: -0.22080885099371098
        y: -0.14099918892064084
        z: 0.45176306557506185
      orientation: 
        x: 0.9062748444355868
        y: 0.4225988475743994
        z: 0.007907262606498518
        w: 0.0036872171233396376
    '''

    # 发布消息
    obs_pub.publish(obstacle_pose)
    time.sleep(1)
    tar_pub.publish(target_pose)


if __name__ == '__main__':
    try:
        publish_pose()
    except rospy.ROSInterruptException:
        pass