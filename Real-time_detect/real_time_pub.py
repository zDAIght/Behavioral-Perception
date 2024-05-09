import sys
import numpy as np
import rospy
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
import cv2
import time

target_path = "../"
sys.path.append(target_path)

from utils.realsenseD455 import Camera
from yolo_pose.detect import YoloPoseDetect



def real_time_display():
    cv2.namedWindow('Real-time camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('Real-time camera', 640, 480)

    start_time = time.time()
    sign = 1
    while not rospy.is_shutdown():
        cur_time = time.time()
        color_img, depth_img = cam.get_data()

        # yolo-pose添加骨骼点
        _, key_img = pose_detect.detect(color_img)

        rgbd_img = np.empty((depth_img.shape[0], depth_img.shape[1], 4), dtype=np.uint8)
        rgbd_img[:, :, :3] = color_img
        rgbd_img[:, :, 3] = depth_img

        rgbd_msg = bridge.cv2_to_imgmsg(rgbd_img, encoding="rgba8")

        # color_height, color_width, color_channels = color_img.shape
        # depth_height, depth_width = depth_img.shape
        #
        # color_img_msg = bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
        # depth_img_msg = bridge.cv2_to_imgmsg(depth_img, encoding="32FC1")
        #
        # color_img_msg.header.stamp = rospy.Time.now()
        # color_img_msg.width = color_width
        # color_img_msg.height = color_height
        #
        # depth_img_msg.header.stamp = rospy.Time.now()
        # depth_img_msg.width = depth_width
        # depth_img_msg.height = depth_height

        # if sign == 1:
        #     image_pub.publish(rgbd_msg)  # 发布图片
        #     sign = 0
        # else:
        #     sign = 1
        image_pub.publish(rgbd_msg)  # 发布图片
        # print("cost time:", end - cur_time)  # 看一下每一帧的执行时间，从而确定合适的rate
        rospy.Rate(25)  # 10hz

        fps = 1 / (cur_time - start_time)

        cv2.putText(key_img, "Real-time FPS {0}".format(float('%.1f' % fps)), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Real-time camera', key_img)

        start_time = cur_time

        if cv2.waitKey(1) == ord('q'):
            sign = Bool(data=False)
            sign_pub.publish(sign)
            # 退出程序
            break
    print("关闭相机")
    cv2.destroyWindow('Real-time camera')


if __name__ == '__main__':
    rospy.init_node("real-time publish", anonymous=True)
    image_pub = rospy.Publisher('real-time_image', Image, queue_size=10)
    sign_pub = rospy.Publisher('video_stream', Bool, queue_size=1)
    bridge = cv_bridge.CvBridge()
    key_points = Float32MultiArray()
    cam = Camera()

    # yolo-pose参数设置
    dev = '0'
    wgt = '../yolo_pose/last.pt'
    pose_detect = YoloPoseDetect(dev, wgt)
    print("Yolo-Pose模型以导入")

    real_time_display()
