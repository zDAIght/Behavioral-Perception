import sys
import time

import copy
import rospy
import numpy as np
import torch
import torch.nn.functional as F
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from sklearn.preprocessing import LabelEncoder

import cv2

target_path = "../"
sys.path.append(target_path)
from model.resnet_rnn import ClassifyRegressNet
from utils.function import image_convert_from_img, action_target
from yolo_pose.detect import YoloPoseDetect

images = []
# 指定字体、颜色和大小等参数
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)  # 白色文字
size = 1  # 字体大小
thickness = 2  # 文字线宽
add_sign, exit_sign = True, False
# 信号
pub_sign = False

cam_intrinsics = np.array([380.15350342, 0., 316.6630249, 0., 379.74990845, 239.63673401, 0, 0, 1]).reshape(3, 3)
cam_pose = np.loadtxt('../cam_pose/camera_pose.txt', delimiter=' ')
cam_depth_scale = np.loadtxt('../cam_pose/camera_depth_scale.txt', delimiter=' ')


def pub_sign_callback(data):
    global pub_sign
    pub_sign = True


def bool_callback(data):
    global exit_sign
    # sign = data.data
    exit_sign = True
    rospy.signal_shutdown("Exiting ...")


def image_callback(data):
    global bridge
    cv_img = bridge.imgmsg_to_cv2(data, "rgba8")
    image = [cv_img[:, :, :3], cv_img[:, :, 3]]
    if add_sign:
        images.append(image)


def detect():
    global images, wrist_pixel_poses, add_sign, pub_sign

    if len(images) == 28:
        add_sign = False
        print("\n连续图像帧数为：{}".format(len(images)))

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        img = copy.deepcopy(image)
        # 在图片中心位置写入文本
        text = 'Waiting for the video stream ...'
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=size, thickness=thickness)[0]
        x = (img.shape[1] - text_width) // 2
        y = (img.shape[0] + text_height) // 2
        cv2.putText(img, text, (x, y), font, size, color, thickness)
        cv2.imshow('action detection', img)
        cv2.waitKey(100)

        '''识别图像'''
        key_points_images, wrist_pixel_poses = [], []
        for img in images:
            key_points, key_rgb = pose_detect.detect(img[0])
            key_points_images.append([key_rgb, img[1]])
            wrist_pixel_poses.append(key_points['right_wrist'])
        print("\n手腕关键点：{}\n{}".format(len(wrist_pixel_poses), wrist_pixel_poses))
        image28 = image_convert_from_img(key_points_images, use_depth)

        with torch.no_grad():
            if is_regress:
                output_class, output_regress = net(image28)
            else:
                output_class = net(image28)[0]
            output_class = F.softmax(output_class, dim=1)
            [score, cls_label_pred] = output_class.max(1, keepdim=True)
            action = pecle.inverse_transform([int(cls_label_pred.item())])[0]
            score = np.array(score.item())
            print("预测动作：{}, 置信度：{}".format(action, score))

            if is_regress:
                reg_label_pred = output_regress.cpu().data.squeeze().numpy()
                reg_label_pred = np.round(np.resize(reg_label_pred, (28, 3)), decimals=4)
                print("预测轨迹为: ")
                print(reg_label_pred)

        '''显示序列图像的最后一帧和预测结果'''
        pred_action = 'Action prediction : {}'.format(action)
        pred_score = 'Action Score : {:.3f}'.format(score)  # - np.random.uniform(0, 0.1)
        disp_image = copy.deepcopy(key_points_images[-1][0])
        cv2.putText(disp_image, pred_action, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(disp_image, pred_score, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('action detection', disp_image)
        cv2.waitKey(500)

        '''根据预测动作确定机械臂目标点坐标'''
        if action in action_target.keys():
            # target_pose = action_target[action]
            target_pose = action_target["TakeT-tube"]

            '''选择模型预测轨迹位置'''

            '''将预测位置坐标换算至机械臂坐标系'''
            x, y = int(wrist_pixel_poses[-1][0]), int(wrist_pixel_poses[-1][1])
            camera_depth_img = key_points_images[-1][1] * 1000.
            camera_z = camera_depth_img[y][x] * cam_depth_scale
            camera_x = np.multiply(x - cam_intrinsics[0][2], camera_z / cam_intrinsics[0][0])
            camera_y = np.multiply(y - cam_intrinsics[1][2], camera_z / cam_intrinsics[1][1])
            if camera_z == 0:
                assert ValueError('z-pose error')
            camera_point = np.asarray([camera_x, camera_y, camera_z])
            camera_point.shape = (3, 1)
            camera2robot = cam_pose
            world_position = np.dot(camera2robot[0:3, 0:3], camera_point) + camera2robot[0:3, 3:]
            world_position = world_position[0:3, 0]
            print('world_pose: ', world_position)
            # world_position[0] = -0.3
            # world_position[1] = np.random.rand() * 0.05
            # world_position[2] = 0.4 + (np.random.rand() * 0.05)

            '''启用Moveit控制'''
            if pub_sign:
                '''发布手腕障碍物预测位置'''
                obstacle_pose = PoseStamped()
                obstacle_pose.header.stamp = rospy.Time.now()
                obstacle_pose.pose.position.x = world_position[0]
                obstacle_pose.pose.position.y = world_position[1]
                obstacle_pose.pose.position.z = world_position[2]
                obstacle_pose.pose.orientation.x = 0.0
                obstacle_pose.pose.orientation.y = 0.0
                obstacle_pose.pose.orientation.z = 0.0
                obstacle_pose.pose.orientation.w = 1.0
                obs_pub.publish(obstacle_pose)

                time.sleep(0.5)

                '''机械臂根据预测动作执行相应的动作'''
                target_position = PoseStamped()
                target_position.header.stamp = rospy.Time.now()
                target_position.pose.position.x = target_pose[0]
                target_position.pose.position.y = target_pose[1]
                target_position.pose.position.z = target_pose[2]
                target_position.pose.orientation.x = 0.9062748444355868
                target_position.pose.orientation.y = 0.4225988475743994
                target_position.pose.orientation.z = 0.007907262606498518
                target_position.pose.orientation.w = 0.0036872171233396376
                # tar_pub.publish(target_position)

                pub_sign = False

        images, wrist_pixel_poses = [], []
        add_sign = True


if __name__ == '__main__':
    rospy.init_node('img_process_node', anonymous=True)
    cv2.namedWindow("action detection", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow("action detection", 640, 480)
    bridge = CvBridge()

    # 设置网络
    model_path = '../pec_model_ckpt/2023-07-15-10:02:42/ClassifyNet_0.903.pth'

    action_label_file = "../pec_action_label.txt"
    with open(action_label_file, 'r') as f:
        action_list = []
        action_data = f.readlines()
        for action in action_data:
            action_list.append(action.replace('\n', ''))
    pecle = LabelEncoder()
    pecle.fit(action_list)  # 将动作编码

    class_num = len(action_list)  # 分类总数
    regress_num = 28 * 3  # 轨迹回归数量
    is_regress = False
    use_depth = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # yolo-pose参数设置
    dev = '0'
    wgt = '../yolo_pose/last.pt'
    pose_detect = YoloPoseDetect(dev, wgt)
    print("Yolo-Pose模型以导入")

    # 导入网络
    print("导入网络 ...")
    net = ClassifyRegressNet(fc_hidden1=1024, fc_hidden2=768, drop_p=0.5, embed_dim=512,
                             class_num=class_num, regress_num=regress_num,
                             is_regress=is_regress, use_depth=use_depth).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    print("网络导入成功")

    obs_pub = rospy.Publisher("obstacle_pose", PoseStamped, queue_size=10)
    tar_pub = rospy.Publisher("target_pose", PoseStamped, queue_size=10)
    rospy.Subscriber('real-time_image', Image, image_callback)
    rospy.Subscriber('video_stream', Bool, bool_callback)
    rospy.Subscriber('publish_sign', Bool, pub_sign_callback)
    while not rospy.is_shutdown():
        detect()
        if exit_sign:
            break
    rospy.spin()
