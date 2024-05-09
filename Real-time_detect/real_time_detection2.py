import os
import sys
import time

import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

import cv2

target_path = "../"
sys.path.append(target_path)
from model.resnet_rnn import ClassifyRegressNet
from utils.function import image_convert_from_img, action_target
from utils.realsenseD455 import Camera
from utils.gpr import GPR
from yolo_pose.detect import YoloPoseDetect

# 指定字体、颜色和大小等参数
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)  # 白色文字
size = 1  # 字体大小
thickness = 2  # 文字线宽
add_sign, exit_sign = True, False
# 信号
pub_sign = False
pose_sign = True
# 机械臂初始位置
init_pose = [-0.270, -0.00, 0.500]  # 初始位置
cam_intrinsics = np.array([380.15350342, 0., 316.6630249, 0., 379.74990845, 239.63673401, 0, 0, 1]).reshape(3, 3)
cam_pose = np.loadtxt('../cam_pose/camera_pose.txt', delimiter=' ')
cam_depth_scale = np.loadtxt('../cam_pose/camera_depth_scale.txt', delimiter=' ')


def pub_sign_callback(data):
    global pub_sign
    pub_sign = True


#
#
# def bool_callback(data):
#     global exit_sign
#     # sign = data.data
#     exit_sign = True
#     rospy.signal_shutdown("Exiting ...")
#
#
# def image_callback(data):
#     global bridge
#     cv_img = bridge.imgmsg_to_cv2(data, "rgba8")
#     image = [cv_img[:, :, :3], cv_img[:, :, 3]]
#     if add_sign:
#         images.append(image)


def pixel2world(depth_image, x, y):
    camera_z = depth_image[y][x] * cam_depth_scale
    camera_x = np.multiply(x - cam_intrinsics[0][2], camera_z / cam_intrinsics[0][0])
    camera_y = np.multiply(y - cam_intrinsics[1][2], camera_z / cam_intrinsics[1][1])
    if camera_z == 0:
        assert ValueError('z-pose error')
    camera_point = np.asarray([camera_x, camera_y, camera_z])
    camera_point.shape = (3, 1)
    camera2robot = cam_pose
    world_position = np.dot(camera2robot[0:3, 0:3], camera_point) + camera2robot[0:3, 3:]
    world_position = world_position[0:3, 0]

    return world_position


def detect():
    global pub_sign, pose_sign, add_sign

    cam = Camera()
    cv2.namedWindow('Real-time camera', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow('Real-time camera', 640, 480)

    images = []
    key_points_images, arm_pixel_poses = [], []

    save = False

    while True:
        color_img, depth_img = cam.get_data()
        #
        # if add_sign:
        #     images.append([color_img, depth_img])
        #     add_sign = False
        # else:
        #     add_sign = True

        key = cv2.waitKey(1) & 0xFF

        # add_sign = False

        # image = np.zeros((480, 640, 3), dtype=np.uint8)
        # # img = copy.deepcopy(image)
        # # 在图片中心位置写入文本
        # text = 'Waiting for the video stream ...'
        # (text_width, text_height) = cv2.getTextSize(text, font, fontScale=size, thickness=thickness)[0]
        # x = (image.shape[1] - text_width) // 2
        # y = (image.shape[0] + text_height) // 2
        # cv2.putText(image, text, (x, y), font, size, color, thickness)
        # cv2.imshow('Real-time camera', image)

        '''识别图像'''
        # for img in images:
        key_points, key_rgb = pose_detect.detect(color_img)

        cv2.imshow('Real-time camera', key_rgb)
        cv2.waitKey(5)

        if key & 0xFF == ord('s'):
            save = True
            print("开始")
            print("开始")
            print("开始")
        if save:
            key_points_images.append([key_rgb, depth_img])
            arm_pixel_poses.append([key_points['right_wrist'], key_points['right_elbow']])
        # print("\n手臂关键点：{}\n{}".format(len(arm_pixel_poses), arm_pixel_poses))
        # image28 = image_convert_from_img(key_points_images, use_depth)

        # with torch.no_grad():
        #     if is_regress:
        #         output_class, output_regress = net(image28)
        #     else:
        #         output_class = net(image28)[0]
        #     output_class = F.softmax(output_class, dim=1)
        #     [score, cls_label_pred] = output_class.max(1, keepdim=True)
        #     action = pecle.inverse_transform([int(cls_label_pred.item())])[0]
        #     score = np.array(score.item())
        #     print("预测动作：{}, 置信度：{}".format(action, score))
        #
        #     if is_regress:
        #         reg_label_pred = output_regress.cpu().data.squeeze().numpy()
        #         reg_label_pred = np.round(np.resize(reg_label_pred, (28, 3)), decimals=4)
        #         print("预测轨迹为: ")
        #         print(reg_label_pred)

        # '''显示序列图像和预测结果'''
        # pred_action = 'Action prediction : {}'.format(action)
        # pred_score = 'Action Score : {:.3f}'.format(score)  # - np.random.uniform(0, 0.1)
        # for key_img in key_points_images:
        #     disp_image = copy.deepcopy(key_img[0])
        #     cv2.putText(disp_image, pred_action, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        #     cv2.putText(disp_image, pred_score, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        #     cv2.imshow('Real-time camera', disp_image)
        #     cv2.waitKey(60)
        #
        # '''根据预测动作确定机械臂目标点坐标'''
        # if action in action_target.keys():
        #     # target_pose = action_target[action]
        #     target_pose = action_target["TakeT-tube"]

        '''将像素坐标换算至机械臂坐标系'''
        wrist_world_poses, elbow_world_poses = [], []
        if key & 0xFF == ord('q') or key == 27:

            if len(key_points_images) == len(arm_pixel_poses):
                for index in range(len(key_points_images)):
                    camera_depth_img = key_points_images[index][1] * 1000.
                    depth_image = np.expand_dims(camera_depth_img, axis=2)

                    arm_pixel_pose = arm_pixel_poses[index]
                    wrist_x, wrist_y = int(arm_pixel_pose[0][0]), int(arm_pixel_pose[0][1])
                    elbow_x, elbow_y = int(arm_pixel_pose[1][0]), int(arm_pixel_pose[1][1])

                    wrist_world_pose = pixel2world(depth_image, wrist_x, wrist_y)
                    elbow_world_pose = pixel2world(depth_image, elbow_x, elbow_y)

                    wrist_world_poses.append(wrist_world_pose)
                    elbow_world_poses.append(elbow_world_pose)

                    cv2.imwrite('RGB/{:06d}.png'.format(index), key_points_images[index][0])

                for index in range(len(key_points_images)):
                    with open('Keypoints/wrist-output.txt', 'w') as file:
                        for wrist in wrist_world_poses:
                            for item in wrist:
                                file.write(str(item) + ',')
                            file.write('\n')

                    with open('Keypoints/elbow-output.txt', 'w') as file:
                        for elbow in elbow_world_poses:
                            for item in elbow:
                                file.write(str(item) + ',')
                            file.write('\n')



            else:
                break
            cv2.destroyAllWindows()
            break

            #
            # print("手腕\n", list(wrist_world_poses), "\n")
            # print("手肘\n", list(elbow_world_poses), "\n")

            # '''选择模型预测轨迹位置'''
            # gpr = GPR()
            # wrist_pred_pose = gpr.predict(wrist_world_poses)[-1]
            # elbow_pred_pose = gpr.predict(elbow_world_poses)[-1]
            #
            # # wrist_pred_pose = wrist_world_poses[-1]
            # # elbow_pred_pose = elbow_world_poses[-1]
            #
            # print('wrist_world_pose: ', wrist_pred_pose)
            # print('elbow_world_pose: ', elbow_pred_pose)
            #
            # '''启用Moveit控制'''
            # if pub_sign:
            #     '''发布手腕障碍物预测位置'''
            #     # obstacle_pose = PoseStamped()
            #     # obstacle_pose.header.stamp = rospy.Time.now()
            #     # obstacle_pose.pose.position.x = world_position[0]
            #     # obstacle_pose.pose.position.y = world_position[1]
            #     # obstacle_pose.pose.position.z = world_position[2]
            #     # obstacle_pose.pose.orientation.x = 0.0
            #     # obstacle_pose.pose.orientation.y = 0.0
            #     # obstacle_pose.pose.orientation.z = 0.0
            #     # obstacle_pose.pose.orientation.w = 1.0
            #     obs_pose = [wrist_pred_pose[0],
            #                 wrist_pred_pose[1],
            #                 wrist_pred_pose[2],
            #                 elbow_pred_pose[0],
            #                 elbow_pred_pose[1],
            #                 elbow_pred_pose[2]]
            #     obstacle_pose = Float32MultiArray(data=obs_pose)
            #     obs_pub.publish(obstacle_pose)
            #
            #     time.sleep(0.5)
            #     # [9.23879533e-01 - 3.82683432e-01 - 2.34326020e-17  5.65713056e-17]
            #     target_position = PoseStamped()
            #     target_position.header.stamp = rospy.Time.now()
            #     target_position.pose.orientation.x = 9.23879533e-01
            #     target_position.pose.orientation.y = -3.82683432e-01
            #     target_position.pose.orientation.z = -2.34326020e-17
            #     target_position.pose.orientation.w = 5.65713056e-17
            #
            #     if pose_sign:
            #         '''机械臂根据预测动作执行相应的动作'''
            #         target_position.pose.position.x = target_pose[0]
            #         target_position.pose.position.y = target_pose[1]
            #         target_position.pose.position.z = target_pose[2]
            #         tar_pub.publish(target_position)
            #
            #         pose_sign = False
            #
            #     else:
            #         '''机械臂回到初始位置'''
            #         target_position.pose.position.x = init_pose[0]
            #         target_position.pose.position.y = init_pose[1]
            #         target_position.pose.position.z = init_pose[2]
            #         tar_pub.publish(target_position)
            #
            #         pose_sign = True
            #
            #     pub_sign = False


        # add_sign = True


if __name__ == '__main__':

    # 设置网络
    model_path = '../pec_model_ckpt/2023-08-13-15:57:30/ClassifyNet_0.883.pth'

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

    detect()
