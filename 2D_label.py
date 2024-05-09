import glob
import json
import os
import sys
from copy import deepcopy
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import logging

# Callback function for clicking on OpenCV window
pix_pose, point, camera_pose = [], [], None
window_name = 'base'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# camera_color_img, camera_depth_img = robot.get_camera_data()
def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pix_pose, camera_pose
        pix_pose = [x, y]

        # Get click point in camera coordinates
        click_z = camera_depth_img[y][x] * cam_depth_scale
        click_x = np.multiply(x - cam_intrinsics[0][2], click_z / cam_intrinsics[0][0])
        click_y = np.multiply(y - cam_intrinsics[1][2], click_z / cam_intrinsics[1][1])
        if click_z == 0:
            return
        click_point = np.asarray([click_x, click_y, click_z])
        click_point.shape = (3, 1)

        camera_pose = deepcopy(click_point)
        camera_pose = camera_pose.squeeze()
        # print(camera_pose)
        point.append(camera_pose)

        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow(window_name, img)

        # 相机坐标系转世界坐标系
        # camera2robot = cam_pose
        # target_position = np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:]
        #
        # target_position = target_position[0:3, 0]
        # print('world_pose: ')
        # print(target_position)


if __name__ == '__main__':
    with open('D455_intrinsics.json', 'r') as f:
        intr = json.load(f)
    intr_width = intr['width']
    intr_height = intr['height']
    intr_fx = intr['fx']
    intr_fy = intr['fy']
    intr_ppx = intr['ppx']
    intr_ppy = intr['ppy']

    cam_intrinsics = np.array([intr_fx, 0., intr_ppx, 0., intr_fy, intr_ppy, 0, 0, 1]).reshape(3, 3)
    cam_pose = np.loadtxt('cam_pose/camera_pose.txt', delimiter=' ')
    cam_depth_scale = np.loadtxt('cam_pose/camera_depth_scale.txt', delimiter=' ')

    data_path = "PE_Dataset"
    dir_list = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    dir_list.sort()

    window_name = 'camera pose labeling...'
    cv2.namedWindow(window_name)

    '''进入循环'''
    for dir_path in dir_list:
        txt_file = os.path.join(dir_path, 'label.txt')

        if os.path.exists(txt_file):
            continue
        else:
            print("开始标注文件夹：" + os.path.basename(dir_path))
            color_dir_path = os.path.join(dir_path, "color")
            depth_dir_path = os.path.join(dir_path, "depth")

            color_list = [os.path.join(color_dir_path, f) for f in os.listdir(color_dir_path)]
            color_list.sort()
            depth_list = [os.path.join(depth_dir_path, f) for f in os.listdir(depth_dir_path)]
            depth_list.sort()

            point_num = 0
            points = []

            while point_num < len(color_list):
                cv2.setMouseCallback(window_name, mouseclick_callback)

                camera_color_img = np.array(Image.open(color_list[point_num]))
                camera_depth_img = np.array(Image.open(depth_list[point_num]))

                bgr_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)

                # Show color and depth frames
                cv2.imshow(window_name, bgr_data)

                img = deepcopy(bgr_data)
                while True:
                    key = cv2.waitKey(20) & 0xFF
                    if key == ord('q'):
                        sys.exit()
                    elif key == ord('s'):  # 按下'q'键退出
                        break
                    elif key == ord('c'):  # 按下'c'键清除标记点
                        img = deepcopy(bgr_data)
                        point = []
                        cv2.imshow(window_name, img)

                if len(point) == 1:
                    print("已标注完第{}张图片，坐标点为：{}".format(point_num+1, camera_pose))
                    points.append(camera_pose)
                    point = []
                    point_num += 1
                else:
                    point = []
                    logger.warning("标注不正确，重新标注")

            if len(points) == 28:
                with open(txt_file, 'w') as txt:
                    for elem in points:
                        txt.write("{:.4f}".format(elem[0]*1000) + ',' +
                                  "{:.4f}".format(elem[1]*1000) + ',' +
                                  "{:.4f}".format(elem[2]*1000) + '\n')
            else:
                assert ValueError("标注点数量不够")

    cv2.destroyAllWindows()

    # if len(click_point_pix) != 0:
    #     bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0, 0, 255), 2)
    #
    # cv2.imshow('color', bgr_data)
    # # cv2.imshow('depth', camera_depth_img)
    # cv2.waitKey(0)
