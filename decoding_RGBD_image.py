"""
    读取视频video和深度图文件h5py
    从中抽离彩色图和深度图
"""
import json
import os
import shutil
import time

from tqdm import tqdm
from utils.tool import del_file


def remodel(old_path, file_list, image_type, act, sub_num, num):
    new_path = os.path.join(old_path, "a_{}_s{}-{}".format(act, sub_num, num))

    if image_type != "json":
        for i, images in enumerate(file_list):
            image_new_path = os.path.join(new_path+"_c{:02d}".format(i), image_type)

            # 判断保存文件夹是否存在，不存在创建，存在清空
            if not os.path.exists(image_new_path):
                os.makedirs(image_new_path)
            else:
                del_file(image_new_path)

            for image in images:
                shutil.copy(image, image_new_path)
                time.sleep(0.01)
    else:
        for i, files in enumerate(file_list):
            new_label_path = os.path.join(new_path + "_c{:02d}".format(i), "label.txt")
            with open(new_label_path, 'w') as f:
                for point in files:
                    pose = point["camera pose"]
                    for j in range(len(pose)):
                        if j == len(pose) - 1:
                            f.write(str(pose[j]) + "\n")
                        else:
                            f.write(str(pose[j]) + " ")


def window_slide_images(image_list, window_len, slide_step):
    window_start = 0
    window_list = [0] * window_len
    images = []
    while window_start < len(image_list) - len(window_list) + 1:
        window_end = window_start + len(window_list)
        window_list = image_list[window_start:window_end]
        images.append(window_list)
        # 将滑动窗口向右移动 n 个元素
        window_start += slide_step

    # 最后一次滑动覆盖最后一个元素
    last_window = image_list[-len(window_list):]
    images.append(last_window)

    return images


if __name__ == '__main__':
    rgbd_path = "/home/magic/ZhangGZ/intention-aware-HRC/PE_dataset_process/images_RGBD/task01"
    save_path = "PE_Dataset"
    video_list = [os.path.join(rgbd_path, i) for i in os.listdir(rgbd_path)]
    video_list.sort()

    image_list = [os.path.join(save_path, i) for i in os.listdir(save_path)]
    for images in image_list:
        im_list = [os.path.join(images+"/color", i) for i in os.listdir(images+"/color")]
        is_list = [os.path.join(images+"/depth", i) for i in os.listdir(images+"/depth")]

        if len(im_list) != 28 or len(is_list) != 28:
            print(images)
        print("d")

    # 定义滑动窗口
    frames = 28
    step = 20
    frame_num = None

    for s, subject in enumerate(video_list):
        trajectory_list = [os.path.join(subject, i) for i in os.listdir(subject)]
        trajectory_list.sort()

        print("\n开始转移第{}位受试者的数据".format(s+1))
        for trajectory in tqdm(trajectory_list):
            color_path = os.path.join(trajectory, "color")
            depth_path = os.path.join(trajectory, "depth")

            json_file = os.path.join(trajectory, "label.json")
            with open(json_file, 'r') as file:
                label_list = json.load(file)["human_keypoints"]

            color_list = [os.path.join(color_path, i) for i in os.listdir(color_path)]
            color_list.sort()
            depth_list = [os.path.join(depth_path, i) for i in os.listdir(depth_path)]
            depth_list.sort()

            if len(color_list) != len(depth_list):
                assert ValueError("数量不等")

            color_slide_list = window_slide_images(color_list, frames, step)
            depth_slide_list = window_slide_images(depth_list, frames, step)
            label_slide_list = window_slide_images(label_list, frames, step)

            action, count = trajectory[-3:-2], trajectory[-1:]

            subject_num = subject[-2:]

            # 将图片移动到新的文件中
            remodel(save_path, color_slide_list, "color", action, subject_num, count)
            remodel(save_path, depth_slide_list, "depth", action, subject_num, count)
            remodel(save_path, label_slide_list, "json", action, subject_num, count)
            print()

    # with open("PE_data/v_1_g02_c02/label.txt", 'r') as file:
    #     label = np.array([])
    #     for line in file:
    #         data_line = line.strip("\n").split()
    #         data_line = [np.float(x) for x in data_line]
    #         label = np.append(label, data_line)
    # print(list(label))

    # color_path = "PE_data/v_1_g01_c01/color/000.png"
    # depth_path = "PE_data/v_1_g01_c01/depth/000.tiff"
    #
    # pic = Image.open(color_path, 'r')
    # rgb = np.array(pic)
    # pic = Image.open(depth_path, 'r')
    # depth = np.expand_dims(np.array(pic), axis=2)
    #
    # x = np.concatenate((depth, rgb), axis=2)
    # print(x.shape)
    # print()

    # 按照动作重命名文件夹
    # action_1, action_2, action_3, action_4, action_5 = 0, 0, 0, 0, 0
    # action_num = 0
    # action_name = "0"
    #
    # for traj in traj_list:
    #
    #     index = int(traj.split('/')[-1][-1])
    #     traj_index = traj.split('/')[-1][-2:]
    #
    #     if index == 1 or index == 2:
    #         action_name = "1"
    #         action_1 += 1
    #         action_num = action_1
    #     if index == 3 or index == 4:
    #         action_name = "2"
    #         action_2 += 1
    #         action_num = action_2
    #     if index == 5 or index == 6:
    #         action_name = "3"
    #         action_3 += 1
    #         action_num = action_3
    #     if index == 7 or index == 8:
    #         action_name = "4"
    #         action_4 += 1
    #         action_num = action_4
    #     if index == 9 or index == 0:
    #         action_name = "5"
    #         action_5 += 1
    #         action_num = action_5
    #
    #     dir_new_name = os.path.join(rgbd_path, "v_{}_g{:02d}".format(action_name, action_num))
    #     os.rename(traj, dir_new_name)
    #
    #     print(traj, "已重命名")
