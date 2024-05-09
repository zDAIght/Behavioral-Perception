import copy
import json
import os
import time
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from utils.function import train, validation, DatasetProcess, pec_annotation, PECDatasetProcess
from model.resnet_rnn import ClassifyRegressNet
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import LabelEncoder
from yolo_pose.detect import YoloPoseDetect


def window_slide(file_list, window_len, slide_step):
    if len(file_list) < window_len:
        file = copy.deepcopy(file_list)
        diff = window_len - len(file_list)
        if diff < len(file_list):
            for i in range(diff):
                file_list.insert(len(file) - diff + 2 * i, file[len(file) - diff + i])
        else:
            # 循环直到达到目标长度
            copied_list = []
            while len(copied_list) < window_len:
                # print(file_list)
                # print(file)
                # print(len(copied_list))
                # 遍历原始列表的每个元素并复制
                for element in file:
                    copied_list.append(element)
                    if len(copied_list) >= window_len:
                        break
            copied_list.sort()
            file_list = copied_list

    window_start = 0
    window_list = [0] * window_len
    images = []
    while window_start < len(file_list) - len(window_list) + 1:
        window_end = window_start + len(window_list)
        window_list = file_list[window_start:window_end]
        images.append(window_list)
        # 将滑动窗口向右移动 n 个元素
        window_start += slide_step

    # 最后一次滑动覆盖最后一个元素
    if images[-1][-1] != file_list[-1]:
        last_window = file_list[-len(window_list):]
        images.append(last_window)

    return images


def run():
    pecle = LabelEncoder()
    pecle.fit(action_list)  # 将动作编码

    tasks = [os.path.join(data_file_path, t) for t in os.listdir(data_file_path)]
    tasks.sort()

    total_truth_list = []
    print("读取数据集...")
    for task in tqdm(tasks):
        subjects = [os.path.join(task, s) for s in os.listdir(task)]
        subjects.sort()
        for subject in subjects:
            takes = [os.path.join(subject, k) for k in os.listdir(subject)]
            takes.sort()
            for take in takes:
                action_file = os.path.join(take, "action_sequence.json")
                with open(action_file, 'r') as file:
                    action_json = json.load(file)
                action_sequence = action_json["Action Sequence"]
                action_slide = window_slide(action_sequence, 3, 2)

                for ac_list in action_slide:
                    pec_action = [pec_annotation[ac_list[1]]]
                    pec_action = pecle.transform(pec_action)
                    color_list = [os.path.join(take, "color", "{:04d}.png".format(index))
                                  for index in range(ac_list[0], ac_list[2])]
                    # print(color_list)
                    # print(ac_list)
                    color_slide_list = window_slide(color_list, 28, 10)
                    for color_image28 in color_slide_list:
                        image_list, motion_list, center_point = [], [], None
                        for color_image in color_image28:
                            image_index = os.path.basename(color_image)
                            depth_image = os.path.join(take, "depth", "{}.tiff".format(image_index.split(".")[0]))

                            # json_path = os.path.join(motion_file_path, current_path,
                            #                          "3d_objects",
                            #                          "frame_{}.json".format(image_index.rsplit(".", 1)[0]))
                            # with open(json_path, 'r') as file:
                            #     json_file = json.load(file)
                            # for obj in json_file:
                            #     if obj["class_name"] == "RightHand":
                            #         box = obj["bounding_box"]
                            #         x0, y0, z0 = box["x0"], box["y0"], box["z0"]
                            #         x1, y1, z1 = box["x1"], box["y1"], box["z1"]
                            #         center_point = [(x1 + x0) / 2, (y1 + y0) / 2, (z1 + z0) / 2]
                            # motion_list.append(center_point)
                            image_list.append([color_image, depth_image])

                        '''所有的动作,轨迹,rgb图和深度图以读入'''
                        total_truth_list.append([pec_action.item(), image_list])

    # 创建tensorboard
    writer = SummaryWriter(log_dir='runs/accuracy_' + hms_time)

    all_X_list, all_y_list = [], []
    # 获取所有训练数据
    print("划分数据集...")
    for ground_truth in total_truth_list:
        all_X_list.append(ground_truth[1])
        all_y_list.append(ground_truth[0])

    # 划分数据集
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.2,
                                                                      random_state=42)

    # 将图片处理拼接后导入程序
    train_set = PECDatasetProcess(train_list, train_label, [res_size, res_size], pose_detect,
                                  use_depth=use_depth, is_regress=is_regress)
    valid_set = PECDatasetProcess(test_list, test_label, [res_size, res_size], pose_detect,
                                  use_depth=use_depth, is_regress=is_regress)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # 生成网络
    print("导入网络...\n")
    net = ClassifyRegressNet(fc_hidden1=1024, fc_hidden2=768, drop_p=0.5, embed_dim=512,
                             class_num=class_num, regress_num=regress_num,
                             is_regress=is_regress, use_depth=use_depth).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=init_learning_rate)
    lr_scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    # 记录训练数据
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    data_save_path = os.path.join(save_data_path, hms_time)
    # if not os.path.exists(data_save_path):
    #     os.makedirs(data_save_path)
    if not os.path.exists(os.path.join(save_model_path, hms_time)):
        os.makedirs(os.path.join(save_model_path, hms_time))

    train_loss, train_score, test_loss, test_score = None, None, None, None

    # 开始训练
    print("开始训练...\n")
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, net, device, train_loader, optimizer, epoch,
                                           [w1, w2], is_regress=is_regress)
        epoch_test_loss, epoch_test_score = validation(net, device, lr_scheduler, valid_loader, save_model_path,
                                                       epoch, [w1, w2], is_regress=is_regress)
        if is_regress:
            # 保存准确率前三的模型
            if sum(epoch_test_score) > sum(min(top_kr)):
                idx = top_kr.index(min(top_kr))
                top_kr[idx] = epoch_test_score
                models_params[idx] = net.state_dict()

            writer.add_scalars(main_tag="loss",
                               tag_scalar_dict={"train_loss": np.mean(train_losses),
                                                "val_loss": np.mean(epoch_test_loss)},
                               global_step=epoch)
            writer.add_scalars(main_tag="Classification Accuracy",
                               tag_scalar_dict={"train_cls_acc": np.round(np.mean(train_scores[0]) * 100, 2),
                                                "val_cls_acc": np.round(np.mean(epoch_test_score[0]) * 100, 2)},
                               global_step=epoch)
            writer.add_scalars(main_tag="Regression Accuracy",
                               tag_scalar_dict={"train_reg_acc": np.round(np.mean(train_scores[1]) * 100, 2),
                                                "val_reg_acc": np.round(np.mean(epoch_test_score[1]) * 100, 2)},
                               global_step=epoch)

            if (epoch + 1) % 5 == 0:
                torch.save(net.state_dict(), os.path.join(os.path.join(save_model_path, hms_time),
                                                          'ClsRegNet_{:.3f}-{:.3f}.pth'.format(
                                                              epoch_test_score[0], epoch_test_score[1])))
                print("在第{}个epoch保存模型".format(epoch + 1))

        else:
            # 保存准确率前三的模型
            if epoch_test_score > min(top_kc):
                idx = top_kc.index(min(top_kc))
                top_kc[idx] = epoch_test_score
                models_params[idx] = net.state_dict()

            # save results
            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)

            # save all train test results
            train_loss = np.array(epoch_train_losses)
            train_score = np.array(epoch_train_scores)
            test_loss = np.array(epoch_test_losses)
            test_score = np.array(epoch_test_scores)
            # np.savetxt(os.path.join(data_save_path, 'epoch{:03d}_training_losses.txt'.format(epoch + 1)),
            #            train_loss, delimiter=' ')
            # np.savetxt(os.path.join(data_save_path, 'epoch{:03d}_training_scores.txt'.format(epoch + 1)),
            #            train_score, delimiter=' ')
            # np.savetxt(os.path.join(data_save_path, 'epoch{:03d}_test_loss.txt'.format(epoch + 1)),
            #            test_loss, delimiter=' ')
            # np.savetxt(os.path.join(data_save_path, 'epoch{:03d}_test_score.txt'.format(epoch + 1)),
            #            test_score, delimiter=' ')
            writer.add_scalar(tag="train losses",
                              scalar_value=np.mean(train_losses),
                              global_step=epoch)
            writer.add_scalar(tag="train scores",
                              scalar_value=np.mean(train_scores),
                              global_step=epoch)
            writer.add_scalar(tag="epoch test losses",
                              scalar_value=np.mean(epoch_test_losses),
                              global_step=epoch)
            writer.add_scalar(tag="epoch test scores",
                              scalar_value=np.mean(epoch_test_scores),
                              global_step=epoch)

            if (epoch + 1) % 3 == 0:
                torch.save(net.state_dict(), os.path.join(os.path.join(save_model_path, hms_time),
                                                          'ClassifyNet_{:.3f}.pth'.format(epoch_test_score)))
                print("在第{}个epoch保存模型".format(epoch+1))

    if is_regress:
        # 保存模型
        for i, model in enumerate(models_params):
            if model is not None:
                torch.save(model, os.path.join(os.path.join(save_model_path, hms_time),
                                               'ClsRegNet_{:.3f}-{:.3f}.pth'.format(top_kr[i][0], top_kr[i][1])))
            print("已保存模型：ClsRegNet_{:.3f}-{:.3f}.pth".format(top_kr[i][0], top_kr[i][1]))
    else:
        # 保存模型
        for i, model in enumerate(models_params):
            if model is not None:
                torch.save(model, os.path.join(os.path.join(save_model_path, hms_time),
                                               'ClassifyNet_{:.3f}.pth'.format(top_kc[i])))
            print("已保存模型：ClassifyNet_{:.3f}.pth".format(top_kc[i]))


if __name__ == '__main__':
    # 设置路径
    data_file_path = "/home/magic/ZhangGZ/intention-aware-HRC/PEC_dataset_process/PEC_DataSet"
    action_label_file = "pec_action_label.txt"

    save_model_path = "pec_model_ckpt"  # 模型保存路径
    save_data_path = "train_result"  # 训练数据保存路径
    hms_time = time.strftime("%Y-%m-%d-%H:%M:%S")

    with open(action_label_file, 'r') as f:
        action_list = []
        action_data = f.readlines()
        for action in action_data:
            action_list.append(action.replace('\n', ''))

    # 训练参数
    res_size = 224  # ResNet 图像尺寸
    class_num = len(action_list)  # 分类数量
    regress_num = 28 * 3  # 轨迹回归数量
    epochs = 50
    batch_size = 15
    log_interval = 15  # 打印训练信息的间隔

    # 设置学习率
    init_learning_rate = 1e-3  # 初始值
    step_size = int(epochs / 2)  # 设置多少个epoch后调整一次
    gamma = 0.1  # 衰减系数

    w1, w2 = 1, 20  # 分类回归网络的loss比重
    use_depth = True
    is_regress = False

    # 检查GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # yolo-pose参数设置
    dev = '0'
    wgt = 'yolo_pose/last.pt'
    pose_detect = YoloPoseDetect(dev, wgt)
    print("Yolo-Pose模型以导入")

    # 记录准确率前三和对应模型参数的列表
    top_kr = [[0., 0.]] * 3
    top_kc = [0.0] * 3
    models_params = [None] * 3

    # 训练参数
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0,
              'pin_memory': True, 'drop_last': True} if use_cuda else {}

    run()
