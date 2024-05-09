import copy
import os
from tqdm import tqdm
import numpy as np
import torch
import json
import torch.nn.functional as F
from utils.function import image_convert_from_pecDir, pec_annotation
from model.resnet_rnn import ClassifyRegressNet
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


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
                # 遍历原始列表的每个元素并复制
                for element in file_list:
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


def compute_percentage(arr):
    total_elements = len(arr)
    unique_elements, counts = np.unique(arr, return_counts=True)
    percentages = (counts / total_elements) * 100
    result = dict(zip(unique_elements, percentages))
    return result


def compute_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def compute_pck(predictions, ground_truth, threshold):
    num_points = len(ground_truth)
    correct_count = 0

    for i in range(num_points):
        pred_point = predictions[i]
        true_point = ground_truth[i]

        if compute_distance(pred_point, true_point) <= threshold:
            correct_count += 1

    return correct_count


def run():
    pecle = LabelEncoder()
    pecle.fit(action_list)  # 将动作编码

    task2 = ["Approach", "TakeT-tube", "WaitingRobot", "TakePreservCyt", "TwistPreservCyt",
             "TakePipette", "UsePipette", "PlacePipette", "PlacePreservCyt", "Departure"]

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

    action2 = {}
    for i in task2:
        action2[i] = []

    cls_num, reg_num = 0, 0
    cls_test_list, cls_pred_list = [], []
    reg_error = []
    total_num = len(total_truth_list)
    '''从数据集中导入图像进行预测'''
    print("\n开始评估")
    for file in tqdm(total_truth_list):
        images28, label, poses = image_convert_from_pecDir(file, use_depth, is_regress)
        with torch.no_grad():
            if is_regress:
                output_class, output_regress = net(images28)
                output_class = net(images28)[0]
                output_class = F.softmax(output_class, dim=1)
                cls_label_pred = output_class.max(1, keepdim=True)[1]
                reg_label_pred = (output_regress.cpu().data.squeeze().numpy()).reshape(28, 3)

            else:
                output_class = net(images28)[0]
                output_class = F.softmax(output_class, dim=1)
                cls_label_pred = output_class.max(1, keepdim=True)[1]

        cls_test_list.append(pecle.inverse_transform([label]))
        cls_pred_list.append(pecle.inverse_transform([cls_label_pred.item()]))

        if cls_label_pred == label:
            cls_num += 1

        # reg_label_test = poses.reshape(28, 3)
        # reg_num += compute_pck(reg_label_pred, reg_label_test, th)
        # reg_error.append(np.mean(np.linalg.norm(reg_label_test - reg_label_pred, axis=1)))

        ac_label = pecle.inverse_transform([label])
        if "task2" in file[1][0][0]:
            action2[ac_label[0]].append(cls_label_pred.item())

    # 计算混淆矩阵
    cm = confusion_matrix(cls_test_list, cls_pred_list)
    report = classification_report(cls_test_list, cls_pred_list, output_dict=True)  # target_names=action_list,
    with open("pec_result.json", "w") as json_file:
        json.dump(report, json_file, indent=4)

    # 计算每个类别的查准率和查全率
    precision, recall, f1 = [], [], []
    for i in range(cm.shape[0]):
        true_positive = cm[i, i]
        false_positive = sum(cm[:, i]) - true_positive
        false_negative = sum(cm[i, :]) - true_positive

        precision_i = true_positive / (true_positive + false_positive)
        recall_i = true_positive / (true_positive + false_negative)
        f1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)

        precision.append(precision_i)
        recall.append(recall_i)
        f1.append(f1_i)

    # 计算总的查全率（宏平均）和查准率
    macro_recall = sum(recall) / len(recall)
    macro_precision = sum(precision) / len(precision)
    macro_f1 = sum(f1) / len(f1)

    # 计算总的回归误差
    total_reg_error = 0  # np.mean(reg_error)
    # 计算分类准确率
    cls_pred_acc = cls_num / total_num
    # 计算回归PCK
    reg_pred_PCK = reg_num / (total_num * 28)

    result = f"动作分类：\n查准率：{macro_precision} 查全率：{macro_recall} 准确率：{cls_pred_acc}\n\n回归预测：\n" \
             f"总体误差：{total_reg_error} 回归PCK：{reg_pred_PCK}\n\n混淆矩阵：\n标签：{cls_test_list}\n" \
             f"预测：{cls_pred_list}\n混淆矩阵：\n{cm}\n\n"

    for i in task2:
        arr = action2[i]
        res = i + " >>> "
        percentages = compute_percentage(arr)
        for num, percentage in percentages.items():
            res += f"{pecle.inverse_transform([num])[0]}:{percentage:.2f}% | "
        result += res + "\n"

    print(result)
    with open("pec_result.txt", 'w') as file:
        file.write(result)
    print('结果已写入txt文件')

    with open("pec_action_cls.json", 'w', encoding='utf-8') as f:
        json.dump(action, f, indent=4)
    print("混淆矩阵已保存")

    print("report>>>")
    # Extract precision, recall, and F1-score for each class
    precision_list = [report[cls]['precision'] for cls in action_list]
    recall_list = [report[cls]['recall'] for cls in action_list]
    f1_score_list = [report[cls]['f1-score'] for cls in action_list]

    # Extract Micro average, Macro average, and Weighted average
    micro_precision = report['micro avg']['precision']
    micro_recall = report['micro avg']['recall']
    micro_f1 = report['micro avg']['f1-score']

    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    weighted_precision = report['weighted avg']['precision']
    weighted_recall = report['weighted avg']['recall']
    weighted_f1 = report['weighted avg']['f1-score']

    # Output the results
    print("Precision for each class:", precision_list)
    print("Recall for each class:", recall_list)
    print("F1 score for each class:", f1_score_list)
    print("Micro Precision:", micro_precision)
    print("Micro Recall:", micro_recall)
    print("Micro F1:", micro_f1)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Weighted Precision:", weighted_precision)
    print("Weighted Recall:", weighted_recall)
    print("Weighted F1:", weighted_f1)


if __name__ == '__main__':
    # 设置路径
    data_file_path = "/home/magic/ZhangGZ/intention-aware-HRC/PEC_dataset_process/PEC_DataSet"
    action_label_file = "pec_action_label.txt"

    # 设置网络
    model_path = './pec_model_ckpt/2023-07-15-10:02:42/ClassifyNet_0.903.pth'

    with open(action_label_file, 'r') as f:
        action_list = []
        action_data = f.readlines()
        for action in action_data:
            action_list.append(action.replace('\n', ''))

    class_num = len(action_list)  # 分类总数
    regress_num = 28 * 3  # 轨迹回归数量
    th = 0.1  # PCK指标阈值，距离0.1m
    is_regress = False
    use_depth = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 导入网络
    print("导入网络 \n...")
    net = ClassifyRegressNet(fc_hidden1=1024, fc_hidden2=768, drop_p=0.5, embed_dim=512,
                             class_num=class_num, regress_num=regress_num,
                             is_regress=is_regress, use_depth=use_depth).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    print("网络导入成功")

    run()
