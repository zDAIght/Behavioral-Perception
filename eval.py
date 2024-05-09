import os
from tqdm import tqdm
import numpy as np
import torch
import json
import torch.nn.functional as F
from utils.function import image_convert_from_dir
from model.resnet_rnn import ClassifyRegressNet
from sklearn.metrics import confusion_matrix


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
    action = {}
    for i in range(10):
        action[str(i)] = []

    cls_num, reg_num = 0, 0
    cls_test_list, cls_pred_list = [], []
    reg_error = []
    total_num = len(file_list)
    '''从数据集中导入图像进行预测'''
    print("\n开始评估")
    for file in tqdm(file_list):
        images28, label, poses = image_convert_from_dir(data_path, file, use_depth)
        with torch.no_grad():
            output_class, output_regress = net(images28)
            output_class = F.softmax(output_class, dim=1)
            cls_label_pred = output_class.max(1, keepdim=True)[1]
            reg_label_pred = (output_regress.cpu().data.squeeze().numpy()).reshape(28, 3)

        cls_test_list.append(label)
        cls_pred_list.append(cls_label_pred.item())

        if cls_label_pred == label:
            cls_num += 1

        reg_label_test = poses.reshape(28, 3)
        reg_num += compute_pck(reg_label_pred, reg_label_test, th)
        reg_error.append(np.mean(np.linalg.norm(reg_label_test - reg_label_pred, axis=1)))

        action[str(label)].append(cls_label_pred.item())

    # 计算混淆矩阵
    cm = confusion_matrix(cls_test_list, cls_pred_list)
    # 计算每个类别的查准率和查全率
    precision, recall = [], []
    for i in range(cm.shape[0]):
        true_positive = cm[i, i]
        false_positive = sum(cm[:, i]) - true_positive
        false_negative = sum(cm[i, :]) - true_positive

        precision.append(true_positive / (true_positive + false_positive))
        recall.append(true_positive / (true_positive + false_negative))
    # 计算总的查全率（宏平均）和查准率
    macro_recall = sum(recall) / len(recall)
    macro_precision = sum(precision) / len(precision)

    # 计算总的回归误差
    total_reg_error = np.mean(reg_error)
    # 计算分类准确率
    cls_pred_acc = cls_num / total_num
    # 计算回归PCK
    reg_pred_PCK = reg_num / (total_num * 28)

    result = f"动作分类：\n查准率：{macro_precision} 查全率：{macro_recall} 准确率：{cls_pred_acc}\n\n回归预测：\n" \
             f"总体误差：{total_reg_error} 回归PCK：{reg_pred_PCK}\n\n混淆矩阵：\n标签：{cls_test_list}\n" \
             f"预测：{cls_pred_list}\n混淆矩阵：\n{cm}\n\n"

    for i in range(10):
        arr = action[str(i)]
        res = "action " + str(i) + ">>> "
        percentages = compute_percentage(arr)
        for num, percentage in percentages.items():
            res += f"{num}:{percentage:.2f}% | "
        result += res + "\n"

    print(result)
    with open("result.txt", 'w') as file:
        file.write(result)
    print('结果已写入txt文件')

    with open("action_cls.json", 'w', encoding='utf-8') as f:
        json.dump(action, f, indent=4)
    print("混淆矩阵已保存")


if __name__ == '__main__':
    data_path = 'PE_Dataset'
    file_list = [files for files in os.listdir(data_path)]
    file_list.sort()

    # 设置网络
    model_path = './model_ckpt/2023-06-21-15:13:07/ClsRegNet_0.949-0.772.pth'
    class_num = 10  # 分类总数
    regress_num = 28 * 3  # 轨迹回归数量
    th = 1  # PCK指标阈值，距离1mm
    is_regress = True
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
