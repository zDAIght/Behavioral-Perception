import numpy as np
import torch
import time
import torch.nn.functional as F
from utils.function import image_convert_from_dir
from model.resnet_rnn import ClassifyRegressNet

data_path = 'PE_Dataset'
model_path = 'model_ckpt/2023-05-28-10:03:31/ClsRegNet_0.953-0.645.pth'
class_num = 10
regress_num = 28 * 3  # 轨迹回归数量

is_regress = True
use_depth = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ClassifyRegressNet(fc_hidden1=1024, fc_hidden2=768, drop_p=0.5, embed_dim=512,
                         class_num=class_num, regress_num=regress_num,
                         is_regress=is_regress, use_depth=use_depth).to(device)
net.load_state_dict(torch.load(model_path))
print("网络导入成功")

'''从数据集中导入图像进行预测'''
images28, cls_label, reg_label = image_convert_from_dir(data_path, "v_1_g01-3_c02", use_depth)

'''网络预测'''
net.eval()
start_time = time.time()
output_class, output_regress = net(images28)

'''处理预测结果'''
output_class = F.softmax(output_class, dim=1)
[score, cls_label_pred] = output_class.max(1, keepdim=True)
reg_label_pred = output_regress.cpu().data.squeeze().numpy()

reg_label = np.resize(reg_label, (28, 3))
reg_label_pred = np.round(np.resize(reg_label_pred, (28, 3)), decimals=4)

time = time.time() - start_time

print(">>>classification result:")
print("ground truth: {} | label pred: {} | label score: {:.3f}".format(cls_label, cls_label_pred.item(), score.item()))
print(">>>regression result:")
print("ground truth | label pred")
for i in range(len(reg_label)):
    print("{} | {}".format(reg_label[i].tolist(), reg_label_pred[i]))
print("检测用时 %.3fs" % time)
