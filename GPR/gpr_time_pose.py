import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义高斯过程回归模型
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e+5))  # 定义径向基函数协方差函数
gpr = GaussianProcessRegressor(kernel=kernel,
                               alpha=0.1,
                               n_restarts_optimizer=5)

data_path = "/home/magic/ZhangGZ/intention-aware-HRC/PE_dataset_process/images_RGBD/task01/subject03"

traj_list = [os.path.join(data_path, f) for f in os.listdir(data_path)]
traj_list.sort()

keypoints = []

for traj in traj_list:
    with open(os.path.join(traj, "label.json"), 'r') as f:
        keypoint = json.load(f)["human_keypoints"]
        keypoints.append(keypoint)

# for i in range(72, 73):
#     differences, poses = [], []
#     for keypoint in keypoints[i]:
#         poses.append(keypoint["camera pose"])
#         # poses_x.append(keypoint["camera pose"][0])
#         # poses_y.append(keypoint["camera pose"][1])
#         # poses_z.append(keypoint["camera pose"][2])
#
#     pose = np.array(poses[:61])*100
#     k = len(pose)
#     x = np.linspace(1, k, k)
#
#     # X = (np.array(pose[:-1])*100).reshape(k-1, 3)
#     X = x.reshape(-1, 1)
#     Y = pose
#
#     # gpr_fit(X, Y)
#     # 训练模型
#     gpr.fit(X, Y)

# 进行预测 假设有待预测点的输入特征存储在prediction_points中，格式为 [(x_pred1, y_pred1, z_pred1), (x_pred2, y_pred2, z_pred2), ...,
# (x_predm, y_predm, z_predm)]
test_poses = []
for keypoint in keypoints[13]:
    test_poses.append(keypoint["camera pose"])

test_pose = np.array(test_poses[:64]) * 100000  # [:61]
total_num = len(test_pose)
y_test = test_pose.reshape(total_num, 3)
x_test = np.linspace(1, total_num, total_num).reshape(-1, 1)
time = round((total_num / 17), 2)
inter_time = 1

inter_num = int(total_num - inter_time * 17)
inter_num = 28
X = x_test[:inter_num]  # 观测值
Y = y_test[:inter_num]  # 预测值

gpr.fit(X, Y)

all_y_test, all_y_pred, all_y_diff = [], [], []
X_test = np.linspace(inter_num + 1, total_num, total_num - inter_num).reshape(-1, 1)

# for i in range(inter_num, total_num):
Y_pred, sigma = gpr.predict(X_test, return_std=True)  # 预测输出和方差
# X_test = np.insert(X_test, len(X_test), i+1, axis=0)
# all_y_pred.append(Y_pred[-1])
# all_y_test.append(y_test[i])

# all_y_err = abs(np.array(all_y_test) - np.array(all_y_pred))
y_truth = y_test[inter_num:total_num]
all_y_err = abs(Y_pred - y_truth)

print()

# 绘制置信度预测图
fig = plt.figure(figsize=(20, 6))
y_label = ['X', 'Y', 'Z']
# 绘制预测结果
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(x_test, y_test[:, i], 'b-', label='observed value')  # 绘制观测点
    plt.plot(X_test, Y_pred[:, i], 'r-', label='predicted value')  # 绘制待预测点
    # plt.errorbar(X_test, Y_pred[:, i], yerr=1.96 * sigma[i], color='r', alpha=0.5)  # 绘制95%置信区间
    # plt.fill_between(X_test.flatten(), Y_pred[:, i] + sigma, Y_pred[:, i] - sigma)
    plt.axvline(x=inter_num, color='g', linestyle='--', label='predicted position')
    plt.xlabel('time step')
    plt.ylabel(y_label[i])
    plt.legend()
    plt.title(f'GPR prediction {y_label[i]}-axis results')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(y_test[:, 0], y_test[:, 1], y_test[:, 2])
# ax.plot(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2])
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# plt.show()
