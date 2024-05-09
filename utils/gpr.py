import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt


class GPR:
    def __init__(self):
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-7, 1e+5))  # 定义径向基函数协方差函数
        self.gpr = GaussianProcessRegressor(kernel=kernel,
                                            alpha=0.1,
                                            n_restarts_optimizer=5)

    def predict(self, poses):
        total_num = len(poses)
        pred_step_num = 5
        x_test = np.linspace(1, total_num, total_num).reshape(-1, 1)
        y_test = np.array(poses).reshape(total_num, 3)
        self.gpr.fit(x_test, y_test)
        x_pred = np.linspace(total_num + 1, total_num + pred_step_num, pred_step_num).reshape(-1, 1)
        y_pred, _ = self.gpr.predict(x_pred, return_std=True)  # 预测输出和方差

        return y_pred


if __name__ == '__main__':
    poses = np.array(
        [[[284.5, 226.25], [259.0, 187.5]], [[242.5, 208.0], [209.25, 166.25]], [[242.25, 207.5], [209.5, 165.375]],
         [[242.125, 207.5], [209.25, 166.75]], [[241.875, 208.25], [209.25, 167.0]],
         [[242.375, 208.75], [209.125, 167.5]], [[241.75, 209.0], [208.75, 167.75]],
         [[242.25, 209.5], [209.125, 169.0]], [[241.625, 209.75], [208.5, 169.25]],
         [[242.25, 209.5], [208.875, 169.125]], [[243.0, 210.25], [208.75, 170.75]],
         [[243.0, 210.5], [209.25, 170.625]], [[243.25, 210.25], [208.875, 171.0]],
         [[243.625, 210.25], [209.875, 170.5]], [[243.5, 210.5], [209.375, 170.0]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]], [[243.75, 210.75], [210.25, 171.25]],
         [[243.75, 210.75], [210.25, 171.25]]]
    )
    print(len(poses))
