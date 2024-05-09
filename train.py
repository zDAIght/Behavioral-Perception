import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from utils.function import train, validation, DatasetProcess
from model.resnet_rnn import ClassifyRegressNet
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR


def run():
    #  读取动作标签文件
    # with open(action_name_path, 'r') as file:
    #     action_names = []
    #     for line in file:
    #         data_line = line.strip("\n")
    #         action_names.append(data_line)
    #
    # # 将标签编码为分类编号
    # le = LabelEncoder()
    # le.fit(action_names)
    # # a = list(le.classes_)
    # action_category = le.transform(action_names).reshape(-1, 1)
    # enc = OneHotEncoder()
    # enc.fit(action_category)

    # 创建tensorboard
    writer = SummaryWriter(log_dir='runs/accuracy_'+hms_time)

    # 从文件名称中提取动作分类标签
    fnames = os.listdir(data_path)
    fnames.sort()
    all_image, all_label = [], []
    for f in fnames:
        # 获取动作分类
        loc1 = f.find('a_')
        loc2 = f.find('_s')
        action = f[(loc1 + 2): loc2]

        all_image.append(f)
        if is_regress:
            # 获取回归轨迹
            label_txt_path = os.path.join(data_path, f, "label.txt")
            with open(label_txt_path, 'r') as file:
                pose = np.array([])
                for line in file:
                    data_line = line.strip("\n").split(' ')
                    data_line = [np.float(x)*1000 for x in data_line]
                    pose = np.append(pose, data_line)
            all_label.append([int(action), list(pose)])
        else:
            all_label.append(int(action))

    # 获取所有训练数据
    all_X_list = all_image  # 所有视频帧图像
    all_y_list = all_label  # 所有视频分类标签和轨迹

    # 划分数据集
    train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.2,
                                                                      random_state=42)

    # 将图片处理拼接后导入程序
    train_set = DatasetProcess(data_path, train_list, train_label, [res_size, res_size],
                               use_depth=use_depth, is_regress=is_regress)
    valid_set = DatasetProcess(data_path, test_list, test_label, [res_size, res_size],
                               use_depth=use_depth, is_regress=is_regress)

    train_loader = data.DataLoader(train_set, **params)
    valid_loader = data.DataLoader(valid_set, **params)

    # 生成单分支分类网络
    # net = ResCNNClassify(fc_hidden1=1024, fc_hidden2=768, drop_p=0,
    #                      embed_dim=512, class_num=class_num).to(device)
    net = ClassifyRegressNet(fc_hidden1=1024, fc_hidden2=768, drop_p=0.5, embed_dim=512,
                             class_num=class_num, regress_num=regress_num,
                             is_regress=is_regress, use_depth=use_depth).to(device)
    print("创建网络...\n")

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
    print("导入图像...\n")
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
            if epoch_test_score[0] > 0.95 and epoch_test_score[1] > 0.90:
                torch.save(net.state_dict(), os.path.join(os.path.join(save_model_path, hms_time),
                                                          'ClsRegNet_{:.3f}-{:.3f}.pth'.format(
                                                              epoch_test_score[0], epoch_test_score[1])))
                print("已保存模型：ClsRegNet_{:.3f}-{:.3f}.pth".format(epoch_test_score[0], epoch_test_score[1]))

            writer.add_scalars(main_tag="loss",
                               tag_scalar_dict={"train_loss": np.mean(train_losses),
                                                "val_loss": np.mean(epoch_test_loss)},
                               global_step=epoch)
            writer.add_scalars(main_tag="Classification Accuracy",
                               tag_scalar_dict={"train_cls_acc": np.round(np.mean(train_scores[0])*100, 2),
                                                "val_cls_acc": np.round(np.mean(epoch_test_score[0])*100, 2)},
                               global_step=epoch)
            writer.add_scalars(main_tag="Regression Accuracy",
                               tag_scalar_dict={"train_reg_acc": np.round(np.mean(train_scores[1])*100, 2),
                                                "val_reg_acc": np.round(np.mean(epoch_test_score[1])*100, 2)},
                               global_step=epoch)
        else:
            # 保存准确率前三的模型
            if epoch_test_score > min(top_kc):
                idx = top_kc.index(min(top_kc))
                top_kc[idx] = epoch_test_score
                models_params[idx] = net.state_dict()
            if epoch_test_score > 0.97:
                torch.save(net.state_dict(), os.path.join(os.path.join(save_model_path, hms_time),
                                                          'ClassifyNet_{:.3f}.pth'.format(epoch_test_score)))
                print("已保存模型：ClassifyNet_{:.3f}.pth".format(epoch_test_score))

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

    if not is_regress:
        # plot
        plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(1, epochs + 1), train_loss[:, -1])  # train loss (on epoch end)
        plt.plot(np.arange(1, epochs + 1), test_loss)  # test loss (on epoch end)
        plt.title("model loss")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        plt.plot(np.arange(1, epochs + 1), train_score[:, -1])  # train accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), test_score)  # test accuracy (on epoch end)
        plt.title("training scores")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc="upper left")
        result_figure = "result_figure/classify_" + hms_time + ".png"
        plt.savefig(result_figure, dpi=600)
        # plt.close(fig)
        plt.show()


if __name__ == '__main__':

    # 设置路径
    data_path = "PE_Dataset"  # PE数据集路径
    action_name_path = 'classify_actions.txt'  # 动作分类编号
    save_model_path = "model_ckpt"  # 模型保存路径
    save_data_path = "train_result"  # 训练数据保存路径
    hms_time = time.strftime("%Y-%m-%d-%H:%M:%S")

    # 训练参数
    res_size = 224  # ResNet 图像尺寸
    class_num = 10  # 分类数量
    regress_num = 28 * 3  # 轨迹回归数量
    epochs = 100
    batch_size = 10
    log_interval = 6  # 打印训练信息的间隔

    # 设置学习率
    init_learning_rate = 1e-3  # 初始值
    step_size = int(epochs/2)  # 设置多少个epoch后调整一次
    gamma = 0.1  # 衰减系数

    w1, w2 = 0.7, 2.0  # 分类回归网络的loss比重
    use_depth = True
    is_regress = True

    # 检查GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 记录准确率前三和对应模型参数的列表
    top_kr = [[0., 0.]] * 3
    top_kc = [0.0] * 3
    models_params = [None] * 3

    # 训练参数
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0,
              'pin_memory': True, 'drop_last': True} if use_cuda else {}

    run()
