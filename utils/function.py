import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2

# action_encoder = {0: "take tube", 1: "give up", 2: "draw up liquid", 3: "place tube"}
action_encoder = {0: "approach", 1: "take PS", 2: "wring PS", 3: "take sample",
                  4: "sampling", 5: "discard sample", 6: "wring PS",
                  7: "place PS", 8: "operate oscillator", 9: "departure"}

coax_annotation = {"[0, None]": "approach", "[1, 0]": "grab valve", "[1, 1]": "grab capacitor",
                   "[1, 2]": "grab screwdriver", "[1, 3]": "grab screws", "[1, 4]": "grab membrane",
                   "[1, 5]": "grab soldering tin", "[1, 6]": "grab soldering iron", "[1, 7]": "grab hose",
                   "[2, 0]": "plug valve", "[2, 1]": "plug capacitor", "[2, 3]": "plug screws", "[2, 7]": "plug hose",
                   "[3, 0]": "join valve", "[4, None]": "wait for robot", "[5, 0]": "screw valve",
                   "[6, 2]": "release screwdriver", "[6, 5]": "release soldering tin",
                   "[6, 6]": "release soldering iron", "[7, 1]": "solder capacitor",
                   "[8, 0]": "place valve", "[8, 4]": "place membrane", "[9, None]": "retreat"}

pec_annotation = {"A0": "Approach", "A1O0": "TakeT-tube", "A1O1": "TakePreservCyt", "A1O2": "TakePipette",
                  "A1O3": "TakeSample", "A2O1": "TwistPreservCyt", "A3O0": "PlaceT-tube", "A3O1": "PlacePreservCyt",
                  "A3O2": "PlacePipette", "A3O3": "PlaceSample", "A4O0": "UseT-tube", "A4O2": "UsePipette",
                  "A4O3": "UseSample", "A4O4": "UseOscillator", "A5O3": "DiscardSample", "A6": "WaitingRobot",
                  "A7": "Departure"}

action_target = {"Approach": [], "TakeT-tube": [-0.60, -0.14, 0.38], "TakePreservCyt": [], "TakePipette": [],
                 "TakeSample": [], "TwistPreservCyt": [], "PlaceT-tube": [], "PlacePreservCyt": [],
                 "PlacePipette": [], "PlaceSample": [], "UseT-tube": [], "UsePipette": [],
                 "UseSample": [], "UseOscillator": [], "DiscardSample": [], "WaitingRobot": [],
                 "Departure": []
                 }

# 图像变换
channel4_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406, 1.0],
                                                              std=[0.229, 0.224, 0.225, 0.51])])
channel3_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


# 单分支resnet+cnn分类网络
class ResCNNClassify(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, embed_dim=512, class_num=5):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNClassify, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.class_num = class_num
        self.embed_dim = embed_dim
        self.num_feature = 512

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.conv4_3 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=False)  # 将4通道转换为3通道

        # CNN分类器
        self.cnn_feature = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),

            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        )
        # CNN连接全连接层，输出分类class_num
        self.fc1 = nn.Linear(64, 512)  # fully connected layer, output k classes
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x_3d):
        features = []
        # 将28层图像依照时序关系依次提取特征
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                # x = self.resnet(self.conv4_3(x_3d[:, t, :, :, :]))
                x = self.resnet(x_3d[:, t, :3, :, :])

                '''从这里开始，features的特征通过两个分支进行分类和回归'''
                x = self.cnn_feature(x)  # CNN分类器
                x = nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])

                x = x.view(x.size(0), -1)  # 拉平接全连接层

            # FC layers
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)

            features.append(x)

        features = torch.stack(features, dim=0).transpose_(0, 1)

        # 分类
        class_x = features
        # 将张量 x reshape 为形状为 [batch_size, num_features] 的二维张量
        num_features = class_x.shape[1] * class_x.shape[2]
        class_x = class_x.reshape(-1, num_features)

        # 定义一个全连接层 fc，输入维度为 num_features，输出维度为分类数
        class_fc1 = torch.nn.Linear(num_features, 256).to('cuda:0')
        bn = nn.BatchNorm1d(256, momentum=0.01).to('cuda:0')
        class_fc2 = torch.nn.Linear(256, self.class_num).to('cuda:0')
        # class_fc = torch.nn.Linear(num_features, self.class_num).to('cuda:0')

        class_x = bn(class_fc1(class_x))
        class_x = F.relu(class_x)
        y_class = class_fc2(class_x)
        # y_class = class_fc(class_x)
        return y_class


class DatasetProcess(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data_path, folders, labels, size, use_depth=False, is_regress=False):
        """Initialization"""
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.size = size
        self.use_depth = use_depth
        self.is_regress = is_regress

        if self.use_depth:
            self.transform = channel3_transform
        else:
            self.transform = channel3_transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []

        color_path = os.path.join(path, selected_folder, "color")
        color_image_list = [os.path.join(color_path, i)
                            for i in os.listdir(color_path)]
        color_image_list.sort()

        if self.use_depth:
            depth_path = os.path.join(path, selected_folder, "depth")
            depth_image_list = [os.path.join(depth_path, i)
                                for i in os.listdir(depth_path)]
            depth_image_list.sort()

            if len(color_image_list) != len(depth_image_list):
                assert ValueError("图像数量不对等")

        for i in range(len(color_image_list)):

            if self.use_depth:
                pic_rgb = Image.open(color_image_list[i], 'r').convert('RGB')
                rgb = np.array(pic_rgb)

                pic_dep = Image.open(depth_image_list[i], 'r').convert('F')
                arr_depth = np.array(pic_dep).astype(float)
                arr_depth -= arr_depth.min()
                arr_depth *= 255 / arr_depth.max()

                depth = np.expand_dims(np.array(arr_depth), axis=2)

                x = np.concatenate((rgb, depth), axis=2)
                image = Image.fromarray(np.uint8(x)).convert('RGB')
            else:
                image = Image.open(color_image_list[i], 'r').convert('RGB')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform)  # 拼接图片
        if self.is_regress:
            y_class = torch.LongTensor([self.labels[index][0]])  # 标签分类编号为长整形
            y_regress = torch.FloatTensor([self.labels[index][1]])  # 数据类型为坐标值float类型
            y = [y_class, y_regress.squeeze()]
        else:
            y = torch.LongTensor([self.labels[index]])  # 标签分类编号为长整形

        # print(X.shape)
        return X, y


class PECDatasetProcess(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, folders, labels, size, pose, use_depth=False, is_regress=False):
        """Initialization"""
        self.labels = labels
        self.folders = folders
        self.size = size
        self.use_depth = use_depth
        self.is_regress = is_regress
        self.pose = pose

        if self.use_depth:
            self.transform = channel3_transform
        else:
            self.transform = channel3_transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.folders)

    def read_images(self, selected_folder, use_transform):
        X = []

        for images in selected_folder:
            if self.use_depth:

                pic_rgb = Image.open(images[0], 'r').convert('RGB')
                rgb = np.array(pic_rgb)  # BGR
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # RGB
                _, rgb = self.pose.detect(rgb)  # 将关键点绘制在图像上

                pic_dep = Image.open(images[1], 'r').convert('F')
                arr_depth = np.array(pic_dep).astype(float)
                arr_depth -= arr_depth.min()
                arr_depth *= 255 / arr_depth.max()

                depth = np.expand_dims(np.array(arr_depth), axis=2)

                x = np.concatenate((rgb, depth), axis=2)
                image = Image.fromarray(np.uint8(x)).convert('RGB')
            else:
                image = Image.open(images[0], 'r').convert('RGB')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(folder, self.transform)  # 拼接图片
        if self.is_regress:
            y_class = torch.LongTensor([self.labels[index][0]])  # 标签分类编号为长整形
            y_regress = torch.FloatTensor([self.labels[index][1]])  # 数据类型为坐标值float类型
            y = [y_class, y_regress.squeeze()]
        else:
            y = torch.LongTensor([self.labels[index]])  # 标签分类编号为长整形

        # print(X.shape)
        return X, y


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def train(log_interval, net, device, train_loader, optimizer, epoch, weight, is_regress=False):
    net.train()
    w1, w2 = weight
    losses = []
    cls_scores, reg_scores, scores = [], [], []
    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        if is_regress:
            X, y_class, y_regress = X.to(device), y[0].to(device).view(-1, ), y[1].to(device)

            N_count += X.size(0)

            optimizer.zero_grad()
            output_class, output_regress = net(X)  # output has dim = (batch, number of classes)

            loss_class = F.cross_entropy(output_class, y_class)
            loss_regress = F.mse_loss(output_regress, y_regress)
            loss = w1 * loss_class + w2 * loss_regress

            losses.append(loss.item())

            # 计算分类准确率
            y_cls_pred = torch.max(output_class, 1)[1]  # y_pred != output
            cls_step_score = accuracy_score(y_class.cpu().data.squeeze().numpy(),
                                            y_cls_pred.cpu().data.squeeze().numpy())
            cls_scores.append(cls_step_score)
            # 计算回归精度，即预测值与真实值误差小于dis的占比
            dis = 0.05  # 数据单位为mm，设置误差精度为0.1mm
            diff_matrix = torch.abs((output_regress - y_regress))
            count = torch.sum(diff_matrix < dis)
            reg_step_score = count / (y_regress.shape[0] * y_regress.shape[1])
            reg_scores.append(reg_step_score.cpu().data.squeeze().numpy())
            scores = [cls_scores, reg_scores]

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, Cls_Acc: {:.2f}%, Reg_Acc: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    100 * cls_step_score, 100 * reg_step_score))

        else:
            X, y_class = X.to(device), y.view(-1, ).to(device)

            N_count += X.size(0)

            optimizer.zero_grad()
            output = net(X)[0]  # output has dim = (batch, number of classes)

            loss = F.cross_entropy(output, y_class)
            losses.append(loss.item())

            # to compute accuracy
            y_cls_pred = torch.max(output, 1)[1]  # y_pred != output
            step_score = accuracy_score(y_class.cpu().data.squeeze().numpy(), y_cls_pred.cpu().data.squeeze().numpy())
            cls_scores.append(step_score)
            scores = cls_scores

            loss.backward()
            optimizer.step()

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    100 * step_score))

    return losses, scores


def validation(net, device, optimizer, test_loader, save_model_path, epoch, weight, is_regress=False):
    # set model as testing mode
    net.eval()
    w1, w2 = weight
    scores = []
    cls_test_loss, reg_test_loss, losses = 0, 0, 0
    cls_y, reg_y = [], []
    cls_y_pred, reg_y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            if is_regress:
                X, y_class, y_regress = X.to(device), y[0].to(device).view(-1, ), y[1].to(device)
                output_class, output_regress = net(X)

                loss_class = F.cross_entropy(output_class, y_class)
                loss_regress = F.mse_loss(output_regress, y_regress)
                loss = w1 * loss_class + w2 * loss_regress
                losses += loss.item()
                y_pred = output_class.max(1, keepdim=True)[1]

                cls_test_loss += loss_class.item()
                cls_y.extend(y_class)
                cls_y_pred.extend(y_pred)

                reg_test_loss += loss_regress.item()
                reg_y.extend(y_regress)
                reg_y_pred.extend(output_regress)

            else:
                X, y = X.to(device), y.to(device).view(-1, )

                output = net(X)[0]

                loss = F.cross_entropy(output, y, reduction='sum')
                losses += loss.item()  # sum up batch loss
                y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

                # collect all y and y_pred in all batches
                cls_y.extend(y)
                cls_y_pred.extend(y_pred)

    if is_regress:
        losses /= len(test_loader.dataset)

        # compute accuracy
        cls_y = torch.stack(cls_y, dim=0)
        cls_y_pred = torch.stack(cls_y_pred, dim=0)
        cls_test_score = accuracy_score(cls_y.cpu().data.squeeze().numpy(), cls_y_pred.cpu().data.squeeze().numpy())

        reg_y = torch.stack(reg_y, dim=0)
        reg_y_pred = torch.stack(reg_y_pred, dim=0)

        # 计算回归精度，即预测值与真实值误差小于dis的占比
        dis = 0.05  # 回归误差精度
        diff_matrix = torch.abs((reg_y - reg_y_pred))
        count = torch.sum(diff_matrix < dis)
        reg_test_score = count / (reg_y.shape[0] * reg_y_pred.shape[1])
        scores = [cls_test_score, reg_test_score.cpu().data.squeeze().numpy()]
        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Cls_Accuracy: {:.2f}%, Reg_Accuracy: {:.2f}%\n'.format(
            len(cls_y), losses, 100 * cls_test_score, 100 * reg_test_score))
    else:
        losses /= len(test_loader.dataset)

        # compute accuracy
        cls_y = torch.stack(cls_y, dim=0)
        cls_y_pred = torch.stack(cls_y_pred, dim=0)
        cls_test_score = accuracy_score(cls_y.cpu().data.squeeze().numpy(), cls_y_pred.cpu().data.squeeze().numpy())
        scores = cls_test_score
        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            len(cls_y), losses, 100 * cls_test_score))

    return losses, scores


def image_convert_from_dir(path, folder, use_depth):
    color_path = os.path.join(path, folder, "color")
    color_image_list = [os.path.join(color_path, i)
                        for i in os.listdir(color_path)]
    color_image_list.sort()
    depth_path = os.path.join(path, folder, "depth")

    if use_depth:
        depth_image_list = [os.path.join(depth_path, i)
                            for i in os.listdir(depth_path)]
        depth_image_list.sort()

        if len(color_image_list) != len(depth_image_list):
            assert ValueError("图像数量不对等")

    video = []
    for i in range(len(color_image_list)):
        if use_depth:
            pic_rgb = Image.open(color_image_list[i], 'r').convert('RGB')
            rgb = np.array(pic_rgb)

            pic_dep = Image.open(depth_image_list[i], 'r').convert('F')
            arr_depth = np.array(pic_dep).astype(float)
            arr_depth -= arr_depth.min()
            arr_depth *= 255 / arr_depth.max()

            depth = np.expand_dims(np.array(arr_depth), axis=2)

            x = np.concatenate((rgb, depth), axis=2)
            image = Image.fromarray(np.uint8(x)).convert('RGB')

            image = channel3_transform(image)

        else:
            image = Image.open(color_image_list[i], 'r').convert('RGB')
            image = channel3_transform(image)

        video.append(image)
    X = torch.stack(video, dim=0)
    X = X.unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    loc1 = folder.find('a_')
    loc2 = folder.find('_s')
    y = folder[(loc1 + 2): loc2]

    # 获取回归轨迹
    label_txt_path = os.path.join(path, folder, "label.txt")
    with open(label_txt_path, 'r') as file:
        pose = np.array([])
        for line in file:
            data_line = line.strip("\n").split(' ')
            data_line = [np.float(x) * 1000 for x in data_line]
            pose = np.append(pose, data_line)

    return X, int(y), pose


def image_convert_from_coaxDir(path, use_depth):
    video = []
    for images in path[1]:
        if use_depth:
            pic_rgb = Image.open(images[0], 'r').convert('RGB')
            rgb = np.array(pic_rgb)

            pic_dep = Image.open(images[1], 'r').convert('F')
            arr_depth = np.array(pic_dep).astype(float)
            arr_depth -= arr_depth.min()
            arr_depth *= 255 / arr_depth.max()

            depth = np.expand_dims(np.array(arr_depth), axis=2)

            x = np.concatenate((rgb, depth), axis=2)
            image = Image.fromarray(np.uint8(x)).convert('RGB')

            image = channel3_transform(image)

        else:
            image = Image.open(images[0], 'r').convert('RGB')
            image = channel3_transform(image)

        video.append(image)
    X = torch.stack(video, dim=0)
    X = X.unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    y = path[0][0]

    # 获取回归轨迹
    pose = np.array(path[0][1])

    return X, y, pose


def image_convert_from_pecDir(path, use_depth, is_regress):
    video = []
    for images in path[1]:
        if use_depth:
            pic_rgb = Image.open(images[0], 'r').convert('RGB')
            rgb = np.array(pic_rgb)

            pic_dep = Image.open(images[1], 'r').convert('F')
            arr_depth = np.array(pic_dep).astype(float)
            arr_depth -= arr_depth.min()
            arr_depth *= 255 / arr_depth.max()

            depth = np.expand_dims(np.array(arr_depth), axis=2)

            x = np.concatenate((rgb, depth), axis=2)
            image = Image.fromarray(np.uint8(x)).convert('RGB')

            image = channel3_transform(image)

        else:
            image = Image.open(images[0], 'r').convert('RGB')
            image = channel3_transform(image)

        video.append(image)
    X = torch.stack(video, dim=0)
    X = X.unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    y = path[0]

    if is_regress:
        # 获取回归轨迹
        pose = np.array(path[0][1])

        return X, y, pose

    else:
        return X, y, None


def image_convert_from_img(image_list, use_depth):
    video = []
    for i in range(len(image_list)):
        if use_depth:
            rgb = np.array(image_list[i][0])

            arr_depth = np.array(image_list[i][1]).astype(float)
            arr_depth -= arr_depth.min()
            arr_depth *= 255 / arr_depth.max()
            depth = np.expand_dims(np.array(arr_depth), axis=2)

            x = np.concatenate((rgb, depth), axis=2)
            image = Image.fromarray(np.uint8(x)).convert('RGB')

            image = channel3_transform(image)

        else:
            pic = Image.fromarray(image_list[i])
            image = channel3_transform(pic)

        video.append(image)
    X = torch.stack(video, dim=0)
    X = X.unsqueeze(0).to('cuda:0' if torch.cuda.is_available() else 'cpu')

    return X
