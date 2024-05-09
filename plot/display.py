import json

import numpy as np
from PIL import Image
import cv2
import os
import open3d as o3d


def depth2Gray(im_depth):
    """
    将深度图转至三通道8位灰度图
    (h, w, 3)
    """
    # 16位转8位
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('图像渲染出错 ...')
        raise EOFError

    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    return ret


def depth2RGB(im_depth):
    """
    将深度图转至三通道8位彩色图
    先将值为0的点去除,然后转换为彩图,然后将值为0的点设为红色
    (h, w, 3)
    im_depth: 单位 mm或m
    """
    im_depth = depth2Gray(im_depth)
    im_color = cv2.applyColorMap(im_depth, cv2.COLORMAP_JET)
    return im_color


def in_paint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


color_file = os.path.join("../PE_Dataset/a_7_s03-5_c02", "color")
col_file = [os.path.join(color_file, i) for i in os.listdir(color_file)]
col_file.sort()
depth_file = os.path.join("../PE_Dataset/a_7_s03-5_c02", "depth")
dep_file = [os.path.join(depth_file, i) for i in os.listdir(depth_file)]
dep_file.sort()

count = 1
# for i in range(0, 28, 3):
#     # 打开png图像
#     color_image = cv2.imread(col_file[i])
#     cv2.imshow('color', color_image)
#     cv2.waitKey()
#     # 打开TIFF文件并加载图像
#     depth_image = Image.open(dep_file[i])
#     depth_image = np.asanyarray(depth_image).astype(np.float32)
#     # depth_image = in_paint(depth_image)  # 补全缺失值
#     depth_image_color = depth2RGB(depth_image)
#     cv2.imshow('depth', depth_image_color)
#     cv2.waitKey(0)
#     cv2.imwrite(f'images/1/c{count}.png', color_image)
#     cv2.imwrite(f'images/1/d{count}.png', depth_image_color)
#     cv2.destroyAllWindows()
#     count += 1

with open('../D455_intrinsics.json', 'r') as f:
    intr = json.load(f)
intr_width = intr['width']
intr_height = intr['height']
intr_fx = intr['fx']
intr_fy = intr['fy']
intr_ppx = intr['ppx']
intr_ppy = intr['ppy']

index = 18
color_raw = Image.open(col_file[index])
depth_raw = Image.open(dep_file[index])
color_image = o3d.geometry.Image(np.array(color_raw))
depth_image = o3d.geometry.Image(np.array(depth_raw))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                convert_rgb_to_intensity=False)

inter = o3d.camera.PinholeCameraIntrinsic(intr_width, intr_height, intr_fx, intr_fy, intr_ppx, intr_ppy)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, inter)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                  point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

# 去除离群点
downpcd = pcd.voxel_down_sample(voxel_size=0.000003)
cl, ind = downpcd.remove_radius_outlier(nb_points=16, radius=0.00007)
inlier_cloud = downpcd.select_by_index(ind)

o3d.visualization.draw_geometries([inlier_cloud], window_name='Open3D Origin', width=1920, height=1080, left=50, top=50,
                                  point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
# 保存处理后的点云
# o3d.io.write_point_cloud(os.path.join("plot/images/1", "{:03d}.pcd".format(index)), inlier_cloud)
