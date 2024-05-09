from PIL import Image
import numpy as np

def normalize_depth(depth_map):
    """ 归一化深度图 """
    mean = np.mean(depth_map)
    std = np.std(depth_map)
    depth_map_norm = (depth_map - mean) / std  # 将像素值缩放到0-1范围内
    print(mean, std)
    return depth_map_norm

pic = Image.open('PE_Dataset/v_1_g02-1_c01/depth/000.tiff', 'r').convert('F')
norm = normalize_depth(np.array(pic))
print(np.mean(norm), np.std(norm))
