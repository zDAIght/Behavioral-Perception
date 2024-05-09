import os
import cv2
import json
from tqdm import tqdm
from detect import YoloPoseDetect

# yolo-pose参数设置
dev = '0'
wgt = 'last.pt'
points_detect = YoloPoseDetect(dev, wgt)

color_images_path = "/home/magic/ZhangGZ/intention-aware-HRC/PEC_dataset_process/PEC_DataSet/Images"
depth_images_path = "/home/magic/ZhangGZ/intention-aware-HRC/PEC_dataset_process/PEC_DataSet/Depth_Images"
driver_file_path = "/home/magic/ZhangGZ/intention-aware-HRC/PEC_dataset_process/PEC_DataSet/Driver_Files"

tasks = [os.path.join(color_images_path, t) for t in os.listdir(color_images_path)]
tasks.sort()
keypoints_json = {"pec_key_points": []}
count = 1

for task in tasks:
    subjects = [os.path.join(task, s) for s in os.listdir(task)]
    subjects.sort()
    for subject in subjects:
        takes = [os.path.join(subject, k) for k in os.listdir(subject)]
        takes.sort()
        for take in takes:
            images = [os.path.join(take, "color", rgb) for rgb in os.listdir(take + "/color")]
            images.sort()
            print("{} / {}".format(count, len(tasks)*len(subjects)*len(takes)))
            for image in tqdm(images):
                img = cv2.imread(image)
                pec_key_points, _ = points_detect.detect(img)
                pec_key_points['id'] = os.path.basename(image)
                keypoints_json["pec_key_points"].append(pec_key_points)

            save_path = take.replace("Images", "Driver_Files")
            with open(os.path.join(save_path, "pec_keypoints.json"), 'w') as file:
                json.dump(keypoints_json, file, indent=4)
            count += 1
            keypoints_json["pec_key_points"] = []
            # print(keypoints_json)
