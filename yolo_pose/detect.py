import time
from pathlib import Path

import copy
import time
from pathlib import Path

import numpy as np
import torch
import sys
import cv2


from yolo_pose.models.experimental import attempt_load
from yolo_pose.utils.datasets import LoadImages, letterbox
from yolo_pose.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from yolo_pose.utils.plots import colors, plot_one_box
from yolo_pose.utils.torch_utils import select_device, time_synchronized


class YoloPoseDetect:
    def __init__(self, dev, wgt, imgsz=640, view_img=True, augment=False,
                 conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, hide_labels=True, hide_conf=True):
        self.weights = wgt
        self.kpt_label = 'label'

        self.imgsz = imgsz
        self.view_img = view_img
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

        # Initialize
        set_logging()
        self.device = select_device(dev)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model


    def detect(self, image):
        # source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
        # save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        #     ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        # (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        stride = int(self.model.stride.max())  # model stride

        # if isinstance(self.imgsz, (list,tuple)):
        #     assert len(self.imgsz) ==2; "height and width of image has to be specified"
        #     imgsz[0] = check_img_size(self.imgsz[0], s=stride)
        #     imgsz[1] = check_img_size(self.imgsz[1], s=stride)
        # else:
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        #     modelc.load_state_dict(torch.load('last.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        # dataset = LoadImages(self.image, img_size=imgsz, stride=stride)

        im0s = image  # BGR
        assert im0s is not None, 'Image Is None'
        # Padded resize
        img = letterbox(im0s, imgsz, stride=stride, auto=False)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()
        pec_kpts = {}
        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        # print(pred[..., 4].max())
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                   agnostic=self.agnostic_nms, kpt_label=self.kpt_label)
        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # else:
            s, im0 = '', im0s.copy()

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=self.kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or opt.save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    kpts = det[det_index, 6:]

                    # 将8个人体关键点存入字典中
                    kptsc = copy.deepcopy(kpts)
                    kptsc = kptsc.view(17, 3)[5: 13]
                    pec_kpts_class = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                      'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
                    pec_kpts['id'] = None
                    for index, kpt_class in enumerate(pec_kpts_class):
                        pec_kpts[kpt_class] = [value.tolist() for value in kptsc[index][:2]]
                    # print(pec_kpts)

                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3,
                                 kpt_label=self.kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                    # if opt.save_crop:
                    #     save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # if save_txt_tidl:  # Write to file in tidl dump format
                #     for *xyxy, conf, cls in det_tidl:
                #         xyxy = torch.tensor(xyxy).view(-1).tolist()
                #         line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            # if self.view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(0)

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)
        #
        # if save_txt or save_txt_tidl or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        #     print(f"Results saved to {save_dir}{s}")

        # print(f'Done. ({time.time() - t0:.3f}s)')

        return pec_kpts, im0


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='images', help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    # parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    # parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    # parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    # parser.add_argument('--kpt-label', default='label', help='use keypoint labels')
    # opt = parser.parse_args()
    # print(opt)
    # check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))
    #
    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    #             detect(opt=opt)
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect(opt=opt)

    image_path = 'images/0103.png'
    image = cv2.imread(image_path)
    device = '0'
    weights = 'last.pt'
    yolo_pose_detect = YoloPoseDetect(device, weights)
    pec_keypoints, img = yolo_pose_detect.detect(image)
    cv2.imshow('pig', img)
    cv2.waitKey(0)
    print(pec_keypoints)
    for key, val in pec_keypoints.items():
        print(val)
