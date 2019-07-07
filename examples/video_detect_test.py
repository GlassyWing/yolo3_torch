import cv2

from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet
import logging
import torch.nn as nn

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("../config/yolov3.cfg", img_size=416)
    model.load_darknet_weights("../weights/yolov3.weights")
    model.to("cuda:0")

    video_detector = VideoDetector(model, "../config/coco.names",
                                   font_path="../font/sarasa-bold.ttc",
                                   font_size=18,
                                   skip_frames=20,
                                   conf_thres=0.7,
                                   nms_thres=0.2)

    frames = 0
    for image in video_detector.detect("../data/f35.flv",
                                          # output_path="../data/output.ts",
                                          real_show=True,
                                          show_statistic=True,
                                          skip_times=0):
        # if frames > 10:
        #     break
        # frames += 1
        pass
