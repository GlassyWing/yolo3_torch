import colorsys
import datetime
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
from matplotlib import patches
from matplotlib.ticker import NullLocator
from torch.autograd import Variable

from yolo3.dataset.dataset import pad_to_square, resize
from yolo3.utils.helper import load_classes
from yolo3.utils.model_build import non_max_suppression, rescale_boxes


def _get_statistic_info(detections, unique_labels, classes):
    """获得统计信息"""
    statistic_info = {}
    for label in unique_labels:
        statistic_info[classes[int(label)]] = (detections[:, -1] == label).sum().item()
    return statistic_info


def draw_rect(draw, detection, classes, colors, thickness, font):
    """绘制边框和标签"""
    x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
    label = '{} {:.2f}'.format(classes[int(cls_pred)], cls_conf)
    color = colors[int(cls_pred)]

    c1 = (int(x1), int(y1))
    c2 = (int(x2), int(y2))

    draw.rectangle([c1, c2], outline=tuple(color) + (128,), width=thickness)

    # 绘制文本框
    label_size = draw.textsize(label, font)

    if y1 - label_size[1] >= 0:
        text_origin = int(x1), int(y1) - label_size[1]
    else:
        text_origin = int(x1), int(y1) + 1
    draw.rectangle([text_origin, (text_origin[0] + label_size[0],
                                  text_origin[1] + label_size[1])],
                   fill=tuple(color) + (128,))
    draw.text(xy=text_origin, text=label, fill=(255, 255, 255), font=font)

    return label_size


def draw_summary(draw, font, summary, font_width, font_height, y_offset=60):
    """绘制摘要信息"""
    draw.rectangle(xy=((0, y_offset), (font_width, font_height * len(summary) + y_offset)),
                   fill=(0, 0, 0, 128))

    with ThreadPoolExecutor() as executor:
        for _ in executor.map(
                lambda x: draw.text((3, x[0] * font_height + y_offset), x[1][0] + ":" + str(x[1][1]),
                                    fill=(255, 255, 255, 128),
                                    font=font), enumerate(summary.items())):
            pass


def draw_single_img(img, detections, img_size, classes, colors, thickness, font, statistic=False, scaled=False):
    """绘制单张图片"""
    statistic_info = {}

    # Detected something
    if detections is not None:

        # 使用PIL绘制中文
        base = Image.fromarray(img).convert("RGBA")
        w, h = base.size

        # 如果检测框尚未进行缩放
        if not scaled:
            detections = rescale_boxes(detections, img_size, (h, w))
        unique_labels = detections[:, -1].unique()

        if statistic:
            statistic_info = _get_statistic_info(detections, unique_labels, classes)

        # make a blank image for text, rectangle, initialized to transparent color
        plane = Image.new("RGBA", base.size, (255, 255, 255, 0))

        draw = ImageDraw.Draw(plane)

        font_height = 0
        font_width = 0

        # 绘制所有标签
        with ThreadPoolExecutor() as executor:
            for fw, fh in executor.map(
                    lambda detection: draw_rect(draw, detection, classes, colors, thickness,
                                                font),
                    detections):
                font_height = max(font_height, fh)
                font_width = max(font_width, fw)

        if statistic:
            # 绘制统计信息
            draw_summary(draw, font=font,
                         summary=statistic_info,
                         font_width=font_width,
                         font_height=font_height
                         )
        del draw

        out = Image.alpha_composite(base, plane).convert("RGB")
        img = np.ndarray(buffer=out.tobytes(), shape=img.shape, dtype='uint8', order='C')

        return img, plane, statistic_info

    else:
        logging.debug("Nothing Detected.")
        return img, None, statistic_info


class ImageDetector:
    """图像检测器，只检测单张图片"""

    def __init__(self, model, class_path, thickness=2, font_path=None, font_size=10,
                 conf_thres=0.5,
                 nms_thres=0.4):
        self.model = model
        self.model.eval()
        self.classes = load_classes(class_path)
        self.num_classes = len(self.classes)
        self.thickness = thickness
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        if font_path is not None:
            self.font = ImageFont.truetype(font_path, font_size)
        else:
            self.font = ImageFont.load_default()

        # Prepare colors for each class
        hsv_color = [(1.0 * i / self.num_classes, 1., 1.) for i in range(self.num_classes)]
        colors = [colorsys.hsv_to_rgb(*x) for x in hsv_color]
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        self.colors = np.floor(np.asarray(colors) * 255).astype(int)

    def detect(self, img,
               output_path=None,
               statistic=False):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        image = transforms.ToTensor()(img)
        image = Variable(image.type(Tensor))

        if len(image.shape) != 3:
            image = image.unsqueeze(0)
            image = image.expand((3, image.shape[1:]))

        image, _ = pad_to_square(image, 0)
        image = resize(image, self.model.img_size)

        # Add batch dimension
        image = image.unsqueeze(0)

        prev_time = time.time()
        with torch.no_grad():
            detections = self.model(image)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = detections[0]

        img, plane, statistic_info = draw_single_img(img, detections,
                                                     self.model.img_size,
                                                     self.classes,
                                                     self.colors,
                                                     self.thickness,
                                                     self.font, statistic)

        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        logging.info("\t Inference time: %s" % inference_time)

        if output_path is not None:
            cv2.imwrite(output_path, img)

        return img, plane, detections, statistic_info


class ImageFolderDetector:
    """图像文件夹检测器，检测一个文件夹中的所有图像"""

    def __init__(self, model, class_path):
        self.model = model.eval()
        self.classes = load_classes(class_path)

    def detect(self, dataloader, output_dir, conf_thres=0.8, nms_thres=0.4):

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        imgs = []  # Stores image paths
        img_detections = []  # Stores detections for each image index

        prev_time = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            input_imgs = Variable(input_imgs.type(Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, conf_thres, nms_thres)

            current_time = time.time()
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            logging.info("\t+ Batch %d, Inference time: %s" % (batch_i, inference_time))

            imgs.extend(img_paths)
            img_detections.extend(detections)

        # Bounding-box colors
        colors = plt.get_cmap("tab20b").colors

        logging.info("\nSaving images:")

        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

            logging.info("(%d) Image: '%s'" % (img_i, path))
            # Create plot
            img = np.array(Image.open(path))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            if detections is not None:
                detections = rescale_boxes(detections, self.model.img_size, img.shape[:2])
                unique_labels = detections[:, -1].cpu().unique()
                n_cls_preds = len(unique_labels)
                bbox_colors = random.sample(colors, n_cls_preds)
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    logging.info("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=self.classes[int(cls_pred)],
                             color="white",
                             verticalalignment="top",
                             bbox={"color": color, "pad": 0})

            # Save generated image with detections
            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())

            filename = os.path.basename(path).split(".")[0]
            output_path = os.path.join(output_dir, filename + ".png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()
