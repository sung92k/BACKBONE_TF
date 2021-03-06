import tensorflow as tf
import tqdm
import math
import numpy as np
import cv2
import albumentations
from config import *


class ImageNet_Generator(tf.keras.utils.Sequence):
    def __init__(self, dataset_info_path, batch_size, input_shape, num_classes, augs, is_train=True,
                 label_smooting=False):
        self.dataset_info_path = dataset_info_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.augs = augs
        self.is_train = is_train
        self.label_smooting = label_smooting
        self.data = self.get_dataset(dataset_info_path)
        self.indexes = None
        self.on_epoch_end()

    def get_dataset(self, dataset_info_path):
        img_list = []
        with open(dataset_info_path, 'r') as file:
            lines = file.readlines()
            for img_path in lines:
                img_path = img_path.replace("\\", "/").replace("\n", "")
                img_list.append(img_path)
        return img_list

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data)
        return x, y

    def __data_gen(self, data):
        cv2.setNumThreads(0)
        batch_img = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                             dtype=np.float32)
        batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)
        img_list = []
        cls_list = []
        for img_path in data:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            img_list.append(img)
            txt_path = img_path.replace(img_format, ".txt")
            with open(txt_path, "r") as file:
                lines = file.readlines()
                for index, line in enumerate(lines):
                    if index == 0:
                        cls = line.replace("\n", "")
                        cls_list.append(str(cls))
                        break

        for i in range(len(data)):
            img = cv2.imread(data[i])
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img = self.augs(image=img)['image']
            cls = cls_list[i]
            cls = tf.keras.utils.to_categorical(cls, num_classes=self.num_classes)
            batch_img[i] = img
            batch_cls[i] = cls
        return batch_img, batch_cls


class Yolo_Generator(tf.keras.utils.Sequence):
    def __init__(self, dataset_info_path, batch_size, input_shape, num_classes, anchors, output_shape_list, augs,
                 iou_threshold=0.5, is_train=True, label_smooting=False):
        self.dataset_info_path = dataset_info_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.anchors = anchors
        self.output_shape_list = output_shape_list
        self.n_branch = len(self.output_shape_list)
        self.n_anchor_per_branch = int(len(self.anchors) / len(self.output_shape_list))
        self.augs = augs
        self.iou_threshold = iou_threshold
        self.is_train = is_train
        self.label_smooting = label_smooting
        self.data = self.get_dataset(dataset_info_path)
        self.indexes = None
        self.on_epoch_end()

    def get_dataset(self, dataset_info_path):
        img_list = []
        with open(dataset_info_path, 'r') as file:
            lines = file.readlines()
            for img_path in lines:
                img_path = img_path.replace("\\", "/").replace("\n", "")
                img_list.append(img_path)
        return img_list

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        data = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data)
        return x, y

    def get_iou(self, bbox1, bbox2):
        '''
        :param bbox1: w, h
        :param bbox2: w, h
        :return: iou value
        '''

        bbox1_area = bbox1[0] * bbox1[1]
        bbox2_area = bbox2[0] * bbox2[1]

        intersection_w = min(bbox1[0], bbox2[0])
        intersection_h = min(bbox1[1], bbox2[1])

        intersection_area = intersection_w * intersection_h

        iou = intersection_area / (bbox1_area + bbox2_area - intersection_area + EPS)
        return iou

    def __data_gen(self, data):
        cv2.setNumThreads(0)
        batch_img = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        batch_gt = []
        for out_shape in self.output_shape_list:
            batch_gt.append(np.zeros((self.batch_size, out_shape[0], out_shape[1], self.n_anchor_per_branch, (5 + self.num_classes)), dtype=np.float32))
        for index, img_path in enumerate(data):
            img = cv2.imread(img_path)
            txt_path = img_path.replace(img_format, ".txt")
            bboxes = []
            with open(txt_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    cls, cx, cy, w, h = line.split(" ")
                    cx = int(float(cx) * img.shape[1])
                    cy = int(float(cy) * img.shape[0])
                    w = int(float(w) * img.shape[1])
                    h = int(float(h) * img.shape[0])
                    x_min = max(0, cx - w / 2)
                    y_min = max(0, cy - h / 2)
                    if w != 0 and h != 0:
                        bboxes.append([x_min, y_min, w, h, cls])
            transformed = self.augs(image=img, bboxes=bboxes)
            transformed_img = transformed['image']
            transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
            transformed_bboxes = transformed['bboxes']
            batch_img[index] = transformed_img / 255.
            for bbox in transformed_bboxes:
                x_min, y_min, w, h, cls = bbox
                cx = (x_min + w / 2) / transformed_img.shape[1]
                cy = (y_min + h / 2) / transformed_img.shape[0]
                w = float(w) / transformed_img.shape[1]
                h = float(h) / transformed_img.shape[0]
                gt_bbox = [w, h]
                max_iou = -1
                anchor_index = -1
                for index, anchor in enumerate(self.anchors):
                    anchor_bbox = [anchor[0], anchor[1]]
                    iou = self.get_iou(gt_bbox, anchor_bbox)
                    if max_iou < iou:
                        anchor_index = index
                        max_iou = iou
                if max_iou != -1:
                    branch_index = (self.n_branch - 1) - anchor_index // self.n_branch
                    out_shape = self.output_shape_list[branch_index]
                    cell_w = 1 / out_shape[0]
                    cell_h = 1 / out_shape[1]
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 0] = 1.0
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 1] = float(cx) % cell_w
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 2] = float(cy) % cell_h
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 3] = np.log(float(w) / self.anchors[anchor_index][0])
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 4] = np.log(float(h) / self.anchors[anchor_index][1])
                    batch_gt[branch_index][:, int(float(cy) * out_shape[1]), int(float(cx) * out_shape[0]), (anchor_index % self.n_branch), 5 + int(cls)] = 1.0
            #         x1 = float(cx) - float(w) / 2
            #         y1 = float(cy) - float(h) / 2
            #         x2 = float(cx) + float(w) / 2
            #         y2 = float(cy) + float(h) / 2
            #         cv2.rectangle(transformed_img, (int(x1 * self.input_shape[1]), int(y1 * self.input_shape[0])), (int(x2 * self.input_shape[1]), int(y2 * self.input_shape[0])), (0, 0, 255), 1)
            # for index in range(self.n_branch):
            #     for a_index in range(self.n_anchor_per_branch):
            #         cv2.imshow(str(self.output_shape_list[index]) + "_anchor_" + str(a_index), cv2.resize(batch_gt[index][0, :, :, a_index, 0], (self.input_shape[1], self.input_shape[0])))
            #         cv2.moveWindow(str(self.output_shape_list[index]) + "_anchor_" + str(a_index), self.input_shape[1] * a_index, self.input_shape[0] * index)
            # cv2.imshow("test", cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
            # cv2.waitKey()
        return batch_img, batch_gt


if __name__ == "__main__":
    train_transform = albumentations.Compose([
        albumentations.Resize(height=352, width=352),
        albumentations.SomeOf([
            albumentations.RandomRotate90(p=1),
            albumentations.Sharpen(),
        ], 2, p=0.5),
        albumentations.SomeOf([
            albumentations.RandomBrightness(),
            albumentations.Affine(),
            albumentations.RandomContrast(),
            albumentations.Solarize(),
            albumentations.ColorJitter(),
        ], 3, p=0.5),
        albumentations.Flip(p=0.5),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    gen = Yolo_Generator(dataset_info_path=train_txt_path, batch_size=1, input_shape=(352, 352, 3),
                         output_shape_list=[(11, 11), (22, 22), (44, 44)], num_classes=NUM_CLASSES, anchors=ANCHORS,
                         augs=train_transform, is_train=True)
    for i in tqdm.tqdm(range(gen.__len__())):
        gen.__getitem__(i)
