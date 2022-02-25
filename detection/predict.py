import glob
import cv2
import tqdm
import tensorflow as tf
import numpy as np
from network.common.blocks import StemBlock
from network.neck.neck import FPN
from network.head.head import Yolo_Head
from network.backbone.regnet.regnet import RegNetY
from config import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Detection_Model(in_shape, num_classes, n_block_per_stage, filter_per_stage, kernel_size_per_stage,
                    strides_per_stage, groups_per_stage, activation,
                    weight_decay):
    input_tensor = tf.keras.layers.Input(in_shape)
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], activation,
                         weight_decay)
    backbone = RegNetY(
        stem,
        n_block_per_stage,
        filter_per_stage,
        kernel_size_per_stage,
        strides_per_stage,
        groups_per_stage,
        activation,
        weight_decay
    )
    fpn = FPN(backbone, activation, weight_decay, "add")
    head = Yolo_Head(fpn, activation, num_classes, int(len(ANCHORS) / len(fpn)), weight_decay)
    model = tf.keras.Model(inputs=[input_tensor], outputs=head)
    return model

if __name__ == '__main__':
    conf_thres = 0.5
    video_dir = "C:/Users/sangmin/Desktop/BACKBONE_TF/videos"
    video_list = glob.glob(video_dir + "/**/*", recursive=True)
    scale_list = [BRANCH_0_SCALE_XY, BRANCH_1_SCALE_XY, BRANCH_2_SCALE_XY]
    model = Detection_Model(INPUT_SHAPE,
                            NUM_CLASSES,
                            [1, 1, 1],
                            [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
                            [(3, 3), (3, 3), (3, 3)],
                            [(2, 2), (2, 2), (2, 2)],
                            [32, 32, 32],
                            "relu",
                            WEIGHT_DECAY)
    model.load_weights("./saved_model/ResNet_lr=0.001_wd=1e-05_batchsize=64_epoch=00056.h5")
    # model.summary()
    for video in tqdm.tqdm(video_list):
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if ret:
                resize_img = cv2.resize(frame, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
                yolo_out_list = model.predict(np.expand_dims(cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB).astype(np.float) / 255., axis=0))
                for b_index, yolo_out in enumerate(yolo_out_list):
                    for a_index in range(3):
                        yolo_out = np.reshape(yolo_out, [-1, yolo_out.shape[1], yolo_out.shape[2], 3, 16])
                        yolo_out = np.where(yolo_out > conf_thres, 1.0, 0.0)
                        cv2.imshow(str(yolo_out.shape) + "_anchor_" + str(a_index), cv2.resize(sigmoid(yolo_out[0, :, :, a_index, 7]) * sigmoid(yolo_out[0, :, :, a_index, 7]), (480, 270)))
                        cv2.moveWindow(str(yolo_out.shape) + "_anchor_" + str(a_index), 480 * a_index, 270 * b_index)
                resize_img = cv2.resize(frame, (480, 270))
                cv2.imshow("test", resize_img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            else:
                break