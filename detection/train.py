import tensorflow as tf
import datetime
import os
import albumentations
import warnings
from detection.config import *
from network.backbone.regnet.regnet import RegNetY
from detection.generator import Yolo_Generator
from utils.utils import get_flops
from utils.logger import Logger
from utils.scheduler import CosineAnnealingLRScheduler
from network.common.blocks import StemBlock
from network.neck.neck import FPN
from network.head.head import Yolo_Head
from loss import Yolo_Loss
from keras_radam import RAdam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    backbone_name = "ResNet"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "./logs/fit/" + timestamp
    model_name = backbone_name + "_lr=" + str(LR) + "_wd=" + str(WEIGHT_DECAY) + "_batchsize=" + str(
        BATCH_SIZE)
    save_dir = "./saved_model/" + timestamp
    if os.path.isdir('./logs') == False:
        os.mkdir('./logs')
    if os.path.isdir('./logs/fit') == False:
        os.mkdir('./logs/fit')
    if os.path.isdir('./saved_model') == False:
        os.mkdir('./saved_model')
    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)

    lr_logger = Logger(log_dir + "/lr/" + model_name)
    activation = tf.keras.layers.Activation(tf.nn.relu)
    model = Detection_Model(INPUT_SHAPE,
                            NUM_CLASSES,
                            [1, 1, 1],
                            [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
                            [(3, 3), (3, 3), (3, 3)],
                            [(2, 2), (2, 2), (2, 2)],
                            [32, 32, 32],
                            "relu",
                            WEIGHT_DECAY)
    # model.load_weights("./saved_model/20220209-160037/ResNet_lr=0.001_wd=1e-05_batchsize=64_epoch=00009.h5")
    model.summary()
    # get_flops(model)
    output_shape_list = []
    for output in model.outputs:
        output_shape_list.append(output.get_shape()[1:3])
        print("output tensor = " + output.name + str(output.get_shape()))

    train_transform = albumentations.Compose([
        albumentations.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
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

    valid_transform = albumentations.Compose([
        albumentations.Resize(height=INPUT_SHAPE[0], width=INPUT_SHAPE[1]),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    train_batch_gen = Yolo_Generator(dataset_info_path=train_txt_path, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE, output_shape_list=output_shape_list, num_classes=NUM_CLASSES, anchors=ANCHORS, augs=train_transform, is_train=True)
    valid_batch_gen = Yolo_Generator(dataset_info_path=valid_txt_path, batch_size=max(1, int(BATCH_SIZE/8)), input_shape=INPUT_SHAPE, output_shape_list=output_shape_list, num_classes=NUM_CLASSES, anchors=ANCHORS, augs=valid_transform, is_train=False)

    model.compile(
        optimizer=RAdam(LR),
        loss={
            "Yolo_0": Yolo_Loss(scale=BRANCH_0_SCALE_XY),
            "Yolo_1": Yolo_Loss(scale=BRANCH_1_SCALE_XY),
            "Yolo_2": Yolo_Loss(scale=BRANCH_2_SCALE_XY),
        }
    )

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_dir + '/' + model_name + '_epoch={epoch:05d}.h5',
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    verbose=1,
                                                    period=1),
                 tf.keras.callbacks.TensorBoard(log_dir),
                 CosineAnnealingLRScheduler(total_epochs=EPOCHS, init_lr=LR, warmup_epoch=WARMUP_EPOCHS, n_cycles=4,
                                            lr_decay_rate=LR_DECAY_LATE, verbose=True, logger=lr_logger)
                 ]

    model.fit_generator(train_batch_gen,
                        use_multiprocessing=True,
                        max_queue_size=20,
                        callbacks=callbacks,
                        workers=4,
                        # initial_epoch=9,
                        epochs=EPOCHS,
                        validation_data=valid_batch_gen)
