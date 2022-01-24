import tensorflow as tf
from tensorflow.keras.losses import Loss

class Yolo_Loss(Loss):
    def __init__(self, iou=0.5):
        super().__init__()
        self.iou = iou

    def call(self, y_true, y_pred):
        total_loss = 0
        print(total_loss)
        return total_loss