import tensorflow as tf
from tensorflow.keras.losses import Loss

class Yolo_Loss(Loss):
    def __init__(self, scale_xy, n_anchor_per_branch=3, iou=0.5):
        super().__init__()
        self.scale_xy = scale_xy
        self.n_anchor_per_branch = n_anchor_per_branch
        self.iou = iou

    def call(self, y_true, y_pred):
        total_loss = 0
        object_mask = y_true[:, :, :, :, 0]
        no_object_mask = tf.where(object_mask == 1, 0.0, 1.0)
        y_pred_shape = y_pred.shape
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[1], y_pred_shape[2], self.n_anchor_per_branch, int(y_pred_shape[3] / self.n_anchor_per_branch)])
        conf_loss = tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        no_conf_loss = tf.math.reduce_sum(no_object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        total_loss += conf_loss + no_conf_loss
        return total_loss