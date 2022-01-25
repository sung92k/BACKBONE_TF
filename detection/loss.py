import tensorflow as tf
from tensorflow.keras.losses import Loss

class Yolo_Loss(Loss):
    def __init__(self, batch_size, scale=1.0, n_anchor_per_branch=3, iou=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.scale = scale
        self.n_anchor_per_branch = n_anchor_per_branch
        self.iou = iou
        self.lamda_coord = 5
        self.lamda_noobj = 0.5

    def call(self, y_true, y_pred):
        total_loss = 0
        object_mask = y_true[:, :, :, :, 0]
        no_object_mask = 1 - object_mask
        y_pred_shape = y_pred.shape
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[1], y_pred_shape[2], self.n_anchor_per_branch, int(y_pred_shape[3] / self.n_anchor_per_branch)])
        conf_loss = tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        no_conf_loss = tf.math.reduce_sum(no_object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        xy_loss = self.scale * tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 1:3], tf.sigmoid(y_pred[..., 1:3])))
        wh_loss = self.scale * tf.math.reduce_sum(object_mask * tf.losses.MSE(y_true[..., 3:5], tf.sigmoid(y_pred[..., 3:5])))
        cls_loss = tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 5:], tf.sigmoid(y_pred[..., 5:])))
        total_loss += (self.lamda_coord * conf_loss + self.lamda_noobj * no_conf_loss + xy_loss + wh_loss + cls_loss) / self.batch_size
        return total_loss