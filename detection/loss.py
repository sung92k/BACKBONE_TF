import tensorflow as tf
from tensorflow.keras.losses import Loss

class Yolo_Loss(Loss):
    def __init__(self, scale=1.0, n_anchor_per_branch=3, iou=0.5):
        super().__init__()
        self.scale = scale
        self.n_anchor_per_branch = n_anchor_per_branch
        self.iou = iou
        self.lamda_coord = 5
        self.lamda_noobj = 0.5

    def call(self, y_true, y_pred):
        object_mask = y_true[:, :, :, :, 0]
        no_object_mask = 1 - object_mask
        b, h, w, c = y_pred.shape
        b_f = tf.cast(b, tf.float32)
        y_pred = tf.reshape(y_pred, [-1, h, w, self.n_anchor_per_branch, int(c / self.n_anchor_per_branch)])
        conf_loss = tf.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.math.sigmoid(y_pred[..., 0:1])))
        no_conf_loss = tf.reduce_sum(no_object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.math.sigmoid(y_pred[..., 0:1])))
        total_conf_loss = self.lamda_coord * conf_loss + self.lamda_noobj * no_conf_loss

        xy_loss = tf.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 1:3], tf.math.sigmoid(y_pred[..., 1:3])))
        wh_loss = tf.reduce_sum(object_mask * tf.losses.MSE(y_true[..., 3:5], (y_pred[..., 3:5])))
        total_location_loss = xy_loss + wh_loss

        total_cls_loss = tf.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 5:], tf.math.sigmoid(y_pred[..., 5:])))

        loss = (total_conf_loss + total_location_loss + total_cls_loss) / b_f
        return loss