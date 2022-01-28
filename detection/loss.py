import tensorflow as tf
from tensorflow.keras.losses import Loss

class Yolo_Loss(Loss):
    def __init__(self, batch_size, scale=1.0, n_anchor_per_branch=3, iou=0.5):
        super().__init__()
        self.batch_size = tf.cast(batch_size, tf.float32)
        self.scale = scale
        self.n_anchor_per_branch = n_anchor_per_branch
        self.iou = iou
        self.lamda_coord = 5
        self.lamda_noobj = 0.5

    def call(self, y_true, y_pred):
        object_mask = y_true[:, :, :, :, 0]
        no_object_mask = 1 - object_mask
        y_pred_shape = y_pred.shape
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[1], y_pred_shape[2], self.n_anchor_per_branch, int(y_pred_shape[3] / self.n_anchor_per_branch)])

        conf_loss = tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        no_conf_loss = tf.math.reduce_sum(no_object_mask * tf.losses.binary_crossentropy(y_true[..., 0:1], tf.sigmoid(y_pred[..., 0:1])))
        total_conf_loss = (self.lamda_coord * conf_loss + self.lamda_noobj * no_conf_loss) / self.batch_size

        xy_loss = self.scale * tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 1:3], tf.sigmoid(y_pred[..., 1:3])))
        wh_loss = self.scale * tf.math.reduce_sum(object_mask * tf.losses.MSE(y_true[..., 3:5], tf.sigmoid(y_pred[..., 3:5])))
        total_location_loss = (xy_loss + wh_loss) / self.batch_size

        total_cls_loss = tf.math.reduce_sum(object_mask * tf.losses.binary_crossentropy(y_true[..., 5:], tf.sigmoid(y_pred[..., 5:]))) / self.batch_size
        return total_conf_loss, total_location_loss, total_cls_loss