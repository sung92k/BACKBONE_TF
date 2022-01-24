import tensorflow as tf


def binary_accuracy(y_true, y_pred):
    metric = tf.mean(tf.equal(y_true, tf.round(y_pred)), axis=-1)
    return metric


def categorical_accuracy(y_true, y_pred):
    metric = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.keras.backend.floatx())
    return metric


def sparse_categorical_accuracy(y_true, y_pred):
    metric = tf.cast(
        tf.equal(tf.math.max(y_true, axis=-1), tf.cast(tf.argmax(y_pred, axis=-1), tf.keras.backend.floatx())),
        tf.keras.backend.floatx())
    return metric


def top_1_categorical_accuracy(y_true, y_pred, k=1):
    metric = tf.cast(tf.compat.v1.math.in_top_k(
        y_pred, tf.compat.v1.argmax(y_true, axis=-1), k), tf.keras.backend.floatx())
    return metric


def top_5_categorical_accuracy(y_true, y_pred, k=5):
    metric = tf.cast(tf.compat.v1.math.in_top_k(
        y_pred, tf.compat.v1.argmax(y_true, axis=-1), k), tf.keras.backend.floatx())
    return metric


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    metric = tf.math.in_top_k(y_pred, tf.cast(tf.max(y_true, axis=-1), 'int32'), k)
    return metric


if __name__ == "__main__":
    pred_tensor = tf.constant(
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0], ]
    )

    gt_tensor = tf.constant([
        [[0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1], ]
    ])

    metric = categorical_accuracy(pred_tensor, gt_tensor)
