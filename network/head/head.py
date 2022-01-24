import tensorflow as tf
from network.backbone.regnet.regnet import RegNetY
from network.neck.neck import FPN
from network.common.layers import Conv2dBnAct
from network.common.blocks import StemBlock

def Classification_Head(in_tensor, num_classes, weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor = tf.keras.layers.GlobalAveragePooling2D()(in_tensor)
    out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1),
                                           kernel_regularizer=kernel_regularizer)(out_tensor)
    out_tensor = tf.keras.layers.Dropout(0.2)(out_tensor)
    out_tensor = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
                                           kernel_regularizer=kernel_regularizer, activation='softmax')(out_tensor)
    out_tensor = tf.keras.layers.Reshape((-1,))(out_tensor)
    return out_tensor

def Yolo_Head(in_tensor_list, activation, num_classes, n_anchor, weight_decay):
    kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
    out_tensor_list = []
    for index, in_tensor in enumerate(in_tensor_list):
        out_tensor = in_tensor
        b, w, h, c = out_tensor.shape
        out_tensor = Conv2dBnAct(out_tensor, c, (3, 3), (1, 1), activation=activation, weight_decay=weight_decay)
        out_tensor = tf.keras.layers.Conv2D(n_anchor * (5 + num_classes), (1, 1), (1, 1), activation="sigmoid", kernel_regularizer=kernel_regularizer, name="Yolo_{}".format(index))(out_tensor)
        out_tensor_list.append(out_tensor)
    return out_tensor_list

if __name__ == "__main__":
    input_tensor = tf.keras.layers.Input((416, 416, 3))
    stem = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                         1e-5)
    backbone = RegNetY(
        stem,
        [1, 1, 1],
        [[128, 128, 128], [256, 256, 256], [512, 512, 512]],
        [(3, 3), (3, 3), (3, 3)],
        [(2, 2), (2, 2), (2, 2)],
        [32, 32, 32],
        "relu",
        1e-5
    )
    fpn = FPN(backbone, "relu", 1e-5, "add")
    head = Yolo_Head(fpn, "relu", 80, 3, 1e-5)
    model = tf.keras.Model(inputs=[input_tensor], outputs=head)
    model.summary()
    model.save("test.h5")
