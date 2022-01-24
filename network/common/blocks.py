import tensorflow as tf
from tensorflow.keras.layers import *
from network.common.layers import Conv2dBnAct


def StemBlock(in_tensor, filter_list, kernel_size_list, strides_list, activation, weight_decay):
    '''
    :param in_tensor: Input tensor
    :param filter_list: Conv2d filter number list
    :param kernel_size: Conv2d filter size list
    :param strides_list: Conv2d strides list
    :param activation: Activation function
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    conv2d_1_out_tensor = Conv2dBnAct(in_tensor, filter_list[0], kernel_size_list[0], strides_list[0],
                                      activation=activation,
                                      weight_decay=weight_decay)

    conv2d_2_1_out_tensor = Conv2dBnAct(conv2d_1_out_tensor, filter_list[1] / 2, (1, 1), (1, 1), activation=activation,
                                        weight_decay=weight_decay)
    conv2d_2_2_out_tensor = Conv2dBnAct(conv2d_2_1_out_tensor, filter_list[1], kernel_size_list[1], strides_list[1],
                                        activation=activation, weight_decay=weight_decay)

    identity_out_tensor = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2d_1_out_tensor)

    concat_out_tensor = Concatenate()([conv2d_2_2_out_tensor, identity_out_tensor])
    conv2d_3_out_tensor = Conv2dBnAct(concat_out_tensor, filter_list[2], kernel_size_list[2], strides_list[2],
                                      activation=activation, weight_decay=weight_decay)
    return conv2d_3_out_tensor


def SEBlock(in_tensor, activation, weight_decay, reduction_ratio=4):
    '''
    :param in_tensor: Input tensor
    :param reduction_ratio: Dimensionlity-reduction - reduction ratio(r) value
    :param weight_decay: weight_decay
    :return: Tensor
    '''
    out_tensor = tf.keras.layers.GlobalAveragePooling2D()(in_tensor)
    out_tensor = tf.keras.layers.Reshape((1, 1, -1))(out_tensor)
    out_filters = out_tensor.shape[-1]
    out_tensor = Conv2dBnAct(out_tensor, max(1, int(out_filters / reduction_ratio)), (1, 1), (1, 1),
                             activation=activation, weight_decay=weight_decay)
    out_tensor = Conv2dBnAct(out_tensor, out_filters, (1, 1), (1, 1),
                             activation="sigmoid", weight_decay=weight_decay)
    out_tensor = in_tensor * out_tensor
    return out_tensor


if __name__ == "__main__":
    weight_decay = 1e-5
    input_shape = (416, 416, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    model = StemBlock(input_tensor, [32, 16, 32], [(3, 3), (3, 3), (1, 1)], [(2, 2), (2, 2), (1, 1)], "relu",
                      weight_decay)
    se_block = SEBlock(model, "relu", weight_decay, 2)
    se_block = tf.keras.Model(inputs=[input_tensor], outputs=se_block)
    se_block.summary()
    se_block.save("se_block.h5")
