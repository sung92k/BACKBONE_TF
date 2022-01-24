import os
import tensorflow as tf
from TF.fs_backbone.regnet.regnet import RegNetX
from TF.fs_backbone.resnet.block import ResidualBlock
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze_ckpt(model, save_path):
    filename = os.path.basename(save_path).split(".")[0]
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 60)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 60)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=save_path,
                      name=f"{filename}.pb",
                      as_text=False)
    # Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=save_path,
                      name=f"{filename}.pbtxt",
                      as_text=True)


if __name__ == '__main__':
    input_shape = (576, 576, 3)
    input_tensor = tf.keras.layers.Input(input_shape)
    activation = tf.keras.layers.Activation(tf.nn.relu)
    model = RegNetX(activation=activation)
    # model = ResidualBlock(activation, (32, 32, 64), 1)
    model(input_tensor)
    ckpt_path = "../fs_backbone/regnet/regnet"
    pb_path = "../fs_backbone/regnet/regnet.pb"
    model.load_weights(ckpt_path)
    model.summary()
    freeze_ckpt(model, pb_path)
