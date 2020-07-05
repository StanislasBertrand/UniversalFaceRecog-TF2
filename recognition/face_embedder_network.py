import tensorflow as tf
import numpy as np


def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class FaceEmbedder(object):
    def __init__(self, weights_file):
        weights_dict = load_weights(weights_file)
        self.model = self.load_model(weights_dict)

    def load_model(self, weights_dict = None):
        """
            generate tf.keras model and load weights from dict
        Parameters:
        ----------
            weights_dict : map of layer name -> list of np arrays of weights
        Returns:
        -------
            model : tf.keras.Model with loaded weights
        """
        data            = tf.keras.Input(dtype=tf.float32, shape = (112, 112, 3), name = 'data')
        minusscalar0    = tf.keras.layers.Lambda(lambda x: x - weights_dict['minusscalar0_second']['value'][0])(data)
        mulscalar0      = tf.keras.layers.Lambda(lambda x: x*weights_dict['mulscalar0_second']['value'][0])(minusscalar0)
        conv0_pad       = FaceEmbedder.pad(mulscalar0)
        conv0           = FaceEmbedder.convolution(conv0_pad, weights_dict,  strides=[1, 1], padding='VALID', name='conv0')
        bn0             = FaceEmbedder.batch_normalization(conv0, variance_epsilon=1.9999999494757503e-05, name='bn0')
        relu0           = FaceEmbedder.prelu(bn0, name='relu0')
        stage1_unit1_bn1 = FaceEmbedder.batch_normalization(relu0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1')
        stage1_unit1_conv1sc = FaceEmbedder.convolution(relu0, weights_dict, strides=[2, 2], padding='VALID', name='stage1_unit1_conv1sc')
        stage1_unit1_conv1_pad = FaceEmbedder.pad(stage1_unit1_bn1)
        stage1_unit1_conv1 = FaceEmbedder.convolution(stage1_unit1_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit1_conv1')
        stage1_unit1_sc = FaceEmbedder.batch_normalization(stage1_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_sc')
        stage1_unit1_bn2 = FaceEmbedder.batch_normalization(stage1_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn2')
        stage1_unit1_relu1 = FaceEmbedder.prelu(stage1_unit1_bn2, name='stage1_unit1_relu1')
        stage1_unit1_conv2_pad = FaceEmbedder.pad(stage1_unit1_relu1)
        stage1_unit1_conv2 = FaceEmbedder.convolution(stage1_unit1_conv2_pad, weights_dict, strides=[2, 2], padding='VALID', name='stage1_unit1_conv2')
        stage1_unit1_bn3 = FaceEmbedder.batch_normalization(stage1_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn3')
        plus0           = stage1_unit1_bn3 + stage1_unit1_sc
        stage1_unit2_bn1 = FaceEmbedder.batch_normalization(plus0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1')
        stage1_unit2_conv1_pad = FaceEmbedder.pad(stage1_unit2_bn1)
        stage1_unit2_conv1 = FaceEmbedder.convolution(stage1_unit2_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit2_conv1')
        stage1_unit2_bn2 = FaceEmbedder.batch_normalization(stage1_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn2')
        stage1_unit2_relu1 = FaceEmbedder.prelu(stage1_unit2_bn2, name='stage1_unit2_relu1')
        stage1_unit2_conv2_pad = FaceEmbedder.pad(stage1_unit2_relu1)
        stage1_unit2_conv2 = FaceEmbedder.convolution(stage1_unit2_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit2_conv2')
        stage1_unit2_bn3 = FaceEmbedder.batch_normalization(stage1_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn3')
        plus1           = stage1_unit2_bn3 + plus0
        stage1_unit3_bn1 = FaceEmbedder.batch_normalization(plus1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1')
        stage1_unit3_conv1_pad = FaceEmbedder.pad(stage1_unit3_bn1)
        stage1_unit3_conv1 = FaceEmbedder.convolution(stage1_unit3_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit3_conv1')
        stage1_unit3_bn2 = FaceEmbedder.batch_normalization(stage1_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn2')
        stage1_unit3_relu1 = FaceEmbedder.prelu(stage1_unit3_bn2, name='stage1_unit3_relu1')
        stage1_unit3_conv2_pad = FaceEmbedder.pad(stage1_unit3_relu1)
        stage1_unit3_conv2 = FaceEmbedder.convolution(stage1_unit3_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit3_conv2')
        stage1_unit3_bn3 = FaceEmbedder.batch_normalization(stage1_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn3')
        plus2           = stage1_unit3_bn3 + plus1
        stage2_unit1_bn1 = FaceEmbedder.batch_normalization(plus2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1')
        stage2_unit1_conv1sc = FaceEmbedder.convolution(plus2, weights_dict, strides=[2, 2], padding='VALID', name='stage2_unit1_conv1sc')
        stage2_unit1_conv1_pad = FaceEmbedder.pad(stage2_unit1_bn1)
        stage2_unit1_conv1 = FaceEmbedder.convolution(stage2_unit1_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit1_conv1')
        stage2_unit1_sc = FaceEmbedder.batch_normalization(stage2_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_sc')
        stage2_unit1_bn2 = FaceEmbedder.batch_normalization(stage2_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn2')
        stage2_unit1_relu1 = FaceEmbedder.prelu(stage2_unit1_bn2, name='stage2_unit1_relu1')
        stage2_unit1_conv2_pad = FaceEmbedder.pad(stage2_unit1_relu1)
        stage2_unit1_conv2 = FaceEmbedder.convolution(stage2_unit1_conv2_pad, weights_dict, strides=[2, 2], padding='VALID', name='stage2_unit1_conv2')
        stage2_unit1_bn3 = FaceEmbedder.batch_normalization(stage2_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn3')
        plus3           = stage2_unit1_bn3 + stage2_unit1_sc
        stage2_unit2_bn1 = FaceEmbedder.batch_normalization(plus3, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1')
        stage2_unit2_conv1_pad = FaceEmbedder.pad(stage2_unit2_bn1)
        stage2_unit2_conv1 = FaceEmbedder.convolution(stage2_unit2_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit2_conv1')
        stage2_unit2_bn2 = FaceEmbedder.batch_normalization(stage2_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn2')
        stage2_unit2_relu1 = FaceEmbedder.prelu(stage2_unit2_bn2, name='stage2_unit2_relu1')
        stage2_unit2_conv2_pad = FaceEmbedder.pad(stage2_unit2_relu1)
        stage2_unit2_conv2 = FaceEmbedder.convolution(stage2_unit2_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit2_conv2')
        stage2_unit2_bn3 = FaceEmbedder.batch_normalization(stage2_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn3')
        plus4           = stage2_unit2_bn3 + plus3
        stage2_unit3_bn1 = FaceEmbedder.batch_normalization(plus4, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1')
        stage2_unit3_conv1_pad = FaceEmbedder.pad(stage2_unit3_bn1)
        stage2_unit3_conv1 = FaceEmbedder.convolution(stage2_unit3_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit3_conv1')
        stage2_unit3_bn2 = FaceEmbedder.batch_normalization(stage2_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn2')
        stage2_unit3_relu1 = FaceEmbedder.prelu(stage2_unit3_bn2, name='stage2_unit3_relu1')
        stage2_unit3_conv2_pad = FaceEmbedder.pad(stage2_unit3_relu1)
        stage2_unit3_conv2 = FaceEmbedder.convolution(stage2_unit3_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit3_conv2')
        stage2_unit3_bn3 = FaceEmbedder.batch_normalization(stage2_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn3')
        plus5           = stage2_unit3_bn3 + plus4
        stage2_unit4_bn1 = FaceEmbedder.batch_normalization(plus5, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1')
        stage2_unit4_conv1_pad = FaceEmbedder.pad(stage2_unit4_bn1)
        stage2_unit4_conv1 = FaceEmbedder.convolution(stage2_unit4_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit4_conv1')
        stage2_unit4_bn2 = FaceEmbedder.batch_normalization(stage2_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn2')
        stage2_unit4_relu1 = FaceEmbedder.prelu(stage2_unit4_bn2, name='stage2_unit4_relu1')
        stage2_unit4_conv2_pad = FaceEmbedder.pad(stage2_unit4_relu1)
        stage2_unit4_conv2 = FaceEmbedder.convolution(stage2_unit4_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage2_unit4_conv2')
        stage2_unit4_bn3 = FaceEmbedder.batch_normalization(stage2_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn3')
        plus6           = stage2_unit4_bn3 + plus5
        stage3_unit1_bn1 = FaceEmbedder.batch_normalization(plus6, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1')
        stage3_unit1_conv1sc = FaceEmbedder.convolution(plus6, weights_dict, strides=[2, 2], padding='VALID', name='stage3_unit1_conv1sc')
        stage3_unit1_conv1_pad = FaceEmbedder.pad(stage3_unit1_bn1)
        stage3_unit1_conv1 = FaceEmbedder.convolution(stage3_unit1_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit1_conv1')
        stage3_unit1_sc = FaceEmbedder.batch_normalization(stage3_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_sc')
        stage3_unit1_bn2 = FaceEmbedder.batch_normalization(stage3_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn2')
        stage3_unit1_relu1 = FaceEmbedder.prelu(stage3_unit1_bn2, name='stage3_unit1_relu1')
        stage3_unit1_conv2_pad = FaceEmbedder.pad(stage3_unit1_relu1)
        stage3_unit1_conv2 = FaceEmbedder.convolution(stage3_unit1_conv2_pad, weights_dict, strides=[2, 2], padding='VALID', name='stage3_unit1_conv2')
        stage3_unit1_bn3 = FaceEmbedder.batch_normalization(stage3_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn3')
        plus7           = stage3_unit1_bn3 + stage3_unit1_sc
        stage3_unit2_bn1 = FaceEmbedder.batch_normalization(plus7, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1')
        stage3_unit2_conv1_pad = FaceEmbedder.pad(stage3_unit2_bn1)
        stage3_unit2_conv1 = FaceEmbedder.convolution(stage3_unit2_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit2_conv1')
        stage3_unit2_bn2 = FaceEmbedder.batch_normalization(stage3_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn2')
        stage3_unit2_relu1 = FaceEmbedder.prelu(stage3_unit2_bn2, name='stage3_unit2_relu1')
        stage3_unit2_conv2_pad = FaceEmbedder.pad(stage3_unit2_relu1)
        stage3_unit2_conv2 = FaceEmbedder.convolution(stage3_unit2_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit2_conv2')
        stage3_unit2_bn3 = FaceEmbedder.batch_normalization(stage3_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn3')
        plus8           = stage3_unit2_bn3 + plus7
        stage3_unit3_bn1 = FaceEmbedder.batch_normalization(plus8, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1')
        stage3_unit3_conv1_pad = FaceEmbedder.pad(stage3_unit3_bn1)
        stage3_unit3_conv1 = FaceEmbedder.convolution(stage3_unit3_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit3_conv1')
        stage3_unit3_bn2 = FaceEmbedder.batch_normalization(stage3_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn2')
        stage3_unit3_relu1 = FaceEmbedder.prelu(stage3_unit3_bn2, name='stage3_unit3_relu1')
        stage3_unit3_conv2_pad = FaceEmbedder.pad(stage3_unit3_relu1)
        stage3_unit3_conv2 = FaceEmbedder.convolution(stage3_unit3_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit3_conv2')
        stage3_unit3_bn3 = FaceEmbedder.batch_normalization(stage3_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn3')
        plus9           = stage3_unit3_bn3 + plus8
        stage3_unit4_bn1 = FaceEmbedder.batch_normalization(plus9, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1')
        stage3_unit4_conv1_pad = FaceEmbedder.pad(stage3_unit4_bn1)
        stage3_unit4_conv1 = FaceEmbedder.convolution(stage3_unit4_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit4_conv1')
        stage3_unit4_bn2 = FaceEmbedder.batch_normalization(stage3_unit4_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn2')
        stage3_unit4_relu1 = FaceEmbedder.prelu(stage3_unit4_bn2, name='stage3_unit4_relu1')
        stage3_unit4_conv2_pad = FaceEmbedder.pad(stage3_unit4_relu1)
        stage3_unit4_conv2 = FaceEmbedder.convolution(stage3_unit4_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit4_conv2')
        stage3_unit4_bn3 = FaceEmbedder.batch_normalization(stage3_unit4_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn3')
        plus10          = stage3_unit4_bn3 + plus9
        stage3_unit5_bn1 = FaceEmbedder.batch_normalization(plus10, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1')
        stage3_unit5_conv1_pad = FaceEmbedder.pad(stage3_unit5_bn1)
        stage3_unit5_conv1 = FaceEmbedder.convolution(stage3_unit5_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit5_conv1')
        stage3_unit5_bn2 = FaceEmbedder.batch_normalization(stage3_unit5_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn2')
        stage3_unit5_relu1 = FaceEmbedder.prelu(stage3_unit5_bn2, name='stage3_unit5_relu1')
        stage3_unit5_conv2_pad = FaceEmbedder.pad(stage3_unit5_relu1)
        stage3_unit5_conv2 = FaceEmbedder.convolution(stage3_unit5_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit5_conv2')
        stage3_unit5_bn3 = FaceEmbedder.batch_normalization(stage3_unit5_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn3')
        plus11          = stage3_unit5_bn3 + plus10
        stage3_unit6_bn1 = FaceEmbedder.batch_normalization(plus11, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1')
        stage3_unit6_conv1_pad = FaceEmbedder.pad(stage3_unit6_bn1)
        stage3_unit6_conv1 = FaceEmbedder.convolution(stage3_unit6_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit6_conv1')
        stage3_unit6_bn2 = FaceEmbedder.batch_normalization(stage3_unit6_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn2')
        stage3_unit6_relu1 = FaceEmbedder.prelu(stage3_unit6_bn2, name='stage3_unit6_relu1')
        stage3_unit6_conv2_pad = FaceEmbedder.pad(stage3_unit6_relu1)
        stage3_unit6_conv2 = FaceEmbedder.convolution(stage3_unit6_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit6_conv2')
        stage3_unit6_bn3 = FaceEmbedder.batch_normalization(stage3_unit6_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn3')
        plus12          = stage3_unit6_bn3 + plus11
        stage3_unit7_bn1 = FaceEmbedder.batch_normalization(plus12, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn1')
        stage3_unit7_conv1_pad = FaceEmbedder.pad(stage3_unit7_bn1)
        stage3_unit7_conv1 = FaceEmbedder.convolution(stage3_unit7_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit7_conv1')
        stage3_unit7_bn2 = FaceEmbedder.batch_normalization(stage3_unit7_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn2')
        stage3_unit7_relu1 = FaceEmbedder.prelu(stage3_unit7_bn2, name='stage3_unit7_relu1')
        stage3_unit7_conv2_pad = FaceEmbedder.pad(stage3_unit7_relu1)
        stage3_unit7_conv2 = FaceEmbedder.convolution(stage3_unit7_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit7_conv2')
        stage3_unit7_bn3 = FaceEmbedder.batch_normalization(stage3_unit7_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit7_bn3')
        plus13          = stage3_unit7_bn3 + plus12
        stage3_unit8_bn1 = FaceEmbedder.batch_normalization(plus13, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn1')
        stage3_unit8_conv1_pad = FaceEmbedder.pad(stage3_unit8_bn1)
        stage3_unit8_conv1 = FaceEmbedder.convolution(stage3_unit8_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit8_conv1')
        stage3_unit8_bn2 = FaceEmbedder.batch_normalization(stage3_unit8_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn2')
        stage3_unit8_relu1 = FaceEmbedder.prelu(stage3_unit8_bn2, name='stage3_unit8_relu1')
        stage3_unit8_conv2_pad = FaceEmbedder.pad(stage3_unit8_relu1)
        stage3_unit8_conv2 = FaceEmbedder.convolution(stage3_unit8_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit8_conv2')
        stage3_unit8_bn3 = FaceEmbedder.batch_normalization(stage3_unit8_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit8_bn3')
        plus14          = stage3_unit8_bn3 + plus13
        stage3_unit9_bn1 = FaceEmbedder.batch_normalization(plus14, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn1')
        stage3_unit9_conv1_pad = FaceEmbedder.pad(stage3_unit9_bn1)
        stage3_unit9_conv1 = FaceEmbedder.convolution(stage3_unit9_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit9_conv1')
        stage3_unit9_bn2 = FaceEmbedder.batch_normalization(stage3_unit9_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn2')
        stage3_unit9_relu1 = FaceEmbedder.prelu(stage3_unit9_bn2, name='stage3_unit9_relu1')
        stage3_unit9_conv2_pad = FaceEmbedder.pad(stage3_unit9_relu1)
        stage3_unit9_conv2 = FaceEmbedder.convolution(stage3_unit9_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit9_conv2')
        stage3_unit9_bn3 = FaceEmbedder.batch_normalization(stage3_unit9_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit9_bn3')
        plus15          = stage3_unit9_bn3 + plus14
        stage3_unit10_bn1 = FaceEmbedder.batch_normalization(plus15, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn1')
        stage3_unit10_conv1_pad = FaceEmbedder.pad(stage3_unit10_bn1)
        stage3_unit10_conv1 = FaceEmbedder.convolution(stage3_unit10_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit10_conv1')
        stage3_unit10_bn2 = FaceEmbedder.batch_normalization(stage3_unit10_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn2')
        stage3_unit10_relu1 = FaceEmbedder.prelu(stage3_unit10_bn2, name='stage3_unit10_relu1')
        stage3_unit10_conv2_pad = FaceEmbedder.pad(stage3_unit10_relu1)
        stage3_unit10_conv2 = FaceEmbedder.convolution(stage3_unit10_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit10_conv2')
        stage3_unit10_bn3 = FaceEmbedder.batch_normalization(stage3_unit10_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit10_bn3')
        plus16          = stage3_unit10_bn3 + plus15
        stage3_unit11_bn1 = FaceEmbedder.batch_normalization(plus16, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn1')
        stage3_unit11_conv1_pad = FaceEmbedder.pad(stage3_unit11_bn1)
        stage3_unit11_conv1 = FaceEmbedder.convolution(stage3_unit11_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit11_conv1')
        stage3_unit11_bn2 = FaceEmbedder.batch_normalization(stage3_unit11_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn2')
        stage3_unit11_relu1 = FaceEmbedder.prelu(stage3_unit11_bn2, name='stage3_unit11_relu1')
        stage3_unit11_conv2_pad = FaceEmbedder.pad(stage3_unit11_relu1)
        stage3_unit11_conv2 = FaceEmbedder.convolution(stage3_unit11_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit11_conv2')
        stage3_unit11_bn3 = FaceEmbedder.batch_normalization(stage3_unit11_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit11_bn3')
        plus17          = stage3_unit11_bn3 + plus16
        stage3_unit12_bn1 = FaceEmbedder.batch_normalization(plus17, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn1')
        stage3_unit12_conv1_pad = FaceEmbedder.pad(stage3_unit12_bn1)
        stage3_unit12_conv1 = FaceEmbedder.convolution(stage3_unit12_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit12_conv1')
        stage3_unit12_bn2 = FaceEmbedder.batch_normalization(stage3_unit12_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn2')
        stage3_unit12_relu1 = FaceEmbedder.prelu(stage3_unit12_bn2, name='stage3_unit12_relu1')
        stage3_unit12_conv2_pad = FaceEmbedder.pad(stage3_unit12_relu1)
        stage3_unit12_conv2 = FaceEmbedder.convolution(stage3_unit12_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit12_conv2')
        stage3_unit12_bn3 = FaceEmbedder.batch_normalization(stage3_unit12_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit12_bn3')
        plus18          = stage3_unit12_bn3 + plus17
        stage3_unit13_bn1 = FaceEmbedder.batch_normalization(plus18, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn1')
        stage3_unit13_conv1_pad = FaceEmbedder.pad(stage3_unit13_bn1)
        stage3_unit13_conv1 = FaceEmbedder.convolution(stage3_unit13_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit13_conv1')
        stage3_unit13_bn2 = FaceEmbedder.batch_normalization(stage3_unit13_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn2')
        stage3_unit13_relu1 = FaceEmbedder.prelu(stage3_unit13_bn2, name='stage3_unit13_relu1')
        stage3_unit13_conv2_pad = FaceEmbedder.pad(stage3_unit13_relu1)
        stage3_unit13_conv2 = FaceEmbedder.convolution(stage3_unit13_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit13_conv2')
        stage3_unit13_bn3 = FaceEmbedder.batch_normalization(stage3_unit13_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit13_bn3')
        plus19          = stage3_unit13_bn3 + plus18
        stage3_unit14_bn1 = FaceEmbedder.batch_normalization(plus19, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn1')
        stage3_unit14_conv1_pad = FaceEmbedder.pad(stage3_unit14_bn1)
        stage3_unit14_conv1 = FaceEmbedder.convolution(stage3_unit14_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit14_conv1')
        stage3_unit14_bn2 = FaceEmbedder.batch_normalization(stage3_unit14_conv1, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn2')
        stage3_unit14_relu1 = FaceEmbedder.prelu(stage3_unit14_bn2, name='stage3_unit14_relu1')
        stage3_unit14_conv2_pad = FaceEmbedder.pad(stage3_unit14_relu1)
        stage3_unit14_conv2 = FaceEmbedder.convolution(stage3_unit14_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage3_unit14_conv2')
        stage3_unit14_bn3 = FaceEmbedder.batch_normalization(stage3_unit14_conv2, variance_epsilon=1.9999999494757503e-05, name='stage3_unit14_bn3')
        plus20          = stage3_unit14_bn3 + plus19
        stage4_unit1_bn1 = FaceEmbedder.batch_normalization(plus20, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1')
        stage4_unit1_conv1sc = FaceEmbedder.convolution(plus20, weights_dict, strides=[2, 2], padding='VALID', name='stage4_unit1_conv1sc')
        stage4_unit1_conv1_pad = FaceEmbedder.pad(stage4_unit1_bn1)
        stage4_unit1_conv1 = FaceEmbedder.convolution(stage4_unit1_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage4_unit1_conv1')
        stage4_unit1_sc = FaceEmbedder.batch_normalization(stage4_unit1_conv1sc, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_sc')
        stage4_unit1_bn2 = FaceEmbedder.batch_normalization(stage4_unit1_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn2')
        stage4_unit1_relu1 = FaceEmbedder.prelu(stage4_unit1_bn2, name='stage4_unit1_relu1')
        stage4_unit1_conv2_pad = FaceEmbedder.pad(stage4_unit1_relu1)
        stage4_unit1_conv2 = FaceEmbedder.convolution(stage4_unit1_conv2_pad, weights_dict, strides=[2, 2], padding='VALID', name='stage4_unit1_conv2')
        stage4_unit1_bn3 = FaceEmbedder.batch_normalization(stage4_unit1_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn3')
        plus21          = stage4_unit1_bn3 + stage4_unit1_sc
        stage4_unit2_bn1 = FaceEmbedder.batch_normalization(plus21, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1')
        stage4_unit2_conv1_pad = FaceEmbedder.pad(stage4_unit2_bn1)
        stage4_unit2_conv1 = FaceEmbedder.convolution(stage4_unit2_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage4_unit2_conv1')
        stage4_unit2_bn2 = FaceEmbedder.batch_normalization(stage4_unit2_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn2')
        stage4_unit2_relu1 = FaceEmbedder.prelu(stage4_unit2_bn2, name='stage4_unit2_relu1')
        stage4_unit2_conv2_pad = FaceEmbedder.pad(stage4_unit2_relu1)
        stage4_unit2_conv2 = FaceEmbedder.convolution(stage4_unit2_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage4_unit2_conv2')
        stage4_unit2_bn3 = FaceEmbedder.batch_normalization(stage4_unit2_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn3')
        plus22          = stage4_unit2_bn3 + plus21
        stage4_unit3_bn1 = FaceEmbedder.batch_normalization(plus22, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1')
        stage4_unit3_conv1_pad = FaceEmbedder.pad(stage4_unit3_bn1)
        stage4_unit3_conv1 = FaceEmbedder.convolution(stage4_unit3_conv1_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage4_unit3_conv1')
        stage4_unit3_bn2 = FaceEmbedder.batch_normalization(stage4_unit3_conv1, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn2')
        stage4_unit3_relu1 = FaceEmbedder.prelu(stage4_unit3_bn2, name='stage4_unit3_relu1')
        stage4_unit3_conv2_pad = FaceEmbedder.pad(stage4_unit3_relu1)
        stage4_unit3_conv2 = FaceEmbedder.convolution(stage4_unit3_conv2_pad, weights_dict, strides=[1, 1], padding='VALID', name='stage4_unit3_conv2')
        stage4_unit3_bn3 = FaceEmbedder.batch_normalization(stage4_unit3_conv2, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn3')
        plus23          = stage4_unit3_bn3 + plus22
        bn1             = FaceEmbedder.batch_normalization(plus23, variance_epsilon=1.9999999494757503e-05, name='bn1')
        pre_fc1_flatten = tf.keras.layers.Flatten()(bn1)
        pre_fc1         = FaceEmbedder.dense(pre_fc1_flatten, 512, kernel_initializer = tf.compat.v1.constant_initializer(weights_dict['pre_fc1']['weights']), bias_initializer = tf.compat.v1.constant_initializer(weights_dict['pre_fc1']['bias']), use_bias = True)
        fc1             = FaceEmbedder.batch_normalization(pre_fc1, variance_epsilon=1.9999999494757503e-05, name='fc1')

        model = tf.keras.Model(inputs = data, outputs = fc1)

        weights_names = list(weights_dict.keys())
        weights_conv_names = ["conv", "stage1_unit1_conv1sc"]
        weights_bn_names = ["bn", "stage1_unit1_sc", "stage2_unit1_sc", "stage3_unit1_sc", "stage4_unit1_sc", "fc1"]
        weights_relu_names = ["relu"]

        for layer in model.layers:
            if layer.name in weights_names:
                if any(x in layer.name for x in weights_conv_names):
                    layer.set_weights([weights_dict[layer.name]["weights"]])
                if any(x in layer.name for x in weights_bn_names):
                    mean = weights_dict[layer.name]["mean"]
                    var = weights_dict[layer.name]["var"]
                    scale = weights_dict[layer.name]["scale"] if "scale" in weights_dict[layer.name] else np.ones(weights_dict[layer.name]["mean"].shape)
                    bias = weights_dict[layer.name]["bias"] if "bias" in weights_dict[layer.name] else np.zeros(weights_dict[layer.name]["mean"].shape)
                    layer.set_weights([scale, bias, mean, var])
                if any(x in layer.name for x in weights_relu_names):
                    weights = weights_dict[layer.name]["gamma"]
                    expected_shape = layer.get_weights()[0].shape
                    weights_1_extra_dim = np.repeat(weights[np.newaxis,:], expected_shape[1], axis = 0)
                    weights_2_extra_dims = np.repeat(weights_1_extra_dim[np.newaxis,:], expected_shape[0], axis = 0)
                    layer.set_weights([weights_2_extra_dims])

        return model

    @staticmethod
    def dense(input, units, kernel_initializer, bias_initializer, use_bias):
        return tf.keras.layers.Dense(units, kernel_initializer = kernel_initializer, bias_initializer = bias_initializer, use_bias = use_bias)(input)
        
    @staticmethod
    def prelu(input, name):
        return tf.keras.layers.PReLU(name = name)(input)

    @staticmethod
    def batch_normalization(input, variance_epsilon, name):
        return tf.keras.layers.BatchNormalization(epsilon = variance_epsilon, name = name, trainable = False)(input)

    @staticmethod
    def convolution(input, weights, strides, padding, name):
        weights = weights[name]['weights']
        layer = tf.keras.layers.Conv2D(filters = weights.shape[3], kernel_size = weights.shape[0:2],name=name, strides = strides, padding = padding, use_bias = False)(input)
        return layer

    @staticmethod
    def pad(input):
        return tf.keras.layers.ZeroPadding2D(padding=(1, 1))(input)
