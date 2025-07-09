import tensorflow as tf
from tensorflow.keras import layers

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[256, 256, 1], name='input_image')
    tar = layers.Input(shape=[256, 256, 1], name='target_image')

    x = layers.Concatenate()([inp, tar])  # (bs, 256, 256, 2)

    down1 = downsample(64, 4, False)(x)     # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)       # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)       # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)       # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm = layers.BatchNormalization()(conv)
    leaky = layers.LeakyReLU()(batchnorm)

    zero_pad2 = layers.ZeroPadding2D()(leaky)        # (bs, 33, 33, 512)
    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Helper: same downsample block from generator
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=not apply_batchnorm))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result