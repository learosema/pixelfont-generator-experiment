import tensorflow as tf
from tensorflow.keras import layers

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    block = tf.keras.Sequential()
    block.add(layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=not apply_batchnorm))

    if apply_batchnorm:
        block.add(layers.BatchNormalization())

    block.add(layers.LeakyReLU())

    return block

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))

    block.add(layers.BatchNormalization())

    if apply_dropout:
        block.add(layers.Dropout(0.5))

    block.add(layers.ReLU())
    return block

def Generator():
    inputs = layers.Input(shape=[256, 256, 1])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),   # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),   # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),   # (bs, 8, 8, 1024)
        upsample(512, 4),                       # (bs, 16, 16, 1024)
        upsample(256, 4),                       # (bs, 32, 32, 512)
        upsample(128, 4),                       # (bs, 64, 64, 256)
        upsample(64, 4),                        # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(1, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')  # (bs, 256, 256, 1)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
