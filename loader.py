import tensorflow as tf
import os

def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_png(image, channels=1)  # Grayscale, 512x256

    # Convert to float32 and normalize to [-1, 1]
    image = tf.cast(image, tf.float32) / 127.5 - 1.0

    # Split into input and target
    w = tf.shape(image)[1] // 2
    input_image = image[:, :w, :]
    target_image = image[:, w:, :]

    return input_image, target_image

def load_dataset(directory, batch_size=16, shuffle=True):
    files = tf.data.Dataset.list_files(os.path.join(directory, "*.png"), shuffle=shuffle)
    dataset = files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(100)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset