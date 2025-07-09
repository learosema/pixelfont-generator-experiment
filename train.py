import os
import tensorflow as tf
import numpy as np
from PIL import Image
from generator import Generator
from discriminator import Discriminator
from loss_functions import generator_loss, discriminator_loss

def get_models_and_optimizers():
    generator = Generator()
    discriminator = Discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    return generator, discriminator, gen_optimizer, disc_optimizer

def save_prediction(image_tensor, output_tensor, step, outdir):
    input_img = ((image_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
    output_img = ((output_tensor[0].numpy() + 1) * 127.5).astype(np.uint8)
    combined = np.concatenate([input_img, output_img], axis=1).squeeze()
    Image.fromarray(combined, mode='L').save(os.path.join(outdir, f"step_{step:06}.png"))

@tf.function
def train_step(generator, discriminator, input_image, target,
               gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_grads = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

    return gen_total_loss, disc_loss

def load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, checkpoint_dir):
    checkpoint = tf.train.Checkpoint(generator=generator,
                                     discriminator=discriminator,
                                     gen_optimizer=gen_optimizer,
                                     disc_optimizer=disc_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    return checkpoint, manager