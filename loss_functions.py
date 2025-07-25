import tensorflow as tf

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total = gan_loss + (LAMBDA * l1_loss)
    return total, gan_loss, l1_loss
