import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard as tb


def calculate_variance(
        T: int,
        s: float = 0.001,
): 
    ts = np.arange(T)

    # Improved Denoising Diffusion Probabilistic Models Eq 17
    f = np.cos((ts/T+s)/(1+s) * np.pi/2) ** 2

    alphas = f[1:] / f[:-1]

    alphas_cumprod = f[ts] / f[0]

    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = np.clip(betas, 0.0001, 0.999)


    betas = np.concatenate([np.array([1.0]), betas])

    alphas = np.concatenate([np.array([1.0]), alphas])


    return alphas_cumprod.astype('float32'), betas.astype('float32'), alphas.astype('float32')

# @tf.function
# def add_gauss_noise_to_image(
#     imgs: tf.Tensor,
#     alpha_cumprod: tf.Tensor,
# ):
#     samples = imgs.shape[0]
#     T = alpha_cumprod.shape[0]

#     noise = tf.random.normal(imgs.shape)
#     t = tf.random.uniform([samples], maxval=T, dtype=tf.int32)
    
#     alphas = tf.reshape(tf.gather(alpha_cumprod, t), shape=(-1, 1, 1))
#     noisy_imgs = tf.sqrt(alphas)*imgs + tf.sqrt(1-alphas)*noise

#     return {
#         'X_Noisy': noisy_imgs,
#         't_Input': t,
#     }, 

@tf.function
def add_gauss_noise_to_image(
    imgs: tf.Tensor,
    alpha_cumprod: tf.Tensor,
    T: int,
):
    X_shape = tf.shape(imgs)
    t = tf.random.uniform([X_shape[0]], minval=1, maxval=T, dtype=tf.int32)
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
    noise = tf.random.normal(X_shape)
    return {
        "X_Noisy": alpha_cm ** 0.5 * imgs + (1 - alpha_cm) ** 0.5 * noise,
        "t_Input": t,
    }, noise


#@tf.function
def add_gauss_noise_to_image_context(
    imgs,
    context,
    alpha_cumprod: tf.Tensor,
    T: int,
):
    X_shape = tf.shape(imgs)
    t = tf.random.uniform([X_shape[0]], minval=1, maxval=T, dtype=tf.int32)
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
    noise = tf.random.normal(X_shape)
    context = tf.one_hot(context, 200)
    return {
        "X_Noisy": alpha_cm ** 0.5 * imgs + (1 - alpha_cm) ** 0.5 * noise,
        "t_Input": t,
        "c_Input": context,
    }, noise



@tf.function
def sub_gauss_noise_from_image(
    noisy_imgs: tf.Tensor,
    alpha_cumprod: tf.Tensor,
    t: tf.Tensor,
    noise: tf.Tensor,
):
    alphas = tf.reshape(tf.gather(alpha_cumprod, t), shape=(-1, 1, 1))
    imgs = (noisy_imgs - tf.sqrt(1-alphas)*noise) / tf.sqrt(alphas)


    return imgs

from tqdm import tqdm

def generate(model, batch_size, T, alpha, beta, alpha_cumprod): 
    X = tf.random.normal([batch_size, 28, 28, 1])

    for t in tqdm(range(T - 1, 0, -1)):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_Noisy": X, "t_Input": tf.constant([t] * batch_size)})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
    return X

def generate_context(model, batch_size, context, T, alpha, beta, alpha_cumprod): 
    X = tf.random.normal([batch_size, 28, 28, 1])
    context = tf.one_hot([context]*batch_size, 10)
    for t in tqdm(range(T - 1, 0, -1)):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_Noisy": X, "t_Input": tf.constant([t] * batch_size), "c_Input": context})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
    return X

def generate_context_raw(model, batch_size, context, T, alpha, beta, alpha_cumprod): 
    X = tf.random.normal([batch_size, 28, 28, 1])
    for t in tqdm(range(T - 1, 0, -1)):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_Noisy": X, "t_Input": tf.constant([t] * batch_size), "c_Input": context})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
    return X

def generate_return_all_context(model, batch_size, context, T, alpha, beta, alpha_cumprod): 
    X = tf.random.normal([batch_size, 28, 28, 1])
    context = tf.one_hot([context]*batch_size, 10)
    imgs = []
    for t in tqdm(range(T - 1, 0, -1)):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_Noisy": X, "t_Input": tf.constant([t] * batch_size), "c_Input": context})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
        imgs.append(X.numpy())
    return imgs

def generate_return_all(model, batch_size, T, alpha, beta, alpha_cumprod): 
    X = tf.random.normal([batch_size, 28, 28, 1])
    imgs = []
    for t in tqdm(range(T - 1, 0, -1)):
        noise = (tf.random.normal if t > 1 else tf.zeros)(tf.shape(X))
        X_noise = model({"X_Noisy": X, "t_Input": tf.constant([t] * batch_size)})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
        )
        imgs.append(X.numpy())
    return imgs
# positional encoding
# https://www.tensorflow.org/text/tutorials/transformer?hl=pl
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class TileEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TileEmbedding, self).__init__(**kwargs)

    def call(self, inputs):
        X, embed = inputs
        return tf.tile(embed, [1, tf.shape(X)[1], tf.shape(X)[2], 1])