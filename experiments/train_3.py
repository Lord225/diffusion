import os
import sys
import tensorflow as tf
import tensorboard as tb

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import diffusion as df
from keras.datasets import mnist
import datetime
import argparse

parser = argparse.ArgumentParser(description='Generate images using trained model')

parser.add_argument('--resume', type=str, help='Path to model file')

parser.add_argument('--epoch-start', type=int, help='Epoch to start from')

args = parser.parse_args()


MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

RUN_NAME = "run_image_net" + datetime.datetime.now().strftime("%Y%m%d-%H%M")

from datasets import load_dataset

tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')

# remove images in grayscale
tiny_imagenet = tiny_imagenet.filter(lambda x: x['image'].layers == 3)
tiny_imagenet = tiny_imagenet.to_tf_dataset(
    columns=['image', 'label'],
)

# get subset (first 10'000 images)
tiny_imagenet = tiny_imagenet.take(20000)

T = 1000
batch_size = 64
embedding_size = 128
input_dim = 128

alphas_cumprod, betas, alphas = df.calculate_variance(T)

model = df.build_model_3(T, embedding_size, input_dim)

model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")

if args.resume:
    model.load_weights(args.resume)


model.summary()

#@tf.function
def preprocess_image(image):
    image = tf.image.resize(image, (64, 64))
    if image.shape[-1] != 3:
        image = tf.image.grayscale_to_rgb(image)
    # normalize
    image = tf.cast(image, tf.float32)

    image = image / 255.0 #type: ignore

    return image

# build dataset
train_dataset = tiny_imagenet
train_dataset = train_dataset.shuffle(5000).batch(batch_size).prefetch(2)
train_dataset = train_dataset.map(lambda data: {'image': preprocess_image(data['image']), 'label': data['label']})
train_dataset = train_dataset.map(lambda data: df.add_gauss_noise_to_image_context(data['image'], data['label'], alphas_cumprod, T, 200))


# import matplotlib.pyplot as plt
# # get first 25 baches, check dimensions
# for i, X in enumerate(train_dataset):
#     if i > 25:
#         break
#     print(X)
#     plt.imshow(X['image'][0, :, :, :])
#     plt.show()
    
#checkpoit
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, 'model-image_net-10000-big-{epoch}.weights.h5'),
    save_weights_only=True,
)

# tb
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', RUN_NAME),
    histogram_freq=1,
    write_images=True,
    update_freq=100, #type: ignore
    profile_batch=0,
    embeddings_freq=1,
)
# learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# lr_sheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, monitor='loss')

lr_sheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.00001 if epoch < 100 else 0.0000001)
# def schedule_with_params(lr_init=0.001, lr_end=1.e-6, nb_epochs=100):
#     import math
#     def schedule(epoch):
#         s = (math.log(lr_init) - math.log(lr_end))/math.log(10.)
#         lr = lr_init * 10.**(-float(epoch)/float(max(nb_epochs-1,1)) * s)
#         return lr
#     return tf.keras.callbacks.LearningRateScheduler(schedule)

# lr_sheduler = schedule_with_params(lr_init=0.001, lr_end=1.e-6, nb_epochs=80)

model.fit(train_dataset, epochs=400, callbacks=[tensorboard_cb, lr_sheduler, checkpoint_cb], initial_epoch=args.epoch_start if args.epoch_start else 0)




