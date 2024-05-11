import os
import sys
import keras
import tensorflow as tf
import tensorboard as tb

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import diffusion as df
from keras.datasets import mnist # type: ignore
import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


RUN_NAME = "run_image_net" + datetime.datetime.now().strftime("%Y%m%d-%H%M")


from datasets import load_dataset

tiny_imagenet = load_dataset('Maysee/tiny-imagenet', split='train')

# remove images in grayscale
tiny_imagenet = tiny_imagenet.filter(lambda x: x['image'].layers == 3)

tiny_imagenet = tiny_imagenet.to_tf_dataset(
    columns=['image', 'label'],
)


T = 1000
batch_size = 32
embedding_size = 128
input_dim = 64

alphas_cumprod, betas, alphas = df.calculate_variance(T)

model = df.build_model_3(T, embedding_size, input_dim) # type: keras.Model

model.compile(loss=keras.losses.Huber(), optimizer="nadam")

model.summary()

@tf.function
def preprocess_image(image):
    image = tf.image.resize(image, (64, 64))
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)    
    image = tf.cast(image, tf.float32) / 255. #type: ignore
    return image

# build dataset
train_dataset = tiny_imagenet
train_dataset = train_dataset.batch(batch_size).prefetch(2)
train_dataset = train_dataset.map(lambda data: {'image': preprocess_image(data['image']), 'label': data['label']})
train_dataset = train_dataset.map(lambda data: df.add_gauss_noise_to_image_context(data['image'], data['label'], alphas_cumprod, T, 200))

# get first 25 baches, check dimensions
# for i, (X, y) in enumerate(train_dataset):
#     if i > 25:
#         break
#     print(X, y)
    
#checkpoit
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, 'model-image_net-{epoch}.weights.h5'),
    save_weights_only=True,
)

# tb
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', RUN_NAME),
    histogram_freq=1,
    write_images=True,
    update_freq=100, #type: ignore
    profile_batch=0,
    embeddings_freq=1,
)
# learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
lr_sheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20, monitor='loss')

model.fit(train_dataset, epochs=1000, callbacks=[tensorboard_cb, lr_sheduler, checkpoint_cb])




