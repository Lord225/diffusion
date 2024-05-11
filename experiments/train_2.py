import os
import sys
import tensorflow as tf
import tensorboard as tb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import diffusion as df
from keras.datasets import mnist
import datetime

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
print(MODEL_DIR)
print(os.path.exists(MODEL_DIR))
RUN_NAME = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255

T = 1000
batch_size = 32
embedding_size = 128
input_dim = 64

alphas_cumprod, betas, alphas = df.calculate_variance(T)

model = df.build_model_2(T, embedding_size, input_dim)

model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")

model.summary()

# build dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(2)
train_dataset = train_dataset.map(lambda x, y: df.add_gauss_noise_to_image_context(x, y, alphas_cumprod, T))

# build test dataset
x_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
x_test = x_test.batch(batch_size).prefetch(2)
x_test = x_test.map(lambda x,y: df.add_gauss_noise_to_image_context(x, y, alphas_cumprod, T))

# get one batch
x, y = next(iter(train_dataset))

# print(x, y)

#checkpoit
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, 'model-2-{epoch}.h5'),
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
lr_sheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='loss')

model.fit(train_dataset, epochs=100, validation_data=x_test, callbacks=[tensorboard_cb, lr_sheduler, checkpoint_cb])




