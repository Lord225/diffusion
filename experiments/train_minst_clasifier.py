from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import argparse

parser = argparse.ArgumentParser(description='Generate images using trained model')

parser.add_argument('--generated-data', action='store_true', help='Amount of generated data for each class')
parser.add_argument('--generated-weight', type=float, help='Weight of generated data', default=1.0)


args = parser.parse_args()

import tensorboard as tb

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last',
                 input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid' ))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=1, padding='same', data_format='channels_last'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

import os
import datetime
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models-minst')
print(MODEL_DIR)
print(os.path.exists(MODEL_DIR))
RUN_NAME = "minst_run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")


import tensorboard
import tensorflow as tf

optimizer = Adam(beta_1=0.9, beta_2=0.999)

model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'], weighted_metrics=[])

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, 'model-gen-only-best-2-{epoch}.weights.h5'),
)

# tb
tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', RUN_NAME),
    histogram_freq=1,
    update_freq=10, #type: ignore
    profile_batch=0,
    embeddings_freq=1,
)

lr_sheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * 0.5 ** (epoch / 25))
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# take subset of 9000 images
# x_train = x_train[:9000]
# y_train = y_train[:9000]

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, np.ones_like(y_train, dtype=np.float32)))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, np.ones_like(y_test, dtype=np.float32)))

# Map and preprocess the data
train_dataset = train_dataset.map(lambda x, y, w: (tf.cast(x[..., tf.newaxis], tf.float32), tf.cast(y, tf.int32), w))
test_dataset = test_dataset.map(lambda x, y, w: (tf.cast(x[..., tf.newaxis], tf.float32), tf.cast(y, tf.int32), w))


# Optionally add generated data
if args.generated_data:
    train_dataset_gen = tf.keras.utils.image_dataset_from_directory(
        os.path.join(os.path.dirname(__file__), '..', 'dataset'),
        seed=123,
        image_size=(28, 28),
        color_mode='grayscale',
        batch_size=64,
    )

    # load data into numpy array
    x_gen = []
    y_gen = []

    for x, y in train_dataset_gen:
        x_gen.append(x)
        y_gen.append(y)

    # transform to numpy
    x_gen = np.concatenate(x_gen)
    y_gen = np.concatenate(y_gen)

    #     (10000, 28, 28, 1)
    # (10000,)
    # print(x_gen.shape)
    # print(y_gen.shape)

    # load model that will filter out the misspredicted images
    f = tf.keras.models.load_model(os.path.join(MODEL_DIR, '/home/lord225/pyrepos/diffusion/models-minst/model-38.weights.h5'))

    # predict the images
    y_pred = f.predict(x_gen)

    # filter out the misspredicted images, 
    misses = np.argmax(y_pred, axis=1) != y_gen

    print('in generated dataset model was not able to correcly identify:', np.sum(misses))

    import matplotlib.pyplot as plt
    # rank wrongly predicted images by uncertainty
    # sort by uncertainty mesured by entropy
    entropy = -np.sum(y_pred * np.log(y_pred), axis=1)

    # sort by entropy
    idx = np.argsort(entropy)

    # remove idx that are correcly clasified
    idx = idx[~misses]

    print(idx)
    print(idx.shape)
    print(misses)
    # plot the entropy
    plt.plot(entropy[idx])

    plt.show()

    # plot top 25 most uncertain images
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))

    for i in range(25):
        ax[i // 5, i % 5].imshow(x_gen[idx[i+7000], :, :, 0], cmap='gray')
        ax[i // 5, i % 5].axis('off')
    
    plt.show()

    # remove last 2000 idx from the generated dataset
    # these are garbge that model is not able to clasify.
    x_gen = x_gen[idx[:-3000]]
    y_gen = y_gen[idx[:-3000]]

    print(x_gen.shape)
    print(y_gen.shape)

    # exit()
    



    # load using from_tensor_slices
    train_dataset_gen = tf.data.Dataset.from_tensor_slices((x_gen, y_gen, np.ones_like(y_gen, dtype=np.float32)))

    # Map and preprocess the data
    train_dataset_gen = train_dataset_gen.map(lambda x, y, w: (tf.cast(x, tf.float32), tf.cast(y, tf.int32), w))
    # load the 
    # merge datasets
    train_dataset = train_dataset.concatenate(train_dataset_gen)
    #train_dataset = train_dataset_gen # train only on generated images
    
# batch , shuffle and prefetch
train_dataset = train_dataset.shuffle(1024).batch(64).prefetch(2)
test_dataset = test_dataset.batch(64).prefetch(2)


# 20/50
# 9s 9ms/step - loss: 0.0122 - accuracy: 0.9963 - val_loss: 0.0216 - val_accuracy: 0.9942 - lr: 5.9050e-04

# only gen
# Epoch 23/50
# 131/131 [==============================] - 5s 36ms/step - loss: 0.0221 - accuracy: 0.9937 - val_loss: 0.1803 - val_accuracy: 0.9646 - lr: 5.4337e-04

# subset 9000 of mnist
# 141/141 [==============================] - 5s 33ms/step - loss: 0.0147 - accuracy: 0.9946 - val_loss: 0.0501 - val_accuracy: 0.9872 - lr: 6.9737e-04



model.fit(train_dataset,
          batch_size=64, 
          epochs=50, 
          # valbatch size 64
          validation_data=test_dataset,
          callbacks=[checkpoint_cb, tensorboard_cb, lr_sheduler])