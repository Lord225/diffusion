import keras
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from common import positional_encoding

class TileEmbedding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TileEmbedding, self).__init__(**kwargs)

    def call(self, inputs):
        X, embed = inputs
        return tf.tile(embed, [1, tf.shape(X)[1], tf.shape(X)[2], 1])


def build_unet_2(X, embed):
    # encoder
    cross = []

    cross.append(X)

    X = keras.layers.SeparableConv2D(32, 3, padding='same', activation='elu')(X) # 32x32
    X = keras.layers.BatchNormalization()(X)
    
    # tile using [1, X.shape[1], X.shape[2]
    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, embed_tiled])
    
    X = keras.layers.SeparableConv2D(32, 3, padding='same', activation='elu')(X) # 32x32
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.MaxPooling2D()(X) # 16x16
    
    cross.append(X) # 16x16

    X = keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 16x16
    X = keras.layers.BatchNormalization()(X)
    
    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, embed_tiled])


    X = keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 16x16
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.MaxPooling2D()(X) # 8x8
 
    cross.append(X) # 8x8

    X = keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 8x8
    X = keras.layers.BatchNormalization()(X)

    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, embed_tiled])

    X = keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 8x8
    X = keras.layers.BatchNormalization()(X)
    
    X = keras.layers.MaxPooling2D()(X) # 4x4

    # decoder
    X = keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='elu')(X) # 8x8
    X = keras.layers.BatchNormalization()(X)


    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 8x8

    X = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='elu')(X) # 16x16
    X = keras.layers.BatchNormalization()(X)
    
    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 16x16

    X = keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='elu')(X) # 32x32
    X = keras.layers.BatchNormalization()(X)

    embed_tiled = TileEmbedding()([X, embed])
    X = keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 32x32

    X = keras.layers.Conv2D(1, 3, padding='same')(X) # 32x32

    X = keras.layers.Cropping2D(cropping=(2, 2))(X) # 28x28

    return X

def build_model_2(
    T: int,
    T_embedding: int,
    dim: int,
):
    X_Noisy = keras.layers.Input(shape=(28, 28, 1), name='X_Noisy')
    # input embedding layer
    t_Input = keras.layers.Input(shape=[], name='t_Input', dtype=tf.int32)

    c_Input = keras.layers.Input(shape=[10], name='c_Input', dtype=tf.float32)

    pos_encoding = tf.constant(tf.reshape(positional_encoding(T, T_embedding), (T, T_embedding)), dtype=tf.float32)

    # pad to 32x32
    X = keras.layers.ZeroPadding2D(padding=(2, 2))(X_Noisy)
    X = keras.layers.Conv2D(dim, 3, padding='same', activation='elu')(X)

    time = keras.layers.Lambda(lambda x: tf.gather(pos_encoding, x), output_shape=(T_embedding,))(t_Input)
    encoded_time = keras.layers.Dense(dim)(time) # None, 1000, 64
    encoded_cont = keras.layers.Dense(dim)(c_Input)

    X = encoded_time[:, tf.newaxis, tf.newaxis] + X

    embed = encoded_cont[:, tf.newaxis, tf.newaxis]

    X = build_unet_2(X, embed)

    return keras.Model(inputs=[X_Noisy, t_Input, c_Input], outputs=X, name='UNet')



if __name__ == '__main__':
    model = build_model_2(1000, 64, 16)
    model.summary()






    









