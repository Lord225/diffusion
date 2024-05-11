from matplotlib import axis
import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from common import positional_encoding


def build_unet_2(X, embed):
    # encoder
    cross = []

    cross.append(X)

    X = tf.keras.layers.SeparableConv2D(32, 3, padding='same', activation='elu')(X) # 32x32
    X = tf.keras.layers.BatchNormalization()(X)
    
    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, embed_tiled])
    
    X = tf.keras.layers.SeparableConv2D(32, 3, padding='same', activation='elu')(X) # 32x32
    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.MaxPooling2D()(X) # 16x16
    
    cross.append(X) # 16x16

    X = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 16x16
    X = tf.keras.layers.BatchNormalization()(X)
    
    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, embed_tiled])


    X = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 16x16
    X = tf.keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.MaxPooling2D()(X) # 8x8
 
    cross.append(X) # 8x8

    X = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 8x8
    X = tf.keras.layers.BatchNormalization()(X)

    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, embed_tiled])

    X = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='elu')(X) # 8x8
    X = tf.keras.layers.BatchNormalization()(X)
    
    X = tf.keras.layers.MaxPooling2D()(X) # 4x4

    # decoder
    X = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='elu')(X) # 8x8
    X = tf.keras.layers.BatchNormalization()(X)


    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 8x8

    X = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='elu')(X) # 16x16
    X = tf.keras.layers.BatchNormalization()(X)
    
    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 16x16

    X = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='elu')(X) # 32x32
    X = tf.keras.layers.BatchNormalization()(X)

    embed_tiled = tf.tile(embed, [1, X.shape[1], X.shape[2], 1]) #type: ignore
    X = tf.keras.layers.Concatenate()([X, cross.pop(), embed_tiled]) # 32x32

    X = tf.keras.layers.Conv2D(1, 3, padding='same')(X) # 32x32

    X = tf.keras.layers.Cropping2D(cropping=(2, 2))(X) # 28x28

    return X

def build_model_2(
    T: int,
    T_embedding: int,
    dim: int,
):
    X_Noisy = tf.keras.layers.Input(shape=(28, 28, 1), name='X_Noisy')
    # input embedding layer
    t_Input = tf.keras.layers.Input(shape=[], name='t_Input', dtype=tf.int32)

    c_Input = tf.keras.layers.Input(shape=[10], name='c_Input', dtype=tf.float32)

    pos_encoding = tf.constant(tf.reshape(positional_encoding(T, T_embedding), (T, T_embedding)), dtype=tf.float32)

    # pad to 32x32
    X = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(X_Noisy)
    X = tf.keras.layers.Conv2D(dim, 3, padding='same', activation='elu')(X)

    time = tf.gather(pos_encoding, t_Input)
    encoded_time = tf.keras.layers.Dense(dim)(time) # None, 1000, 64
    encoded_cont = tf.keras.layers.Dense(dim)(c_Input)

    X = encoded_time[:, tf.newaxis, tf.newaxis] + X #type: ignore

    embed = encoded_cont[:, tf.newaxis, tf.newaxis] #type: ignore

    X = build_unet_2(X, embed)

    return tf.keras.Model(inputs=[X_Noisy, t_Input, c_Input], outputs=X, name='UNet')



if __name__ == '__main__':
    model = build_model_2(1000, 64, 16)
    model.summary()






    









