import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import diffusion as df

# argparse
import argparse

parser = argparse.ArgumentParser(description='Generate images using trained model')

parser.add_argument('--model', type=str, help='Path to model file', required=True)
# generate all images
parser.add_argument('--all', action='store_true', help='Generate all images')
# generate class
parser.add_argument('--c', type=int, help='Generate images of a class')
# amout of images to generate (saves to file) (if not specified, shows images)
parser.add_argument('--amount', type=int, help='Amount of images to generate', default=None)
# load model

args = parser.parse_args()

print('Loading model')
T = 1000
batch_size = 32
embedding_size = 128
input_dim = 64
model = df.build_model_2(T, embedding_size, input_dim)
model.load_weights(args.model)
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")
print('Model loaded')

import cv2


print('Generating image')
alphas_cumprod, betas, alphas = df.calculate_variance(T)

if args.amount:
    import os
    generated = df.generate_context(model, args.amount, args.c, T, alphas, betas, alphas_cumprod)

    # save images into /temp/c folder
    for i in range(args.amount):

        if not os.path.exists(f'temp/{args.c}'):
            os.makedirs(f'temp/{args.c}')
        plt.imsave(f'temp/{args.c}/{i}.png', generated[i, :, :, 0], cmap='gray')

    exit()
if args.all:
    generated = []

    for i in range(10):
        generated.append(df.generate_context(model, 4, i, T, alphas, betas, alphas_cumprod))

    # plot all images (10x4)

    fig, ax = plt.subplots(10, 4, figsize=(20, 50))

    for i in range(10):
        for j in range(4):
            ax[i, j].imshow(generated[i][j, :, :, 0], cmap='gray')
            ax[i, j].axis('off')

    plt.show()
else:
    # return list of 8 generated each time step, 8 images each time step
    generated = df.generate_context(model, 8, args.c, T, alphas, betas, alphas_cumprod)

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(8):
        img = generated[i, :, :, 0]
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min())
        (t, tresh_img) = cv2.threshold(img, 0.3, 1, cv2.THRESH_TOZERO)
        ax[i // 4, i % 4].imshow(tresh_img, cmap='gray')
        ax[i // 4, i % 4].axis('off')
    # save img
    plt.savefig(f'temp/{args.c}.png')

    plt.show()




