import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import diffusion as df
import os
# argparse
import argparse

parser = argparse.ArgumentParser(description='Generate images using trained model')

parser.add_argument('--model', type=str, help='Path to model file', required=True)
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
from tqdm import tqdm

print('Generating image')
alphas_cumprod, betas, alphas = df.calculate_variance(T)

if args.amount:

    for i in range(10):
        generated = df.generate_context(model, args.amount, i, T, alphas, betas, alphas_cumprod)

        generated = np.array(generated) # type: np.ndarray

        print('generated', generated.shape)
        # print(generated)
        
        if not os.path.exists(f'./dataset/{i}'):
            os.makedirs(f'./dataset/{i}')

        print('saving images')

        # save images into /temp/c folder
        for j in tqdm(range(args.amount), desc=f'Saving class {i}'):
            # save imgs

            img = generated[j, :, :, 0]
            img = np.array(img)

            # normalize
            img = (img - img.min()) / (img.max() - img.min())

            (t, tresh_img) = cv2.threshold(img, 0.3, 1, cv2.THRESH_TOZERO)

            # mix with original image just a bit
            img = 0.99*tresh_img + 0.01*img

            img = img*255
            img = np.clip(img, 0, 255)

            cv2.imwrite(f'./dataset/{i}/{j}.png', img)