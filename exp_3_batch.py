import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import diffusion as df

# argparse
import argparse

parser = argparse.ArgumentParser(description='Generate images using trained model')

parser.add_argument('--model', type=str, help='Path to model file', required=True)


parser.add_argument('--c', type=int, help='Generate images of a class')


args = parser.parse_args()

print('Loading model')

T = 1000
batch_size = 32
embedding_size = 128
input_dim = 128

alphas_cumprod, betas, alphas = df.calculate_variance(T)

model = df.build_model_3(T, embedding_size, input_dim)

model.load_weights(args.model)
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")

model.summary()

print('Model compiled')

print('Generating image')


generated = df.generate_context(model, 8, args.c, T, alphas, betas, alphas_cumprod, 200, img_shape=(64, 64, 3))

print(generated.shape)
print(generated[0, :, :, :])

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

for i in range(8):
    img = generated[i, :, :, :]
    img = np.array(img)
    # normalize
    #img = (img - np.min(img)) / (np.max(img) - np.min(img))
    ax[i // 4, i % 4].imshow(img)
    ax[i // 4, i % 4].axis('off')

# save img
plt.savefig(f'temp/image-net-{args.c}.png')

plt.show()