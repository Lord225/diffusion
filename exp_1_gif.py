import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import diffusion as df

# load model

print('Loading model')
T = 1000
batch_size = 64
embedding_size = 128
input_dim = 32
model = df.build_model(T, embedding_size, input_dim)
model.load_weights('models/model-24.h5')
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")
print('Model loaded')



# noise = np.random.normal(0, 1, (1, 28, 28, 1))
# t = 1000
# # predict
# pred = model.predict({
#     'X_Noisy': noise,
#     't_Input': np.array([t])
# })

# # plot
# plt.imshow(pred[0, :, :, 0], cmap='gray')
# plt.show()


# generate image using model

print('Generating image')

alphas_cumprod, betas, alphas = df.calculate_variance(T)
print('Alphas:', alphas.shape)
print('Betas:', betas.shape)
print('Alphas cumprod:', alphas_cumprod.shape)

# return list of 8 generated each time step, 8 images each time step
generated = df.generate_return_all(model, 1, T, alphas, betas, alphas_cumprod)

generated = np.array(generated)

import imageio
import matplotlib.pyplot as plt
from PIL import Image
import io

def create_gif(images, filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for image in images:
            # Render image using plt
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')
            ax.axis('off')

            # Convert plt figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pil_img = Image.open(buf)

            # Convert PIL Image to 8-bit numpy array
            img_array = np.array(pil_img)
            img_8bit = (img_array * 255).astype(np.uint8)

            # Append to GIF
            writer.append_data(img_8bit) #type: ignore
            plt.close(fig)


# Create a GIF for each image in the batch
for i in range(generated.shape[1]):
    create_gif(generated[:, i, :, :, 0], f'output_{i}.gif')