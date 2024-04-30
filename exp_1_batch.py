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
model.load_weights('models/model-85.h5')
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
generated = df.generate(model, 8, T, alphas, betas, alphas_cumprod)

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

for i in range(8):
    ax[i // 4, i % 4].imshow(generated[i, :, :, 0], cmap='gray')
    ax[i // 4, i % 4].axis('off')

plt.show()




