import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import diffusion as df



print('Loading model')
T = 1000
batch_size = 32
embedding_size = 128
input_dim = 64
model = df.build_model_2(T, embedding_size, input_dim)
model.load_weights(f'models/model-2-66.h5')
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")
print('Model loaded')




print('Generating image')
alphas_cumprod, betas, alphas = df.calculate_variance(T)

context_1 = tf.one_hot([0]*8, 10)
context_2 = tf.one_hot([1]*8, 10)

# mix each context with the other context on different levels
mixs = np.linspace(0.4, 0.6, 8)
context = [context_1[i] * mix + context_2[i] * (1 - mix) for i, mix in enumerate(mixs)]
context = np.array(context)
print(context)

generated = df.generate_context_raw(model, 8, context, T, alphas, betas, alphas_cumprod)

fig, ax = plt.subplots(2, 4, figsize=(20, 10))

for i in range(8):
    ax[i // 4, i % 4].imshow(generated[i, :, :, 0], cmap='gray')
    ax[i // 4, i % 4].axis('off')

plt.show()




