import tensorflow as tf
import unittest
import numpy as np
from . import common

class TestDiffusion(unittest.TestCase):
    def test_calculate_variance(self):
        alphas_cumprod, betas, alphas = common.calculate_variance(1000)
        self.assertEqual(alphas_cumprod.shape, (1000,))
        self.assertEqual(betas.shape, (999,))

    def test_add_gauss_noise_to_image(self):
        alphas_cumprod, betas, alphas = common.calculate_variance(1000)

        data = np.zeros((10, 28, 28))
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        ((noisy, t), noise) = common.add_gauss_noise_to_image(data, alphas_cumprod) # type: ignore
        self.assertEqual(noisy.shape, data.shape)
        self.assertEqual(t.shape, (10,))
        self.assertEqual(noise.shape, data.shape)
        self.assertEqual(noise.dtype, tf.float32)
