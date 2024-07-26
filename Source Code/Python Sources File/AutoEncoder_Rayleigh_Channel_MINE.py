
#******************************************************************************************************************************************
#                                                                                                                                         *
#                           Deep Learning for Channel Coding via Neural Mutual Information Estimation                                     *
#                                                                                                                                         *
#    This code sets up and trains an autoencoder for 16-QAM **communication systems using deep learning.                                  *
#                                                                                                                                         *
#   1.  It defines the autoencoder's architecture and functionality, including encoding, adding noise, and decoding.                      *
#       It also includes a neural network for estimating mutual information, which is used to optimize the autoencoder's performance.     *
#                                                                                                                                         *
#   2.  The training process involves multiple epochs and steps, adjusting the encoder, decoder, and mutual information estimator         *
#       through gradient-based optimization. Finally, the code tests the autoencoder's performance by evaluating its bit error rate       *
#        across a range of signal-to-noise ratios, providing insights into its effectiveness under different conditions.                  *
#                                                                                                                                         *
#******************************************************************************************************************************************



import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from scipy import special
from tensorflow import keras
import matplotlib.pyplot as plt
assert sys.version_info >= (3, 5)


# ------------------------------------------------------------------- Set random seeds for reproducibility -------------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)
   

# ------------------------------------------------ The AutoEncoder class models a 16-QAM communication system using neural networks ------------------------------------------------

class AutoEncoder:
    def __init__(self, M=16, n=1, training_snr=7, rayleigh=False):
        self.M = M
        self.k = int(np.log2(M))
        self.n = n
        self.training_snr = training_snr
        self.rayleigh = rayleigh
        self.noise_std = self.EbNo_to_noise(training_snr)

        # ---------------------------------------------------- Define custom layers -----------------------------------------------------------

        self.norm_layer = keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))
        self.shape_layer = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2, n]))
        self.shape_layer2 = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2 * n]))
        self.channel_layer = keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std))

        # --------------------------------------------------- Define the encoder model --------------------------------------------------------
        self.encoder = keras.models.Sequential([
            keras.layers.Embedding(M, M, embeddings_initializer='glorot_normal', input_length=1),
            keras.layers.Dense(M, activation="elu"),
            keras.layers.Dense(2 * n, activation=None),
            self.shape_layer,
            self.norm_layer
        ])
    