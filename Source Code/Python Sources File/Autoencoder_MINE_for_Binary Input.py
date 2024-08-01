#************************************************************************************************************************************************
#                                                                                                                                               *
#                                 Deep Learning for Channel Coding via Neural Mutual Information Estimation                                     *
#                                                                                                                                               *
#       This code implements a **Communication System Using Deep Learning** techniques.                                                         *
#                                                                                                                                               *
#       - The main idea is to use an autoencoder to enhance the encoding and decoding of information.                                           *
#         The autoencoder efficiently encodes data, which is then transmitted through a noisy channel.                                          *
#         Finally, the received data is decoded by another deep learning model.                                                                 *
#                                                                                                                                               *
#       - The process involves optimizing the models to minimize the bit error rate (BER) under varying signal-to-noise ratio (SNR)             *
#         conditions. Additionally, neural networks are used to estimate mutual information to further improve the system's performance.        *
#                                                                                                                                               *
#************************************************************************************************************************************************



import sys
import itertools
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from scipy import special
from tensorflow import keras
import matplotlib.pyplot as plt
assert sys.version_info >= (3, 5)
from tensorflow.keras import layers


# ---------------------------------------------------------------------------------- Definition Of Constants --------------------------------------------------------------------------------------

M = 16
k = int(np.log2(M))
n = 1
TRAINING_SNR = 7
BINARY_INP = True
rayleigh = False
    

# ---------------------------------------------- The AutoEncoder class implements an end-to-end communication system using deep learning techniques ------------------------------------------------

class AutoEncoder:
    def __init__(self, M, n, training_snr, rayleigh=False, binary_input=True):
        self.M = M
        self.k = int(np.log2(M))
        self.n = n
        self.training_snr = training_snr
        self.rayleigh = rayleigh
        self.binary_input = binary_input

        self.noise_std = self.EbNo_to_noise(training_snr)

        # Define custom layers
        self.norm_layer = layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))
        self.shape_layer = layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2, n]))
        self.shape_layer2 = layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2 * n]))
        self.channel_layer = layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std))
    