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
    