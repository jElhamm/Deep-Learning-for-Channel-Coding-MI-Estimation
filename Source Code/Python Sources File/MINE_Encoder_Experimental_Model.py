#******************************************************************************************************************************************
#                                                                                                                                         *
#                           Deep Learning for Channel Coding via Neural Mutual Information Estimation                                     *
#                                                                                                                                         *
#    This code implements an **Autoencoder-based communication system** with mutual information estimation (MINE).                        *
#                                                                                                                                         *
#   1.  It simulates a communication channel where messages are encoded using a neural network-based encoder,                             *
#       transmitted through a noisy channel (AWGN or Rayleigh fading), and decoded by a neural network-based decoder.                     *
#                                                                                                                                         *
#   2.  The system utilizes a Mutual Information Neural Estimator (MINE) to optimize the encoder and estimate mutual information          *
#       between the transmitted and received signals. Performance evaluation includes calculating Batch Bit Error Rate (BER) and          *
#       comparing against theoretical limits (16-QAM simulation).                                                                         *
#                                                                                                                                         *
#   3.  This setup allows for studying and optimizing the communication system under different SNR conditions.                            *
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


# --------------------------------------------------------------------- Configure matplotlib ---------------------------------------------------------------------------
# %matplotlib inline

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# -------------------------------------------------------------- Set random seeds for reproducibility ------------------------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)
 