
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
   