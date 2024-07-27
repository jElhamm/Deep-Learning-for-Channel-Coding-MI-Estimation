
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
    
        # --------------------------------------------------- Define the channel model --------------------------------------------------------
        if rayleigh:
            self.channel = keras.models.Sequential([keras.layers.Lambda(lambda x: self.sample_Rayleigh_channel(x, self.noise_std))])
        else:
            self.channel = keras.models.Sequential([self.channel_layer])

        # --------------------------------------------------- Define the decoder model --------------------------------------------------------
        self.decoder = keras.models.Sequential([
            keras.layers.InputLayer(input_shape=[2, n]),
            self.shape_layer2,
            keras.layers.Dense(M, activation="elu"),
            keras.layers.Dense(M, activation="softmax")
        ])

        # Combine encoder, channel, and decoder into an autoencoder model
        self.autoencoder = keras.models.Sequential([self.encoder, self.channel, self.decoder])
    # -------------------------------------------------------------------------------------------------------------------------------------

    def EbNo_to_noise(self, ebnodb):
        """Transform EbNo[dB]/snr to noise power"""
        ebno = 10**(ebnodb/10)
        noise_std = 1/np.sqrt(2*(self.k/self.n)*ebno)
        return noise_std

    def sample_Rayleigh_channel(self, x, noise_std):
        h_sample = (1/np.sqrt(2)) * tf.sqrt(tf.random.normal(tf.shape(x))**2 + tf.random.normal(tf.shape(x))**2)
        z_sample = tf.random.normal(tf.shape(x), stddev=noise_std)
        y_sample = x + tf.divide(z_sample, h_sample)
        return tf.cast(y_sample, tf.float32)
    
    def random_sample(self, batch_size=32):
        msg = np.random.randint(self.M, size=(batch_size, 1))
        return msg

    def B_Ber_m(self, input_msg, msg):
        """Calculate the Batch Bit Error Rate"""
        batch_size = input_msg.shape[0]
        pred_error = tf.not_equal(tf.reshape(input_msg, shape=(-1, batch_size)), tf.argmax(msg, 1))
        bber = tf.reduce_mean(tf.cast(pred_error, tf.float32))
        return bber
    
    def test_encoding(self):
        inp = np.arange(0, self.M)
        coding = self.encoder.predict(inp)
        fig = plt.figure(figsize=(4,4))
        plt.plot(coding[:,0], coding[:, 1], "b.")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18, rotation=0)
        plt.grid(True)
        plt.gca().set_ylim(-2, 2)
        plt.gca().set_xlim(-2, 2)
        plt.show()
    
# ------------------------------------------------ NNFunction is a custom neural network model using TensorFlow's Keras API ---------------------------------------------------------- 

class NNFunction(tf.keras.Model):
    def __init__(self, hidden_dim, layers, activation, **extra_kwargs):
        """
            Initialize a neural network model for function approximation.
        """
        super(NNFunction, self).__init__()
        self._f = tf.keras.Sequential(
            [tf.keras.layers.Dense(hidden_dim, activation) for _ in range(layers)] +
            [tf.keras.layers.Dense(1)]
        )
    