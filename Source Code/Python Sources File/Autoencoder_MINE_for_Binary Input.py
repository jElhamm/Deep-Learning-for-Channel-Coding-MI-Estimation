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
    
        # Define the encoder model
        if binary_input:
            self.encoder = keras.models.Sequential([
                keras.layers.InputLayer(input_shape=[k]),
                keras.layers.Dense(2 * k, activation="elu"),
                keras.layers.Dense(2 * n, activation=None),
                self.shape_layer,
                self.norm_layer
            ])
        else:
            self.encoder = keras.models.Sequential([
                keras.layers.Embedding(M, M, embeddings_initializer='glorot_normal', input_length=1),
                keras.layers.Dense(M, activation="elu"),
                keras.layers.Dense(2 * n, activation=None),
                self.shape_layer,
                self.norm_layer
            ])
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
    def EbNo_to_noise(self, ebnodb):
        '''Convert Eb/N0 (dB) to noise standard deviation.'''
        ebno = 10**(ebnodb / 10)                                                                                                        # Convert dB to linear scale
        noise_std = 1 / np.sqrt(2 * (self.k / self.n) * ebno)                                                                           # Compute noise std deviation
        return noise_std

    def sample_Rayleigh_channel(self, x, noise_std):
        '''Simulate a Rayleigh channel with noise.'''
        h_sample = (1 / np.sqrt(2)) * tf.sqrt(tf.random.normal(tf.shape(x))**2 + tf.random.normal(tf.shape(x))**2)                      # Generate Rayleigh channel coefficients
        z_sample = tf.random.normal(tf.shape(x), stddev=noise_std)                                                                      # Generate noise
        y_sample = x + tf.divide(z_sample, h_sample)                                                                                    # Add noise to the signal
        return tf.cast(y_sample, tf.float32)                                                                                            # Return noisy signal
    
    def random_sample(self, batch_size=32):
        '''Generate random binary or integer samples.'''
        if self.binary_input:
            msg = np.random.randint(2, size=(batch_size, self.k))                                                                       # Binary input
        else:
            msg = np.random.randint(self.M, size=(batch_size, 1))                                                                       # Integer input
        return msg

    def B_Ber_m(self, input_msg, msg):
        '''Compute the Batch Bit Error Rate (BBER).'''
        batch_size = input_msg.shape[0]
        if self.binary_input:
            pred_error = tf.not_equal(input_msg, tf.round(msg))                                                                         # Compare predictions with actual
            pred_error_msg = tf.reduce_max(tf.cast(pred_error, tf.float32), axis=1)                                                     # Max error per sample
            bber = tf.reduce_mean(pred_error_msg)                                                                                       # Mean error over batch
        else:
            pred_error = tf.not_equal(tf.reshape(input_msg, shape=(-1, batch_size)), tf.argmax(msg, 1))                                 # Compare class predictions
            bber = tf.reduce_mean(tf.cast(pred_error, tf.float32))                                                                      # Mean error over batch
        return bber
    
    def test_encoding(self):
        if self.binary_input:
            inp = np.array([list(i) for i in itertools.product([0, 1], repeat=k)])
        else:
            inp = np.arange(0, self.M)
        coding = self.encoder.predict(inp)
        fig = plt.figure(figsize=(4, 4))
        plt.plot(coding[:, 0], coding[:, 1], "b.")
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18, rotation=0)
        plt.grid(True)
        plt.gca().set_ylim(-2, 2)
        plt.gca().set_xlim(-2, 2)
        plt.show()
    
# -------------------------------------- The NNFunction class defines a neural network-based function approximator using the TensorFlow Keras Model API --------------------------------------------

class NNFunction(tf.keras.Model):
    def __init__(self, hidden_dim, layers, activation, **extra_kwargs):
        super(NNFunction, self).__init__()
        self._f = tf.keras.Sequential(
            [tf.keras.layers.Dense(hidden_dim, activation) for _ in range(layers)] +
            [tf.keras.layers.Dense(1)]
        )
    
    def call(self, x, y):
        batch_size = tf.shape(x)[0]
        x_tiled = tf.tile(x[None, :], (batch_size, 1, 1))
        y_tiled = tf.tile(y[:, None], (1, batch_size, 1))
        xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2), [batch_size * batch_size, -1])
        scores = self._f(xy_pairs)
        return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))
    

# ---------------------------------------------- The Trainer class is responsible for training and evaluating the AutoEncoder and NNFunction models ----------------------------------------------

class Trainer:
    def __init__(self, autoencoder, nn_function):
        self.autoencoder = autoencoder
        self.nn_function = nn_function
        self.loss_fn = tf.keras.losses.BinaryCrossentropy() if autoencoder.binary_input else tf.keras.losses.SparseCategoricalCrossentropy()
        self.mean_loss = tf.keras.metrics.Mean()

    def MINE(self, scores):
        def marg(x):
            batch_size = x.shape[0]
            marg_ = tf.reduce_mean(tf.exp(x - tf.linalg.tensor_diag(np.inf * tf.ones(batch_size))))
            return marg_ * ((batch_size * batch_size) / (batch_size * (batch_size - 1.)))

        joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
        marg_term = marg(scores)
        return joint_term - tf.math.log(marg_term)
    
    def plot_loss(self, step, epoch, mean_loss, X_batch, y_pred, plot_encoding):
        template = 'Iteration: {}, Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
        if step % 10 == 0:
            print(template.format(step, epoch, mean_loss.result(), self.autoencoder.B_Ber_m(X_batch, y_pred)))
            if plot_encoding:
                self.autoencoder.test_encoding()

    def plot_batch_loss(self, epoch, mean_loss, X_batch, y_pred):
        template_outer_loop = 'Interim result for Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
        print(template_outer_loop.format(epoch, mean_loss.result(), self.autoencoder.B_Ber_m(X_batch, y_pred)))
    

# --------------------------------------------------------------------------------- TRAINING METHODS --------------------------------------------------------------------------------------------

    def train_mi(self, n_epochs=5, n_steps=20, batch_size=200, learning_rate=0.005):
        optimizer_mi = tf.keras.optimizers.Nadam(learning_rate=learning_rate)                                                           # Initialize optimizer with specified learning rate
        for epoch in range(1, n_epochs + 1):
            print("Training in Epoch {}/{}".format(epoch, n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = self.autoencoder.random_sample(batch_size)                                                                    # Generate a batch of random samples
                with tf.GradientTape() as tape:                                                                                         # Compute gradients using a gradient tape
                    x_enc = self.autoencoder.encoder(X_batch, training=True)                                                            # Encode the batch of samples
                    y_recv = self.autoencoder.channel(x_enc)                                                                            # Pass the encoded samples through the channel
                    x = tf.reshape(x_enc, shape=[batch_size, 2 * self.autoencoder.n])                                                   # Reshape tensors for mutual information estimation
                    y = tf.reshape(y_recv, shape=[batch_size, 2 * self.autoencoder.n])
                    score = self.nn_function(x, y)                                                                                      # Compute mutual information score
                    loss = -self.MINE(score)                                                                                            # Compute loss as negative MINE score
                    gradients = tape.gradient(loss, self.nn_function.trainable_variables)                                               # Compute gradients with respect to NNFunction variables
                    optimizer_mi.apply_gradients(zip(gradients, self.nn_function.trainable_variables))                                  # Apply gradients to update NNFunction weights
                mi_avg = -self.mean_loss(loss)                                                                                          # Average mutual information loss over the steps
            print('Epoch: {}, Mi is {}'.format(epoch, mi_avg))
            self.mean_loss.reset_state()                                                                                                # Reset the mean loss metric for the next epoch
    
    def train_decoder(self, n_epochs=5, n_steps=20, batch_size=200, learning_rate=0.005, plot_encoding=True):
        optimizer_ae = tf.keras.optimizers.Nadam(learning_rate=learning_rate)                                                           # Initialize optimizer with specified learning rate
        for epoch in range(1, n_epochs + 1):
            print("Training Bob in Epoch {}/{}".format(epoch, n_epochs))
            for step in range(1, n_steps + 1):
                X_batch = self.autoencoder.random_sample(batch_size)                                                                    # Generate a batch of random samples
                with tf.GradientTape() as tape:                                                                                         # Compute gradients using a gradient tape
                    y_pred = self.autoencoder.autoencoder(X_batch, training=True)                                                       # Pass the batch through the autoencoder
                    loss = tf.reduce_mean(self.loss_fn(X_batch, y_pred))                                                                # Compute loss using the loss function
                    gradients = tape.gradient(loss, self.autoencoder.decoder.trainable_variables)                                       # Compute gradients with respect to decoder variables
                    optimizer_ae.apply_gradients(zip(gradients, self.autoencoder.decoder.trainable_variables))                          # Apply gradients to update decoder weights
                self.mean_loss(loss)                                                                                                    # Update the mean loss metric
                self.plot_loss(step, epoch, self.mean_loss, X_batch, y_pred, plot_encoding)                                             # Plot the current loss and encoding performance if required
            self.plot_batch_loss(epoch, self.mean_loss, X_batch, y_pred)                                                                # Plot the batch loss for the current epoch
            self.mean_loss.reset_state()                                                                                                # Reset the mean loss metric for the next epoch
    