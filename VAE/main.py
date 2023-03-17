import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

class Vae:
    def __init__(self, input_data_np=None,
        train_data=None, val_data=None,
        train_y=None, val_y=None,
        seed=42, batch_normalization=False,
        optimizer='adamax', learning_rate=0.0005, beta_1=0.9, beta_2=0.999,
        train_percent=75, sample_method='normal', regularization_weight=1):

        self.seed = seed
        self.learning_rate = learning_rate
        self.input_data_np = input_data_np
        self.train_data = train_data
        self.val_data = val_data
        if train_y is None:
            self.train_y = np.concatenate([np.ones(shape=train_data.shape[0]).reshape(-1, 1), train_data], axis=1)
        else:
            self.train_y = train_y
        if val_y is None:
            self.val_y = np.concatenate([np.ones(shape=val_data.shape[0]).reshape(-1, 1), val_data], axis=1)
        else:
            self.val_y = val_y
        self.batch_normalization = batch_normalization
        self.sample_method = sample_method

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        if train_data is not None:
            self.sample_size = None
            _, self.input_size = train_data.shape
        elif input_data_np is not None:
            self.sample_size, self.input_size = input_data_np.shape #FIXME
            self.train_test_split()
        else:
            raise ValueError('Either train_data or input_data_np should be provided')

        self.regularization_weight = regularization_weight
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.optimizer = optimizer
        self.train_percent = train_percent

        self.vae = None
        self.latent_z = self.latent_z_mean = self.latent_z_log_sigma = None
        self.inputs_x = self.encoding = self.decoding = self.decoder = self.encoder = None

    def add_layer(self, num_nodes: int, prev_layer, activation_fn='selu'):
        x = Dense(num_nodes, activation=activation_fn)(prev_layer)
        if self.batch_normalization:
            x = BatchNormalization()(x)
        return x

    def sample_z(self, args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
            i.e. instead of sampling from Q(z|X), sample epsilon = N(0,I)
            z = z_mean + sqrt(var) * epsilon
        from https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
            Arguments args (tensor): mean and log of variance of Q(z|X)
            Returns z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        if self.sample_method == 'normal':
            epsilon = K.random_normal(shape=(batch, dim), seed=self.seed)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build_model(self):
        # latent dimmension and last layer activation function
        latent_dim = 16
        output_activation_fn = 'sigmoid' # was hardcoded into the code

        # Convert the inputs into our first encoding layer
        self.inputs_x = Input(shape=(self.input_size,), name='input')
        # ------------ Encoding Layer -----------------
        self.encoding = self.inputs_x

        num_nodes1 = 64
        activation1 = 'relu'
        self.encoding = self.add_layer(num_nodes1, self.encoding, activation1)

        num_nodes2 = 32
        activation2 = 'selu'
        self.encoding = self.add_layer(num_nodes2, self.encoding, activation2)
        # ------------ Embedding Layer --------------
        self.latent_z_mean = Dense(latent_dim, name='z_mean')(self.encoding)
        self.latent_z_log_sigma = Dense(latent_dim, name='z_log_sigma',
                                        kernel_initializer='zeros')(self.encoding)
        self.latent_z = Lambda(self.sample_z,
            output_shape=(latent_dim,), name='z')(
                [self.latent_z_mean, self.latent_z_log_sigma])

        # Initialise the encoder
        self.encoder = Model(self.inputs_x,
            [self.latent_z_mean, self.latent_z_log_sigma, self.latent_z],
            name='encoder')
        print(self.encoder.summary())

        # Build the decoder network
        # ------------ Dense out -----------------
        self.latent_inputs = Input(shape=(latent_dim), name='z_sampling')
        self.decoding = self.latent_inputs
        # Same num nodes and activation function from decoder step
        self.decoding = self.add_layer(num_nodes2, self.decoding, activation2)
        self.decoding = self.add_layer(num_nodes1, self.decoding, activation1)
        # add last layer
        self.decoding = self.add_layer(self.input_size + 1, self.decoding, output_activation_fn)
        # Initialise the decoder
        self.decoder = Model(self.latent_inputs, self.decoding, name='decoder')
        print(self.decoder.summary())

        # ------------ Out -----------------------
        self.outputs_y = self.decoder(self.encoder(self.inputs_x)[2])
        self.vae = Model(self.inputs_x, self.outputs_y, name='VAE')
    
    def compile(self):
        if self.optimizer == 'adamax':
            opt = optimizers.Adamax(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        else:
            raise ValueError
        loss = self.get_loss(self.inputs_x, self.outputs_y, self.latent_z,
                    self.latent_z_mean, self.latent_z_log_sigma)
        self.vae.compile(optimizer=opt, loss=loss, weighted_metrics=[])
    
    def predict(self, arr=None):
        if arr is None:
            arr = self.input_data_np
        return self.encoder.predict(arr)

    def train_test_split(self):
        # FIXME
        train_split = round(self.train_percent / 100 * self.sample_size)
        indices = np.random.permutation(self.sample_size)
        training_idx, test_idx = indices[:train_split], indices[train_split:]
        val_data = self.input_data_np[test_idx, :]
        self.val_data = (val_data, val_data, np.ones(len(test_idx)))
        self.train_data = self.input_data_np[training_idx, :]


    def fit(self, patience=3, **kwargs):
        if self.vae is None:
            self.build_model()
            self.compile()
        if patience > 0:
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            self.vae.fit(x=self.train_data, 
                            y=self.train_y,
                            shuffle=True,
                            validation_data=(self.val_data, self.val_y),
                            callbacks=[callback],
                            **kwargs
                        )
        else:
            self.vae.fit(x=self.train_data,
                y=self.train_y,
                shuffle=True,
                validation_data=(self.val_data, self.val_y),
                **kwargs
            )

    def get_loss(self, inputs_x, outputs_y, latent_z, latent_z_mean, latent_z_log_sigma):
        regularization_loss = self.get_mmd_distance(latent_z)
        #regularization_loss = self.get_kl_distance(latent_z_mean, latent_z_log_sigma)
        reconstruction_loss = get_mean_squared_error_loss(inputs_x, outputs_y)
        # add mmd loss and true loss
        return K.mean(reconstruction_loss + (self.regularization_weight * regularization_loss))

    @staticmethod
    def get_kl_distance(z_mean, z_log_sigma):
        """ Resources:
        https://keras.io/examples/generative/vae/
        https://github.com/geyang/variational_autoencoder_pytorch
        """
        # KL regularizer. this is the KL of q(z|x) given that the target distribution is N(0,1)
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return kl_loss

    def get_mmd_distance(self, latent_z):
        """
        https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
        https://learning.mpi-sws.org/mlss2016/slides/cadiz16_2.pdf
        http://abdulfatir.com/Implicit-Reparameterization/
        Returns
        -------
        new loss
        """
        batch_size = K.shape(latent_z)[0]
        latent_dim = K.int_shape(latent_z)[1]

        # Randomly sample from a normal distribution
        true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
        # Compute loss between the training latent space and normal data
        x_kernel = compute_kernel(true_samples, true_samples)
        y_kernel = compute_kernel(latent_z, latent_z)
        xy_kernel = compute_kernel(true_samples, latent_z)
        return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)


def compute_kernel(x, y):
    """
    https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python
    """
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))


def get_mean_squared_error_loss(input_x, output_y):
    """ https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/common.py """
    # weights = output_y[0:1, :-1]
    return K.sum(K.square(input_x - output_y[:, 1:]), axis=1)
