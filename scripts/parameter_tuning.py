import numpy as np
import tensorflow as tf


X_aug = np.load('../data/X_aug.npy')
y_aug = np.load('../data/y_aug.npy')

from sklearn.model_selection import train_test_split

X_aug_train_val, X_aug_test, y_aug_train_val, y_aug_test = train_test_split(X_aug, y_aug, test_size=0.07, random_state=42)

X_aug_train, X_aug_val, y_aug_train, y_aug_val = train_test_split(X_aug_train_val, y_aug_train_val, test_size=0.14, random_state=42)

from keras.optimizers import Adam
class model_init:
    def __init__(self, classes, input_shape, num_dense_blocks, num_dense_units, num_conv_blocks, filters, kernel_size, pool_size, dropout_rate, padding, regularizer_strength, conv_activation, dense_activation):
        """
        Initialize the model with the given parameters
        """
        self.classes = classes
        self.input_shape = input_shape
        self.num_dense_blocks = num_dense_blocks
        self.num_dense_units = num_dense_units
        self.num_conv_blocks = num_conv_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.regularizer_strength = regularizer_strength
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation


        self.model = self.build_model()

    def conv_block(self, input, filters, kernel_size, pool_size):
        """
        Convolutional block with Conv2D, Activation and MaxPooling2D layers
        """
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=self.padding, activation=self.conv_activation)(input)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding=self.padding, activation=self.conv_activation)(x)
        x = tf.keras.layers.MaxPooling2D(pool_size)(x)
        return x
    
    def dense_block(self, input, units, dropout):
        """
        Dense block with Dense, Activation and Dropout layers
        """
        x = tf.keras.layers.Dense(units, activation=self.dense_activation, kernel_regularizer=tf.keras.regularizers.l2(self.regularizer_strength))(input)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x
    
    def build_model(self):
        """
        Build the model with the given parameters
        """
        input = tf.keras.layers.Input(shape=self.input_shape)

        x = self.conv_block(input, self.filters, self.kernel_size, self.pool_size)
        for i in range(self.num_conv_blocks - 1):
            if x.shape[1] < 10 or x.shape[2] < 10:
                break
            x = self.conv_block(x, self.filters, self.kernel_size, self.pool_size)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        x = self.conv_block(input, self.filters, self.kernel_size, self.pool_size)
        for i in range(self.num_conv_blocks - 1):
            x = self.conv_block(x, self.filters, self.kernel_size, self.pool_size)

        x = tf.keras.layers.Flatten()(x)
        
        for i in range(self.num_dense_blocks):
            x = self.dense_block(x, self.num_dense_units, self.dropout_rate)
        output = tf.keras.layers.Dense(self.classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=input, outputs=output)
        return model

from keras_tuner.tuners import RandomSearch, BayesianOptimization
from keras_tuner.engine.hyperparameters import HyperParameters

def create_model(hp):
        num_dense_blocks = hp.Int('num_dense_blocks', min_value=2, max_value=4, step=1)
        num_dense_units = hp.Int('num_dense_units', min_value=160, max_value=512, step=32)
        num_conv_blocks = hp.Int('num_conv_blocks', min_value=1, max_value=3, step=1)
        filters = hp.Int('filters', min_value=64, max_value=256, step=32)
        kernel_size = hp.Int('kernel_size', min_value=3, max_value=5, step=1)
        pool_size = hp.Int('pool_size', min_value=2, max_value=4, step=1)
        dropout_rate = hp.Float('dropout_rate', min_value=0.4, max_value=0.7, step=0.1)
        padding = "valid"
        regularizer_strength = hp.Float('regularizer_strength', min_value=0.03, max_value=0.08, step=0.01)
        conv_activation = hp.Choice('conv_activation', values=['relu', 'elu'])
        dense_activation = hp.Choice('dense_activation', values=['relu', 'elu'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG')
        optimizer = Adam(learning_rate=learning_rate)

        model = model_init(6, input_shape=(X_aug_train[0].shape[0], X_aug_train[0].shape[1], 1), num_dense_blocks=num_dense_blocks, num_dense_units=num_dense_units, num_conv_blocks=num_conv_blocks, filters=filters, kernel_size=kernel_size, pool_size=pool_size, dropout_rate=dropout_rate, padding=padding, regularizer_strength=regularizer_strength, conv_activation=conv_activation, dense_activation=dense_activation)

        model = model.build_model()

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['precision', 'accuracy'])

        return model

tuner = BayesianOptimization(
    create_model,
    objective='val_accuracy',
    max_trials=25,
    executions_per_trial=1,
    directory='bayes_directory'
)

tuner.search(X_aug_train, y_aug_train, epochs=12, validation_data=(X_aug_val, y_aug_val), batch_size=16)

print(tuner.results_summary())

print(tuner.get_best_hyperparameters()[0].values)