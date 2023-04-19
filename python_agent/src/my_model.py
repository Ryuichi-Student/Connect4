from tensorflow.keras.optimizers import legacy as legacy_optimizers
import numpy as np
# import cProfile
# from pstats import Stats
from time import sleep
from tensorflow.keras import backend, layers, models, optimizers, losses, regularizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime
import os

# disable_eager_execution()

# CALLBACKS
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')


def create_model(input_shape=(6, 7, 2)):
    l2_reg = 1e-4

    inputs = layers.Input(shape=input_shape)

    # First block
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual tower
    for _ in range(6):
        residual = x
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)

    # Policy head
    policy = layers.Conv2D(2, (1, 1), use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.ReLU()(policy)
    policy = layers.Flatten()(policy)
    # policy = layers.Dense(256, activation='relu')(policy)
    policy = layers.Dense(7, activation='softmax', name='policy')(policy)

    # Value head
    value = layers.Conv2D(1, (1, 1), use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    value = layers.BatchNormalization()(value)
    value = layers.ReLU()(value)
    value = layers.Flatten()(value)
    # value = layers.Dense(256, activation='relu')(value)
    value = layers.Dense(1, activation='tanh', name='value')(value)

    # Create the final model
    final_model = models.Model(inputs=inputs, outputs=[policy, value])

    # Define loss functions and optimizer
    losses_list = {'policy': losses.CategoricalCrossentropy(from_logits=False),
                   'value': losses.MeanSquaredError()}

    # Change if not on Mac.
    optimizer = legacy_optimizers.Adam(learning_rate=0.0001)

    # Compile the model
    final_model.compile(optimizer=optimizer, loss=losses_list,
                        metrics={'policy': 'accuracy', 'value': 'mean_absolute_error'})

    return final_model


class Connect4Model:
    def __init__(self, model_path=None):
        self.model = create_model()
        if model_path is not None:
            self.model.load_weights(model_path)

        self.predict_called = 0

    def predict(self, board):
        self.predict_called += 1
        board = np.stack((np.where(board == 1, 1, 0),
                               np.where(board == -1, 1, 0)),
                               axis=-1)
        input_board = board.reshape((1, 6, 7, 2))

        # pr = cProfile.Profile()
        # pr.enable()
        # with tf.device('/cpu:0'):
        #     action_probs, value = self.model.predict(board, verbose=0)
        action_probs, value = self.model.predict(input_board, verbose=0)
        # action_probs, value = self.model(board, training=False)

        # pr.disable()
        # Stats(pr).sort_stats('tottime').print_stats(10)
        # sleep(10)

        return action_probs[0], value[0][0]

    def batched_predict(self, boards, threads=0, processes=0):
        # Check this on a non-Mac machine.
        if threads*processes != 0:
            # For some reason, this is needed to prevent the model from crashing the ParallelMCTS version.
            # Maybe this is a Mac thing?
            # MPSGraphExecutable.mm:630: failed assertion
            # `Error: Input feed tensor not found in placeholders, tensor corresponds to operation: mps_placeholder_1
            sleep(0.0015*threads*processes)
        current_player_boards = np.where(boards == 1, 1, 0)
        opponent_boards = np.where(boards == -1, 1, 0)
        input_boards = np.stack((current_player_boards, opponent_boards), axis=-1)  # Shape: (N, 6, 7, 2)
        action_probs, values = self.model.predict(input_boards, verbose=0)
        self.predict_called += 1

        return action_probs, values

    def fit(self, inputs, outputs, epochs, batch_size, learning_rate):
        current_player_boards = np.where(inputs == 1, 1, 0)
        opponent_boards = np.where(inputs == -1, 1, 0)
        inputs_reshaped = np.stack((current_player_boards, opponent_boards), axis=-1)  # Shape: (N, 6, 7, 2)
        print("inputs_reshaped.shape", inputs_reshaped.shape)

        backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        self.model.fit(inputs_reshaped, outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                       callbacks=[early_stopping, tensorboard_callback])

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, model_path):
        print("Loading model from", model_path)
        self.model.load_weights(model_path)

