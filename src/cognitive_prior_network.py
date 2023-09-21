import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Multiply
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
from echoAI.Activation.Keras import SReLU

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}


class MaskWeights(Constraint):

    def __init__(self, mask):
        self.mask = mask
        self.mask = K.cast(self.mask, K.floatx())

    def __call__(self, w):
        w = self.mask * w
        return w

    def get_config(self):
        return {'mask': self.mask}


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


def createWeightsMask(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    mask_weights = np.random.rand(noRows, noCols)
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights[mask_weights < prob] = 0
    mask_weights[mask_weights >= prob] = 1
    noParameters = np.sum(mask_weights)
    return [noParameters, mask_weights]


class CognitivePriorNetwork():
    X_test = None
    y_test = None

    def __init__(self, input_shape=12, batch_size=100, epsilon=20, zeta=0.3):
        self.input_shape = input_shape
        # set model parameters
        self.batch_size = batch_size  # batch size
        # self.maxepoches = 50 # max number of epochs

        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = zeta  # the fraction of the weights removed

        # generate an Erdos Renyi sparse weights mask for each layer
        [self.noPar1, self.wm1] = createWeightsMask(self.epsilon, self.input_shape, 200)
        [self.noPar2, self.wm2] = createWeightsMask(self.epsilon, 200, 275)
        # Original code did not make sense for this one
        [self.noPar3, self.wm3] = createWeightsMask(self.epsilon, 275, 100)

        # initialize layers weights
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None

        # initialize weights for SReLu activation function
        self.wSRelu1 = None
        self.wSRelu2 = None
        self.wSRelu3 = None

        # create a SET-MLP model
        self.create_model()
    
    def __str__(self):
        return 'Cognitive Prior Network'

    def create_model(self):
        # create a SET-MLP model
        self.model = Sequential()
        self.model.add(Input((self.input_shape,)))  # Input shape correct?
        self.model.add(Dense(200, name="sparse_1", kernel_constraint=MaskWeights(self.wm1), weights=self.w1))
        self.model.add(SReLU(name="srelu1", weights=self.wSRelu1))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(275, name="sparse_2", kernel_constraint=MaskWeights(self.wm2), weights=self.w2))
        self.model.add(SReLU(name="srelu2", weights=self.wSRelu2))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(100, name="sparse_3", kernel_constraint=MaskWeights(self.wm3), weights=self.w3))
        self.model.add(SReLU(name="srelu3", weights=self.wSRelu3))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(1, name="dense_4", weights=self.w4))
        # Please note that there is no need for a sparse output layer as the number of classes is much smaller
        # than the number of input hidden neurons
        self.model.add(Activation('sigmoid'))

    def fit(self, trainingData, outputData, learning_rate=0.001, verbose=2, patience=10, epochs=50, checkpoint=None,
            save_path=None):

        # RMSPRop learning rate
        # Train the SET-MLP model
        if verbose >= 1:
            self.model.summary()

        # training process in a for loop
        self.loss_per_epoch = []

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_model = None
        # The number of epoch it has waited when loss is no longer minimum.
        wait = 0
        # The epoch the training stops at.
        best_epoch = 0
        # Initialize the best as infinity.
        best = np.inf

        log_dir = f"../logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-epsilon_{self.epsilon}-zeta_{self.zeta}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1)

        for epoch in range(0, epochs):
            rmsprop = RMSprop(lr=learning_rate)
            self.model.compile(loss='mean_squared_error', optimizer=rmsprop)

            historytemp = self.model.fit(x=trainingData,
                                         y=outputData,
                                         batch_size=self.batch_size,
                                         epochs=epoch,
                                         validation_data=(self.X_test, self.y_test),
                                         initial_epoch=epoch - 1,
                                         verbose=verbose,
                                         callbacks=[tensorboard_callback])
            current_loss = historytemp.history['val_loss'][0]
            self.loss_per_epoch.append(current_loss)

            if np.less(current_loss, best):
                best = current_loss
                wait = 0
                # Record the best weights if current results is better (less).
                self.best_model = self.get_model_params()
                best_epoch = epoch

            else:
                wait += 1
                if wait >= patience:
                    print("Stopped training at epoch {}.".format(epoch))
                    print("Restoring model weights from the end of the best epoch {}.".format(best_epoch))
                    self.set_model_params(self.best_model)
                    break

            # Ugly hack to avoid tensorflow memory increase for multiple fit_generator calls.
            # We are not using fit_generator, so we might not need to clear the session.
            self.weightsEvolution()
            K.clear_session()
            self.create_model()

        # also reset to best model after last iteration
        print("Restoring model weights from the end of the best epoch {}.".format(best_epoch))
        self.set_model_params(self.best_model)

        self.loss_per_epoch = np.asarray(self.loss_per_epoch)
        if save_path:
            self.save("%s/best_model" % save_path)

    def weightsEvolution(self):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        self.w1 = self.model.get_layer("sparse_1").get_weights()
        self.w2 = self.model.get_layer("sparse_2").get_weights()
        self.w3 = self.model.get_layer("sparse_3").get_weights()
        self.w4 = self.model.get_layer("dense_4").get_weights()

        self.wSRelu1 = self.model.get_layer("srelu1").get_weights()
        self.wSRelu2 = self.model.get_layer("srelu2").get_weights()
        self.wSRelu3 = self.model.get_layer("srelu3").get_weights()

        [self.wm1, self.wm1Core] = self.rewireMask(self.w1[0], self.noPar1)
        [self.wm2, self.wm2Core] = self.rewireMask(self.w2[0], self.noPar2)
        [self.wm3, self.wm3Core] = self.rewireMask(self.w3[0], self.noPar3)

        self.w1[0] = self.w1[0] * self.wm1Core
        self.w2[0] = self.w2[0] * self.wm2Core
        self.w3[0] = self.w3[0] * self.wm3Core

    def rewireMask(self, weights, noWeights):
        # rewire weight matrix

        # remove zeta largest negative and smallest positive weights
        values = np.sort(weights.ravel())
        firstZeroPos = find_first_pos(values, 0)
        lastZeroPos = find_last_pos(values, 0)
        largestNegative = values[int((1 - self.zeta) * firstZeroPos)]
        smallestPositive = values[
            int(min(values.shape[0] - 1, lastZeroPos + self.zeta * (values.shape[0] - lastZeroPos)))]
        rewiredWeights = weights.copy();
        rewiredWeights[rewiredWeights > smallestPositive] = 1;
        rewiredWeights[rewiredWeights < largestNegative] = 1;
        rewiredWeights[rewiredWeights != 1] = 0;
        weightMaskCore = rewiredWeights.copy()

        # add zeta random weights
        nrAdd = 0
        noRewires = noWeights - np.sum(rewiredWeights)
        while (nrAdd < noRewires):
            i = np.random.randint(0, rewiredWeights.shape[0])
            j = np.random.randint(0, rewiredWeights.shape[1])
            if (rewiredWeights[i, j] == 0):
                rewiredWeights[i, j] = 1
                nrAdd += 1

        return [rewiredWeights, weightMaskCore]

    def get_model_params(self, after_pruning=False):
        w1 = self.model.get_layer("sparse_1").get_weights()
        w2 = self.model.get_layer("sparse_2").get_weights()
        w3 = self.model.get_layer("sparse_3").get_weights()
        w4 = self.model.get_layer("dense_4").get_weights()

        wSRelu1 = self.model.get_layer("srelu1").get_weights()
        wSRelu2 = self.model.get_layer("srelu2").get_weights()
        wSRelu3 = self.model.get_layer("srelu3").get_weights()
        if after_pruning:
            [_, wm1Core] = self.rewireMask(w1[0], self.noPar1)
            [_, wm2Core] = self.rewireMask(w2[0], self.noPar2)
            [_, wm3Core] = self.rewireMask(w3[0], self.noPar3)
            masks = [wm1Core, wm2Core, wm3Core]
        else:
            masks = [self.wm1, self.wm2, self.wm3]
        weights_layers = [w1, w2, w3, w4]
        weights_activation = [wSRelu1, wSRelu2, wSRelu3]
        parameter_counts = [self.noPar1, self.noPar2, self.noPar3]
        model = {"masks": masks,
                 "weights_layers": weights_layers,
                 "weights_activation": weights_activation,
                 "parameter_counts": parameter_counts,
                 "input_shape": self.input_shape,
                 "epsilon": self.epsilon,
                 "zeta": self.zeta}
        return model

    def set_model_params(self, model):
        [self.wm1, self.wm2, self.wm3] = model['masks']
        [self.w1, self.w2, self.w3, self.w4] = model['weights_layers']
        [self.wSRelu1, self.wSRelu2, self.wSRelu3] = model['weights_activation']
        [self.noPar1, self.noPar2, self.noPar3] = model['parameter_counts']
        self.input_shape = model['input_shape']
        self.epsilon = model['epsilon']
        self.zeta = model['zeta']
        K.clear_session()
        self.create_model()

    def save(self, filepath):
        model = self.get_model_params()
        pickle.dump(model, open(filepath, "wb+"))

    def load(self, filepath):
        model = pickle.load(open(filepath, "rb"))
        if any(k not in model for k in ("masks",
                                        "weights_layers",
                                        "weights_activation",
                                        "parameter_counts",
                                        "input_shape",
                                        "epsilon",
                                        "zeta")):
            raise ValueError("Not valid model pickle")
        self.set_model_params(model)

    def predict(self, X, batch_size=2000):
        return self.model.predict(X, batch_size=batch_size)

    def evaluate(self, X, y, batch_size=2000):
        y_pred = self.model.predict(X, batch_size=batch_size)
        loss = tf.keras.losses.mean_squared_error(y_pred, y)
        return 100 * np.mean(loss)
