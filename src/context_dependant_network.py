import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from CPC18_features import CPC18_getDist

_lot_shape_dict = {
    0: "-",
    1: "Symm",
    2: "L-skew",
    3: "R-skew",
    "-": "-",
    "Symm": "Symm",
    "L-skew": "L-skew",
    "R-skew": "R-skew"
}

def build_cd_input_format(classic_format):
    N = len(classic_format)
    gambles = np.zeros((N,24))
    if isinstance(classic_format, pd.DataFrame):
        for i in range(N):
            g = classic_format.iloc[i]
            a_payoff_dist = CPC18_getDist(g.Ha, g.pHa, g.La, "-", 0).ravel()
            gambles[i, 0:len(a_payoff_dist)] = a_payoff_dist
            b_payoff_dist = CPC18_getDist(
                g.Hb, g.pHb, g.Lb, _lot_shape_dict[g.LotShapeB], int(g.LotNumB)
            ).ravel()
            gambles[i, 4:4+len(b_payoff_dist)] = b_payoff_dist
    elif isinstance(classic_format, np.ndarray):
        assert classic_format.shape[1] == 12
        for i in range(N):
            g = classic_format[i, :]
            a_payoff_dist = CPC18_getDist(g[0], g[1], g[2], "-", 0).ravel()
            gambles[i, 0:len(a_payoff_dist)] = a_payoff_dist
            b_payoff_dist = CPC18_getDist(
                g[3], g[4], g[5], _lot_shape_dict[g[7]], int(g[6])
            ).ravel()
            gambles[i, 4:4+len(b_payoff_dist)] = b_payoff_dist
    else:
        raise ValueError("Input can only be a numpy array or pandas DataFrame")
    return gambles

class ContextDependantNetwork:

    def __init__(self):
        self.model = Sequential()
        self.model.add(Input(24,))
        self.model.add(Dense(32, activation="sigmoid"))
        self.model.add(Dense(32, activation="sigmoid"))
        self.model.add(Dense(1, activation="sigmoid"))

    def train(self, X_train, y_train, **kwargs):
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        return self.model.fit(
            x=X_train,
            y=y_train,
            **kwargs
        )

    def save(self, save_path):
        weights = dict()
        for i,l in enumerate(self.model.layers):
            w = l.get_weights()
            weights[f"w{i}"] = w
        pickle.dump(weights, open(save_path, "wb+"))

    def load(self, path):
        weights = pickle.load(open(path, "rb"))
        for k,v in weights.items():
            lay_num = int(k[-1])
            self.model.layers[lay_num].set_weights(v)

    def predict(self, gambles):
        if gambles.shape[1] == 12:
            return self.model.predict(build_cd_input_format(gambles))
        elif gambles.shape[1] == 24:
            return self.model.predict(gambles)
        else:
            raise ValueError("Gambles need to be either in classical format 12 \
                             features or in Peterson format 24 features")

    def evaluate(self, gambles, rates):
        rates_pred = self.predict(gambles).ravel()
        return np.mean((rates_pred - rates.ravel()) ** 2) * 100
