import numpy as np
import json
import os

class OnlineLogisticRegression:
    def __init__(self, n_features, lr=0.1, l2=0.0):
        self.n_features = n_features
        self.lr = lr
        self.l2 = l2
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

    def predict_proba(self, X):
        z = np.dot(X, self.coef_) + self.intercept_
        return 1 / (1 + np.exp(-z))

    def update(self, X, y):
        p = self.predict_proba(X)
        error = y - p
        self.coef_ += self.lr * (error * X - self.l2 * self.coef_)
        self.intercept_ += self.lr * error

    def save(self, path):
        data = {
            'coef_': self.coef_.tolist(),
            'intercept_': self.intercept_,
            'n_features': self.n_features,
            'lr': self.lr,
            'l2': self.l2
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'r') as f:
            data = json.load(f)
        model = cls(data['n_features'], data['lr'], data['l2'])
        model.coef_ = np.array(data['coef_'])
        model.intercept_ = data['intercept_']
        return model
