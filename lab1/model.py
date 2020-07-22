import copy
import numpy as np
from argparse import ArgumentParser

import utils


class Linear:
    def __init__(self, in_features: int, out_features: int, bias=False):
        self.weight = np.random.normal(size=(in_features, out_features))
        self.weight_ahead = copy.deepcopy(self.weight)

        self.weight_grad = np.zeros_like(self.weight)
        self.velocity = copy.deepcopy(self.weight_grad)
        if bias:
            self.bias = np.random.normal(size=out_features)
            self.bias_grad = 0
        else:
            self.bias = None

    def forward(self, inp: np.ndarray):
        return np.dot(inp, self.weight) + \
                (0 if self.bias is None else self.bias)

    def backward(self, inp: np.ndarray, prev_grad: np.ndarray):
        if self.bias is None:
            return np.dot(inp.T, prev_grad)
        else:
            return np.dot(inp, prev_grad), np.dot(np.ones(len(inp)), prev_grad)

    def update(self, lr: float, gamma: float = 0):
        self.velocity = lr * self.weight_grad + gamma * self.velocity
        self.weight -= self.velocity
        self.weight_ahead = self.weight + gamma * self.velocity
        if self.bias is not None:
            self.bias -= lr * self.bias_grad


class MyModel:
    def __init__(self, in_features: int, out_features: int, hid_dim: list):
        self.linear_1 = Linear(in_features, hid_dim[0])
        self.linear_2 = Linear(hid_dim[0], hid_dim[1])
        self.linear_3 = Linear(hid_dim[1], out_features)
        self.params = dict()

    def forward(self, inp: np.ndarray):
        self.params = dict()
        self.params['Z1'] = self.linear_1.forward(inp)
        self.params['A1'] = self.sigmoid(self.params['Z1'])
        self.params['Z2'] = self.linear_2.forward(self.params['A1'])
        self.params['A2'] = self.sigmoid(self.params['Z2'])
        self.params['Z3'] = self.linear_3.forward(self.params['A2'])
        self.params['A3'] = self.sigmoid(self.params['Z3'])

        return self.params['A3']

    def backward(self, inp: np.ndarray, otpt: np.ndarray, gd: np.ndarray,
                 nesterov: bool = True, loss='ce'):
        if loss == 'ce':
            w3_prev_grad = np.multiply(
                    self.mse(otpt, gd, derivative=True),
                    self.sigmoid(self.params['Z3'], derivative=True))
        elif loss == 'mse':
            w3_prev_grad = np.multiply(
                    self.ce(otpt, gd, derivative=True),
                    self.sigmoid(self.params['Z3'], derivative=True))
        else:
            raise ValueError(f"No such loss function: {loss}")
        self.linear_3.weight_grad = self.linear_3.backward(
                self.params['A2'], w3_prev_grad)

        w2_prev_grad = np.multiply(
                np.dot(w3_prev_grad,
                       self.linear_3.weight_ahead.T
                       if nesterov else self.linear_3.weight.T),
                self.sigmoid(self.params['Z2'], derivative=True))
        self.linear_2.weight_grad = self.linear_2.backward(
                self.params['A1'], w2_prev_grad)

        w1_prev_grad = np.multiply(
                np.dot(w2_prev_grad,
                       self.linear_2.weight_ahead.T
                       if nesterov else self.linear_2.weight.T),
                self.sigmoid(self.params['Z1'], derivative=True))
        self.linear_1.weight_grad = self.linear_1.backward(
                inp, w1_prev_grad)

    def update(self, lr=1e-3, gamma=9e-1):
        self.linear_1.update(lr, gamma)
        self.linear_2.update(lr, gamma)

    # Activation function
    def sigmoid(self, inp: np.ndarray, derivative=False):
        def y(x):
            return 1. / (1 + np.exp(-x))
        if derivative:
            return np.multiply(y(inp), (1 - y(inp)))
        else:
            return y(inp)

    # Loss function
    # TODO 1.multi output 2.check if output is one-hot encoding
    def mse(self, otpt: np.ndarray, gt: np.ndarray, derivative=False):
        if derivative:
            return 2 * (otpt-gt) / len(otpt)
        else:
            return 0.5 * np.mean(((gt-otpt)**2 + (otpt-gt)**2))

    # Cross entropy
    def ce(self, otpt: np.ndarray, gt: np.ndarray, eps=1e-8, derivative=False):
        if derivative:
            return (((1-gt) / (1-otpt+eps)) - (gt/otpt+eps)) / len(otpt)
        else:
            return -np.mean(gt*np.log(otpt+eps) + (1-gt)*np.log(1-otpt+eps))

    def pred(self, otpt):
        assert all(e <= 1 and e >= 0 for e in otpt), "Value(s) is wrong!"
        return np.array([[1] if e > 0.5 else [0] for e in otpt])

    def cal_acc(self, pred: np.ndarray, gd: np.ndarray):
        assert pred.shape == gd.shape,\
            "The shape of predictions is different with ground truth."
        return np.equal(pred, gd).sum() / len(gd)


def main(params):
    np.random.seed(7)
    model = MyModel(in_features=2, out_features=1, hid_dim=[8, 16])
    if params['data'] == 'linear':
        inputs, labels = utils.generate_linear(n=100)
    elif params['data'] == 'xor':
        inputs, labels = utils.generate_xor_easy()
    else:
        raise KeyError("Please select a correct input data")

    # Start training
    e = 0
    loss_recorder = []
    while(1):
        e += 1
        outputs = model.forward(inputs)
        if params['loss_fn'] == 'ce':
            model.backward(inputs, outputs, labels, nesterov=False, loss='ce')
            loss = model.ce(outputs, labels)
        elif params['loss_fn'] == 'mse':
            model.backward(inputs, outputs, labels, nesterov=False, loss='mse')
            loss = model.mse(outputs, labels)
        else:
            raise ValueError(f"No such loss function: {params['loss_fn']}")
        model.update(lr=params['lr'], gamma=params['momentum'])
        loss_recorder.append(loss)
        if e % params['show_period'] == 0:
            preds = model.pred(outputs)
            print(f"[{e:^7d}]Accuracy now is "
                  f"{model.cal_acc(preds, labels):.2%}"
                  f", Loss now is {loss:.2f}",)
            if loss < 0.01 or model.cal_acc(preds, labels) == 1:
                print("Model has almost fitted the data")
                print(f"Model prediction:\n {outputs}")
                utils.show_result(inputs, labels, preds)
                utils.show_loss_rec(loss_recorder, range(1, e+1))
                break


def param_loader():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Global learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Gamma value in momentum")
    parser.add_argument("--show-period", type=int, default=100,
                        help="Period to show training result")
    parser.add_argument("--data", type=str, choices=['linear', 'xor'],
                        default='linear', help="Input data type")
    parser.add_argument("--loss_fn", type=str, choices=['ce', 'mse'],
                        default='ce', help="Choose your loss function")
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == "__main__":
    p = param_loader()
    main(p)
