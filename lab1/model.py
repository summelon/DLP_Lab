import numpy as np
import utils


class Linear:
    def __init__(self, in_features: int, out_features: int, bias=False):
        self.weight = np.random.normal(size=(in_features, out_features))
        self.weight_grad = 0
        if bias:
            self.bias = np.random.normal(size=out_features)
            self.bias_grad = 0
        else:
            self.bias = None

    def forward(self, inp: np.ndarray):
        return np.dot(inp, self.weight) + \
                (0 if self.bias is None else self.bias)

    # TODO Check if transpose is necessary
    def backward(self, inp: np.ndarray, prev_grad: np.ndarray):
        if self.bias is None:
            return np.dot(inp.T, prev_grad)
        else:
            return np.dot(inp, prev_grad), np.dot(np.ones(len(inp)), prev_grad)

    def update(self, lr: float):
        self.weight -= lr * self.weight_grad
        if self.bias is not None:
            self.bias -= lr * self.bias_grad


class MyModel:
    def __init__(self, in_features: int, out_features: int, hid_dim: list):
        self.linear_1 = Linear(in_features, hid_dim[0])
        self.linear_2 = Linear(hid_dim[0], out_features)
        self.params = dict()

    def forward(self, inp: np.ndarray):
        self.params = dict()
        self.params['Z1'] = self.linear_1.forward(inp)
        self.params['A1'] = self.sigmoid(self.params['Z1'])
        self.params['Z2'] = self.linear_2.forward(self.params['A1'])
        self.params['A2'] = self.sigmoid(self.params['Z2'])

        return self.params['A2']

    def backward(self, inp: np.ndarray, otpt: np.ndarray, gd: np.ndarray):
        w2_prev_grad = np.multiply(
                self.mse(otpt, gd, derivative=True),
                self.sigmoid(self.params['Z2'], derivative=True))
        self.linear_2.weight_grad = self.linear_2.backward(
                self.params['A1'], w2_prev_grad)

        w1_prev_grad = np.multiply(
                np.dot(w2_prev_grad, self.linear_2.weight.T),
                self.sigmoid(self.params['Z1'], derivative=True))
        self.linear_1.weight_grad = self.linear_1.backward(inp, w1_prev_grad)

    def update(self, lr=1e-3):
        self.linear_1.update(lr)
        self.linear_2.update(lr)

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
    def mse(self, otpt: np.ndarray, gd: np.ndarray, derivative=False):
        if derivative:
            return 2 * (otpt-gd)
        else:
            return 0.5 * ((gd-otpt)**2 + (otpt-gd)**2)

    def pred(self, otpt):
        assert all(e <= 1 and e >= 0 for e in otpt), "Value(s) is wrong!"
        return np.array([[1] if e > 0.5 else [0] for e in otpt])

    def cal_acc(self, pred: np.ndarray, gd: np.ndarray):
        assert pred.shape == gd.shape,\
            "The shape of predictions is different with ground truth."
        return np.equal(pred, gd).sum() / len(gd)


def main():
    model = MyModel(in_features=2, out_features=1, hid_dim=[8])
    # inputs, labels = utils.generate_linear()
    inputs, labels = utils.generate_xor_easy()

    for e in range(100000):
        outputs = model.forward(inputs)
        model.backward(inputs, outputs, labels)
        model.update()
        if e % 1000 == 0:
            preds = model.pred(outputs)
            loss = model.mse(outputs, labels).mean()
            print(f"[{e:^7d}]Accuracy now is "
                  f"{model.cal_acc(preds, labels):.2%}"
                  f", Loss now is {loss:.2f}",)


if __name__ == "__main__":
    main()
