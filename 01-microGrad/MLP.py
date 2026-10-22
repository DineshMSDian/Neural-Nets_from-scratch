from typing import Any

from micrograd import Value
import random

class Neuron:
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Any:
        weighted_sum = sum([(wi*xi) for wi, xi in zip(self.w, x)], self.b)
        out_with_act_fun = weighted_sum.tanh()

        return out_with_act_fun


if __name__ == '__main__':
    n = Neuron(2)
    x = [2.0, 3.0]
    out = n(x)
    print(out.data)