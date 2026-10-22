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

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x) -> Any:
        outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts) -> None:
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
if __name__ == '__main__':
    # n = Neuron(2)
    # x = [2.0, 3.0]
    # out = n(x)
    # print(out.data)

    # layer = Layer(2, 3)

    # x = [2.0, 3.0]
    # outputs = layer(x)
    # print(layer.neuron[0].w)

    n = MLP(3, [4, 4, 1])
    x = [2.0, -3.0, 2.1]
    out = n(x)
    print(len(n.layers))
    print(out.data)
    print(len(n.parameters()))

    # x = [2.0, 3.0, -1.0]
    # network = MLP(3, [4, 4, 1])
    # out = network(x)
    # print(len(network.layers))
    