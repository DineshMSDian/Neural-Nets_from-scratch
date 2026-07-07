# Micrograd Value class from scratch

class Value:

    def __init__(self, data, children=(), _op='', label = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(children) # Previous nodes that created me.
        self._op = _op
        self.lebel = label

    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+') # Value(data, _children) This new Value was created from these two parent nodes (self = a and other = b).
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*') # Value(data, _children) This new Value was created from these two parent nodes (self = a and other = b).)
        return out
    
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

d = a*b + c; d.label='d'
print(d)
print(c._prev)