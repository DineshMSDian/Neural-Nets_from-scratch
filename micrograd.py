# Micrograd Value class from scratch

class Value:

    def __init__(self, data, children=(), _op=''):
        self.data = data
        self._prev = set(children) # Previous nodes that created me.

    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+') # Value(data, _children) This new Value was created from these two parent nodes (self = a and other = b).
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*') # Value(data, _children) This new Value was created from these two parent nodes (self = a and other = b).)
        return out
    
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a*b + c
print(d)
print(c._prev)