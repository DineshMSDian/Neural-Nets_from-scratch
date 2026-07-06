# Micrograd Value class from scratch

class Value:

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f'Value(data={self.data})'
    
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out
    
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a*b + c

print(d)