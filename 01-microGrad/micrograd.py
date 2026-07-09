# Micrograd Value class from scratch

class Value:

    # Constructor
    def __init__(self, data):
        self.data = data # the actual number this node holds

    def __repr__(self):
        return f'Value(data={self.data})' # for clean printng

    def __add__(self, other):
        # Called automatically when we add (a + b) a and b's are Values (obj)
        # Importantly our __add__ logic, unpack the .data from self and other and compute(addition) both then again repack as value
        # must wrap result back as a Value (obj)
        out  = Value(self.data + other.data)
        return out
    
    def __mul__(self, other):
        # Same as __add__()
        out = Value(self.data * other.data)
        return out