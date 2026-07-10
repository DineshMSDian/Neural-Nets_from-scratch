import math

# Micrograd Value class from scratch

class Value:

    # Constructor that holds, what are the informations that a Value obj can remembers
    def __init__(self, data, _children=(), _op='',label=''):
        # _children defaults to an empty tuple () because most Values start life
        # as plain leaf inputs like a = Value(2.0), which weren't "made" from
        # anything, so they have zero parents.
        # set(()) → empty set, meaning "no parents, I'm a leaf."
        # __add__ and __mul__ don't rely on this default — they explicitly pass
        # (self, other) as _children, since a result like a+b genuinely does have two parents.

        self.data = data                # the actual number this node holds
        self._prev = set(_children)   # parent Value(s) that combined to create me
        self._op = _op                  # which operator (+ or *) combined them 
        self.grad = 0.0               # grad = sensitivity, how much o/p changes per tiny change in i/p
        self.label = label              # optional name for readability, e.g. 'a', 'd

    def __repr__(self):
        return f'Value(data={self.data})' # for clean printng

    def __add__(self, other):
        # Called automatically when we add (a + b) a and b's are Values (obj)
        # Importantly our __add__ logic, unpack the .data from self and other and compute(addition) both then again repack as value
        # must wrap result back as a Value (obj)

        out  = Value(self.data + other.data, (self, other), '+') # returns the new obj with info contains, 1. computed Value, 2. Parents, operator
        return out
    
    def __mul__(self, other):
        # Same unpack->compute->repack pattern as __add__, just * instead of +
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
    # tanh for neurons
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1) / (math.exp(2*x) +1) # exp stands for exponential. computes e power x, where e = 2.718.. is euler's number
        out = Value(t, (self, ), 'tanh')
        return out