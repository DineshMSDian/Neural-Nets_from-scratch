import math

# Micrograd Value class from scratch

class Value:

    # Constructor that holds, what are the informations that a Value obj can remembers
    def __init__(self, data, _children=(), _op='',label=''):
        """
        _children defaults to an empty tuple () because most Values start life as plain leaf inputs like a = Value(2.0), 
        which weren't "made" from anything, so they have zero parents.
        set(()) → empty set, meaning "no parents, I'm a leaf."
        __add__ and __mul__ don't rely on this default — they explicitly pass (self, other) as _children, 
        since a result like a+b genuinely does have two parents.
        """

        self.data = data                # the actual number this node holds
        self._backward = lambda: None  # default: "do nothing", true for leaf nodes, gets the fuction from operators or activation func()
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

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward # it stores the func() only, it doesnt not calling
        return out
    
    def __mul__(self, other):
        # Same unpack->compute->repack pattern as __add__, just * instead of +
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward # it stores the func() only, it doesnt not calling
        return out

    # to brak the tanh and build it from ground up, we need
    # tanh(x) = e^2x -1 / e^2x +1
    # 1. we need multiply x by 2
    # 2. then exponent
    # 3. subtraction
    # 4. divide the by denominator 
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            # the derivative of euler's number e^x is exactly e^x which is (out.data)
            # we multiply this local derivative (out.data) with the incomming derivative (out.grad)
            self.grad = out.data * out.grad

        out._backward = _backward
        return out

    # To negate a number (e.g., -x), we just multiply it by -1 using our __mul__ primitive.
    def __neg__(self):
        return self * -1
        
    # To subtract (e.g., a - b), we convert it to addition: a + (-b).
    # This completely reuses the __add__ and __neg__ code! 
    def __sub__(self, other):
        return self + (-other)
    
     # Like subtraction, division (e.g., a / b) is broken down using existing primitives by rewriting it as a multiplication: (a * (b^-1)).
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1)

    # To make other**-1 work, we use a basic Power primitive:
    def __pow__(self, other):
        assert isinstance(other, (int, float))

        # Forward pass: x^other
        out = Value(self**other, (self,), f'**{other}')

        # Derivative rule for x^n is n * x^(n-1) (The Power Rule)
        def _backward():
            # Local derivative (other * x^(other-1)) * upstream gradient
            self.grad += (other * (self.data ** (other -1))) * out.grad

        out._backward = _backward
        return out

    # tanh for neurons
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) -1) / (math.exp(2*x) +1) # exp stands for exponential. computes e power x, where e = 2.718.. is euler's number
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward # it stores the func() only, it doesnt not calling
        return out   

    def backward(self):
        """
            Compute gradients for every Value node that contributes to self.

            Step-by-step execution:
            1. Initialize an empty list `topo` to store nodes in topological order.
            2. Initialize an empty set `visited` to prevent visiting the same node
            multiple times.
            3. Define the recursive function `build_topo(v)`:
            a. If v has not been visited, mark it as visited.
            b. Recursively visit every parent in v._prev.
            c. Append v to topo only after all its parents have been explored.
            4. Call build_topo(self) to build the computational graph's topological
            ordering, starting from the output node.
            5. Set self.grad = 1.0 because d(self)/d(self) = 1.
            6. Iterate through topo in reverse order:
            a. Call node._backward() to propagate the node's gradient to its
                parents using the local derivatives and the chain rule.
            b. Each node's _backward function should use += to accumulate
                gradients when multiple paths contribute to the same node.

            Why reverse the topological order?
            - build_topo visits parents before appending the current node.
            - Therefore, the output node is appended after its ancestors.
            - Reversing topo processes the output first, then propagates gradients
            backward through the graph toward the input nodes.

            Example:
                a = Value(3.0, label='a')
                b = a + a
                b.backward()

                b.grad == 1.0
                a.grad == 2.0

            Note:
            - This method assumes gradients are initialized appropriately.
            - Calling backward() repeatedly on the same graph can accumulate
            gradients unless they are reset first.
        """
        topo = [] # for or backprobagation, we use topological sort using dfs
        visited = set() 

        def build_topo(v):
            if v not in visited: # dfs
                visited.add(v)

                for child in v._prev:
                    build_topo(child)

                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()