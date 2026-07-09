from micrograd import Value
from draw_graph import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a * b; e.label='e'
d = c + e; d.label='d'
f = Value(-2.0, label='f')
l = d * f; l.label='l'

h = 0.001 # tiny change
# manual backprop
l.grad=1.0 # "The sensitivity of the output with respect to itself is exactly 1."
"""
1. dl/dd

l= d * f
dl/dd = ? -> f
cuz: f(x*y) = df/dx = y, df/dy = x
by derivatives:
dl/dd = l(d+h) - l(d)/h new - old -> eqn 1
# since l = d * f

= ((d+h)*f - d*f)/h
extract the () -> d*f + h*f - d*f / h
+ d*f - d*f cancels, then h*f/h, h/h cancels
so, dl/dd = f ; f(x*y) = df/dx = y, df/dy = x
Derived using the definition of the derivative.
"""
# so
f.grad =  4.0
d.grad = -2.0
"""
2. dl/dc
dl/dc = dl/dd * dd/dc -> chain rule
so need to cal dd/dc
d = c + e, f(x+y) = 1.0, 
by applying chain rule 
dl/dc = (-2.0) * 1.0 = -2.0

"""
e.grad = -2.0
c.grad = -2.0

"""
3. dl/da
dl/da = dl/de *  de/da
de/da = ?, e = a * b same as dl/dd f(x*y) = df/dx = y, df/dy = x, a=b data, b=a data
so, dl/da = dl/de * de/da = (-2) * (-3) = 6

dl/db = dl/de * de/db = (-2) * (2)
"""
a.grad = 6.0
b.grad = -4.0

"""
now full backprop dl/da =  dl/dd * dd/dc * de/da
                        = -2.0 * 1.0 * -3.0 = 
"""

dot = draw_dot(l)
dot.render('graph')