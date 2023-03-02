# B-splines and NURBS

## **B-spline**

### **Definition**

A b-spline (or basis-spline) is a parametric curve in $D$ dimensional space is given by the equation
$$
S(t) = \sum_{i=0}^n P_i \, N_{i,p}(t),
$$
where $t$ is the parametric parameter, $(P_0, P_1, \cdots, P_n)$ are a set of $D$ dimensional control point, and $\big(N_{0,p}(t), N_{1,p}(t), ..., N_{n,p}(t)\big)$ are the $p^{th}$ degree basis functions. The basis functions are piece-wise polynomial functions defined over a series of intervals $[t_0, t_1), [t_1, t_2), \cdots, [t_{m-1}, t_m)$. For the polynomials to be well defined over for all $t_0 \le t < t_m$ then we have the condition
$$
t_i \le t_{i+1} \quad \text{for all} \quad i \in [0, 1, \cdots, m-1].
$$
The basis functions are defined recursively, with the $0^{th}$ order basis function the piecewise constant function that is non-zero on only a single interval
$$
N_{i,0} = \begin{cases} 
    1 & \text{for} \quad t_{i} \le t < t_{i+1}, \\
    0 & \text{otherwise},
\end{cases}
$$
and higher degree basis functions define by the recursion relationship
$$
N_{i,p}(t) = \frac{(t-t_i)}{(t_{i+p}-t_i)} N_{i,p-1}(t)
+ \frac{(t_{i+1+p}-t)}{(t_{i+1+p}-t_{i+1})} N_{i+1,p-1}(t).
$$
From these definitions, it is clear that the $N_{i,p}(t)$ basis function is a $p$-degree piecewise polynomials that is only non-zero over the finite interval $[t_i, t_{i+1+p})$. In addition since the number of entries in the knot-vector is related to the number of control points and the degree of the b-spline, we can identify the condition
$$
m = n+p+1.
$$

Using these conditions the we can create a small BSpline class in python as follows
```python
class BSpline:

    def __init__(self, 
                 p: int, 
                 c: Sequence[CPoint], 
                 t: Sequence[float]):
        t_lo = t[0]
        for t_hi in t[1:]:
            if t_lo > t_hi:
                raise ValueError('Invalid knot-vector')
        if len(t) != len(c) + p + 1:
            raise ValueError('Invalid knot-vector length')
        self.p = p
        self.c = c
        self.t = t
```

### **Evaluation with de Boor's algorithm**

A b-spline can be efficiently evaluated using the recursive definition of the basis functions. Consider a general b-spline of degree $p$ with $n+1$ control points.
$$
S(t) = \sum_{i=0}^n P_i \, N_{i,p}(t),
$$
we can make the expansion given above for the basis functions to get
$$
S(t) = \sum_{i=0}^n \frac{t-t_i}{t_{i+p}-t_i} P_i N_{i,p-1}(t)
+ \sum_{i=0}^n \frac{t_{i+1+p}-t}{t_{i+1+p}-t_{i+1}} P_i N_{i+1,p-1}(t).
$$
Changing the limits of the sum in the second term to go from 1 to $n+1$
$$
S(t) = \sum_{i=0}^n \frac{t-t_i}{t_{i+p}-t_i} P_i N_{i,p-1}(t)
+ \sum_{i=1}^{n+1} \frac{t_{i+p}-t}{t_{i+p}-t_i} P_{i-1} N_{i,p-1}(t).
$$
and if we make the identification,
$$
Q^{(0)}_i = \begin{cases} 
P_i & \text{for} \quad 0 \le i \le n, \\
0 & \text{otherwise},
\end{cases}
$$
which allows us to gracefully handle the first and last term, then we can combine the sums to a single expression
$$
S(t) = \sum_{i=0}^{n+1} \bigg(\frac{t-t_i}{t_{i+p}-t_i} Q^{(0)}_i 
+ \frac{t_{i+p}-t}{t_{i+p}-t_i} Q^{(0)}_{i-1}\bigg) N_{i,p-1}(t),
$$
which looks like a b-spline of degree $p-1$ 
$$
S(t) = \sum_{i=0}^{n+1} Q^{(1)}_i N_{i,p-1}(t),
$$
with $n+2$ new controls points 
$$
Q^{(1)}_i = \alpha \, Q^{(0)}_i + (1-\alpha) \, Q^{(0)}_{i-1}
\quad \text{where}\quad
\alpha = \frac{t-t_i}{t_{i+p}-t_i}.
$$
(Note: This is expansion is not of general use because the 'higher order' control points $Q^{(p)}$ for $p>0$ have a non-trivial t dependance. However, it is useful for evaluation when we have specific values of $t$ which allow use to calculate the 'higher order' control points $Q$.)

By repeated expansion of the basis functions and regrouping terms we can re-write the spline as
$$
S(t) = \sum_{i=0}^{n+p} Q^{(p)}_i B_{i,0}(t),
$$
where using the definitions of the zero'th order basis functions, this evaluates to simply
$$
S(t) = Q^{(p)}_i \quad\text{if}\quad t_i \le t \le t_{i+1}.
$$
The new control points can be calculated from each other with the relation
$$
Q^{(r+1)}_i = \alpha \, Q^{(r)}_i + (1-\alpha) \, Q^{(r)}_{i-1}
\quad \text{where}\quad
\alpha = \frac{t-t_i}{t_{i+p-r}-t_i}.
$$

Example $p=2$, $P=[1]$, $t=[0,1,2,3]$
$$
S(t) = (Q^{(0)}_0 = 1) N_{0,2}(t)
$$
$$
S(t) = \bigg(
    Q^{(1)}_0 = \frac{t}{2} Q^{(0)}_0 + 
    \frac{2-t}{2} Q^{(0)}_{-1}  
    \bigg) N_{0,1}(t) + \\
    \bigg(
    Q^{(1)}_1 = \frac{t-1}{2} Q^{(0)}_1 +
     \frac{3-t}{2} Q^{(0)}_0
    \bigg) N_{1,1}(t)
$$
$$
S(t) = \Big(
    Q^{(2)}_0 = t \, Q^{(1)}_0 + (1-t) Q^{(1)}_{-1}
\Big) N_{0,0}(t) + \\
\Big(
    Q^{(2)}_1 = (t-1) Q^{(1)}_1 + (2-t) Q^{(1)}_0
\Big) N_{1,0}(t) + \\
\Big(
    Q^{(2)}_2 = (t-2) Q^{(1)}_2 + (3-t) Q^{(1)}_1
\Big) N_{2,0}(t)
$$
$$
S(t) = \bigg(
    \frac{t}{2} 
    \big(
        t \, N_{0,0}(t) +
        (2-t) N_{1,0}(t)
    \big) \\+
    \frac{3-t}{2} 
    \big(
        (t-1) N_{1,0}(t)
        (3-t) N_{2,0}(t)
    \big)
\bigg)
$$
$$
S(t) =
    \frac{t^2}{2} N_{0,0}(t) + \\
    \frac{t(2-t) + (3-t)(t-1)}{2} N_{1,0}(t) + \\
    \frac{(3-t)^2}{2} N_{2,0}(t)
$$
$$
S(t) =\begin{cases}
    t^2/2 & \text{for}\quad 0 \le t < 1 \\
    \big(t(2-t) + (3-t)(t-1)\big)/2 \quad & 
    \text{for}\quad 1 \le t < 2 \\
    \big((3-t)^2\big)/2 & \text{for}\quad 2 \le t < 3
    \end{cases}
$$

To implement de Boor's algorithm, we need two helper functions. First we need to find the index of the interval in the knot-index where the x value falls. Here is a naive implementation in python that uses a linear search:
```python
def knot_idx(x: float, t: Sequence[float]):
    if t < t[0] or t >= t[-1]:
        raise ValueError('')
    for k, ti in enumerate(t[1:]):
        if t < ti:
            return k
```
Next, we need to expand the control points for indices beyond their initial range. Here is an example based on an if statement:
```python
def q0(c, i):
    if i < 0 or i >= len(c):
        return c[0]*0
    else:
        return c[i]
```
Now we can give a basic implementation of de Boor's algorithm: 
```python
def deboor(p, c, t, x):
    k = knot_idx(x, t)
    q = [q0(c, k-r) for r in range(p, -1, -1)]
    for r in range(p):
        for j in range(p, r, -1):
            i = k+j-p
            ipp = k+j-r
            a = (x-t[i])/(self.t[ipp]-self.t[i])
            q[j] = a*q[j] + (1-a)*q[j-1]
    return q[p]
```
First you set up the relevant $Q^{(0)}_i$ terms, which will be the $i=k$ and the $p$ below that. Then we loop over the list $p$ times. For each time we update the $Q$ s in reverse order so that we can overwrite the values.

### **Derivatives**

Consider the derivative of a b-spline with respect to the parametric parameter $t$
$$
\frac{\mathrm{d}}{\mathrm{d}t} C(t) =
C'(t) = \sum_{i=0}^n P_i \, N'_{i,p}(t),
$$
where the derivative clearly passes to the basis functions because the control points are independent of $t$. The derivative of the basis functions is
$$
N'_{i,p} = \frac{p}{t_{i+p}-t_i} N_{i,p-1} - 
\frac{p}{t_{i+p+1}-t_{i+1}} N_{i+1,p-1}.
$$
(Note: This relation can be 'discovered' by evaluating by hand the first expression for the lowest order basis functions, and then can be proven by induction.) Using this expansion we can rewrite the derivative of the basis spline
$$
C'(t) = 
\sum_{i=0}^n \frac{p \, P_i}{t_{i+p}-t_i} N_{i,p-1}(t) -
\sum_{i=0}^n \frac{p \, P_i}{t_{i+p+1}-t_{i+1}}N_{i+1,p-1}(t),
$$
then re-index ($i \to i-1$) and combine the sums yielding
$$
C'(t) = \sum_{i=0}^{n+1} 
\frac{p \, (Q_i-Q_{i-1})}{t_{i+p}-t_{i}} 
\, N_{i,p-1}(t),
$$
where we have used the same method as for evaluating to define a 'larger' set of control points $Q_i$ that are equal to zero for $i < 0$ and $i > n$. We can now identify that the derivative is simply another b-spline of degree $p-1$ with $n+1$ new control points
$$
R_i = \frac{p \, (Q_i-Q_{i-1})}{t_{i+p}-t_{i}}
$$
defined over the same knot-sequence. Higher order derivatives can easily be calculated by repeating the 

## **NURBS**
### **Definition**
An extension of the b-spline is the NURBS or non-uniform rational b-spline that gives each $D$-dimensional control point $P_i$ to have weights $w_i$. The curve multiplies each control point by the weight, but also divides the result by the sum of the weights
$$
S(t) = \frac{\sum_{i=0}^n w_i \, P_i \, N_{i,p}(t)}
{\sum_{i=0}^n w_i \, N_{i,p}(t)},
$$
which turns the polynomial expression for the b-spline into a rational polynomial expression. 

### **Evaluation**
The $D$-dimensional NURBS curve can be evaluated using the de Boor algorithm as a ratio of two b-spline curves, a $D$-dimensional $C(t)$ and a 1-D $w(t)$, $define over the same knot-vector
$$
S(t) = \frac{C(t)}{w(t)}
\quad\text{where}\quad
\big\langle C(t), w(t)\big\rangle = 
\sum_{i=0}^n \big\langle w_i P_i, \, w_i\big\rangle \, N_{i,p}(t).
$$
The derivative of a NURBS curve be calculated using the same $C(t)$ and $w(t)$ and the product rule
$$
S'(t) = \frac{1}{w(t)}\Big(C'(t) - S(t)\,w'(t)\Big),
$$
and higher order derivatives using a generalized Libnitz's rule that utilized lower the calculation of the lower order derivatives
$$
S^{(n)}(t) = \frac{1}{w(t)}\bigg(C^{(n)} - \sum_{i=1}^n {n \choose k} S^{(n-k)}(t)\, w^{(k)}(t) \bigg).
$$

### **Differential Geometry**
A NURBS curve $S(t): [t_0, t_m) \to \mathbb{R}^D$ defines a mapping from a 1-D parameter space to $D$-dimensional Euclidean space, and that 3 dimensional curve has certain properties that are relevant and useful to calculate.

#### **Arc length**
The arc length along a curve $S(t)$ is given by the integral
$$
s(t) = \int_{t_0}^t \!\!\! \big\lVert S'(\tau) \big\rVert \, \mathrm{d}\tau.
$$
There is no general closed form for a rational polynomial expression, so there is in general no closed form for a NURBS curve. However, since the expression for $s(t)$ is a monotic function it is in principle invertible and the respective derivatives are 
$$
\frac{\mathrm{d} s}{\mathrm{d} t} = \big\lVert S'(\tau) \big\rVert
\quad\text{where}\quad
\frac{\mathrm{d} t}{\mathrm{d} s} = \frac{1}{\big\lVert S'(\tau) \big\rVert}.
$$

#### **Tangent vector**
The tangent vector is a unit vector that is tangent to the curve at a point, and is simply the normalized derivative with respect to the parametric parameter $t$
$$
\mathbf{T} = \frac{S'(t)}{\lVert S'(t) \rVert}.
$$

#### **Curvature**
The curvature is the magnitude of the derivative of the tangent vector with respect to arc-length
$$
\kappa = \bigg\lVert 
    \frac{\mathrm{d}}{\mathrm{d}s} \mathbf{T} 
\bigg\rVert.
$$
Using the chain rule and the expression for $\frac{\mathrm{d}t}{\mathrm{d}s}$ above, we can find an expression for curvature dependent only on the parametric parameter $t$
$$
\kappa = \frac{\big\lVert S'(t) \times S''(t) \big\rVert}
{\big\lVert S'(t) \big\rVert^3},
$$
where the cross product $\times$ gives the signed area of the parallelogram formed from the two vectors.

#### **Frenet-Serret frame**

When working with curves in 3-D space, the Frenet-Serret frame defines an orthonormal basis ($\mathbf{T}$, $\mathbf{N}$, $\mathbf{B}$)for each point on the curve. The usual methods for finding the Frenet-Serret frame depend on having an arc-length parameterized curve. However, that can be an computational expensive process for a general NURBS curve. Instead we will try to define things only in terms of parametric parameter $t$. The tangent vector is simply the normalized derivative. The tangent vector $T$ is the same as above.
$$
\mathbf{T} = \frac{S'(t)}{\lVert S'(t) \rVert}.
$$
The first and second derivative of the curve define the oscillating plane, so the binormal $B$ can be found as the norm of
$$
\mathbf{B} = \frac{S'(t) \times S''(t)}
{\big\lVert S'(t) \times S''(t)\big\rVert}.
$$
We can now form the normal vector as a 
$$
\mathbf{N} = \mathbf{B} \times \mathbf{T}.
$$
