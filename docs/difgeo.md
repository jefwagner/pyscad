# Differential Geometry Solid Modeling

## Space Curves
A space curve 
$$
S(t): \mathbb{R} \to \mathbb{R}^D,
$$ 
defines a mapping from a 1-D parameter space to $D$-dimensional Euclidean space. There are several properties of the curve that are relevant and useful to calculate.

### **Arc length**
The arc length along a curve $S(t)$ is given by the integral
$$
s(t) = \int_{t_0}^t \!\!\! \big\lVert S'(\tau) \big\rVert \, \mathrm{d}\tau.
$$
This integral can be calculated numerically for a generic curve $S(t)$. However, the majority of curves will not be generic curve, they will be NURBS curves, represented as rational polynomial expressions. However, there is no general closed form for such an arc-length integral for a rational polynomial expression, so there is in general no closed form for a NURBS curve. However, since the  expression for $s(t)$ is a monotic function it is in principle invertible and the respective derivatives are 
$$
\frac{\mathrm{d} s}{\mathrm{d} t} = \big\lVert S'(\tau) \big\rVert
\quad\text{where}\quad
\frac{\mathrm{d} t}{\mathrm{d} s} = \frac{1}{\big\lVert S'(\tau) \big\rVert}.
$$

### **Tangent vector**
The tangent vector is a unit vector that is tangent to the curve at a point, and is simply the normalized derivative with respect to the parametric parameter $t$
$$
\mathbf{T} = \frac{S'(t)}{\lVert S'(t) \rVert}.
$$

### **Curvature**
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
where the cross product $\times$ gives the signed area of the parallelogram formed from the two vectors from the first and second derivatives $S'$ and $S''$.

### **Frenet-Serret Frame**
It is possible to form an orthonormal basis around every point along the curve. For a curve in $D$ dimensional space, the general method is to take the first $D$ derivatives and perform a Gram-Schmidt orthogonalization. In all dimensions, the vector based on the first derivative is always the tangent vector $\mathbf{T}$. In 3-dimensional space the next two vectors are known as the normal $\mathbf{N}$ and bi-normal $\mathbf{B}$.

## **Surfaces**
Consider a surface as a mapping 
$$
S(u,v): \mathbb{R}^2 \to \mathbb{R}^D
$$
from a 2-dimensional parameter space to $D$ dimensional surface. 

### **Tensor product surfaces**
Most of the surfaces we will consider will be constructed as the direct product of two 1-dimensional space-curves

### 