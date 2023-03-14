# Boundary representation

A brep, or boundary representation, is a way of modeling solids by describing the boundary between 'inside' and 'outside'. To fully describe a brep inside a solid  modeling program, we have to capture all of the topological information about the boundary, including:
- 2-D: faces - defines a portion of the 2-D surface of the solid
- 1-D: edges - a 1-D section of a curve where two faces meet
- 0-D: vertices - A 0-D point in $\mathbb{R}^3$ where three or more edges meet

A b-rep is perhaps best shown with an example
Example: Cube
- 6 faces
- 12 edges
- 8 vertices

## Face

A face $F$ consist of:
- a surface $S_a$
- edge loop ${E_{a0}, E_{a1}, \ldots, E_{an}}$

The surface $S_a$ is a parametric surface defined by the function $S_a(u,v) \to x,y,z$.
The edge loop consist of an ordered list of edges ${E_{a0}, E_{a1}, \ldots, E_{an}}$
that border the surface. 

### **Orientation**

The order of the edges in the edge loop give some detail towards the orientation of the
surface. Every two consecutive edges should meet at a vertex. The 'outside' of the
surface is defined from the edges in the edgeloop by the rigtht-hand rule: If you take a
vector tangent to the curve of an edge, and cross it with a vector perpendicular to the
edge and tangent to the surface, you will get a vector that points towards the outside
of the solid.

## Edge

An edge $E$ consist of:
- a primary curve $C_p$
- a primary surface $S_p$
- a primary vertex $V_p$
- a secondary curve $C_s$
- a secondary surface $S_s$
- a secondary vertex $V_s$
- a boolean describing the visibility

Every edge will have a primary and secondary curve, surface, and vertex. The curves
define a mapping from a 1-d parameter space $t \in [0,1]$ to the $(u,v)$ space of the
surfaces. The edges are oriented such that the primary definition of the curve $S_p
\circ C_p (t)$ traverses the face defined with the primary surface in a right handed
maner and the endpoints of the curve gives the vertices $S_p \circ C_p (t=0)= V_p$ and
$S_p \circ C_p (t=1) = V_s$. The same holds true for the secondary curve, $S_s \circ C_s
(t)$ traverses the face defined with the secondar surface in a right handed manor and
the endpoints of the curve give the vertices $S_s \circ C_s (t=0)=V_s$ and $S_s \circ
C_s (t=1)=V_p$.


# Algorithms for breps

### Midpoint between points along an edge

Given two points $p_a$ and $p_b$ along a curve $C(t)$ such that $C(t_a) = p_a$ and $C(t_b) = p_b$ where, without loss of generality $t_a < t_b$, find a new point $C(t_c) = p_c$ such that $p_c$ lies on the plane that bisects the line between points $p_a$ and $p_b$.