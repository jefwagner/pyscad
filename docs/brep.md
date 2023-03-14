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
that border the surface. The edges in the edge loop give some detail towards the
orientation of the surface. The 'outside' of the surface is defined from the edges in
the edgeloop by the rigtht-hand rule: If you take a vector tangent to the curve of an
edge, and cross it with a vector perpendicular to the edge and tangent to the surface,
you will get a vector that points towards the outside of the solid.

### Topology

Each face is topologiaclly equivalent to a disk. If there are any holes in a face, the face is split up into separate faces such the hole is along the boundary between
the two faces. This structure gives us some guarantees that are used in the meshing
algorithm for b-reps.

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
corresponding surfaces. The edges are oriented such that the primary definition of the
curve $S_p \circ C_p (t)$ traverses the face defined with the primary surface in a right
handed maner and the endpoints of the curve gives the vertices $S_p \circ C_p (t=0)=
V_p$ and $S_p \circ C_p (t=1) = V_s$. The same holds true for the secondary curve, $S_s
\circ C_s (t)$ traverses the face defined with the secondar surface in a right handed
manor and the endpoints of the curve give the vertices $S_s \circ C_s (t=0)=V_s$ and
$S_s \circ C_s (t=1)=V_p$.

### **Visibility**

For visualizing a solid, each b-rep can be rendered with a mesh . It can also be useful to visualize the edges to
show the intersection between surface. Edges marked 'invisible' will not be rendered
when visualizing edges. 

# Algorithms for breps

We will require two main algorithms for working with b-reps
1. Compute the intersection of two faces
2. Compute a quality mesh on a b-rep.

The first algorithm will be necessary when converting between a CSG tree to a BRep. Each
of the basic CSG objects will be turn into a b-rep: either by initial definition as for
a sphere or cube, or through construction for an extrusion or solid of rotation.
Combining these basic objects involves intersecting the faces of the original basic
objects together. The new face will be defined by the same surface, but will have
different edges defining it.

The second algorithm will be necessary to visualize the solid, and to export the solid
in various file formats. In particular, visualization software depends on defining a
mesh of triangles with properties, such as the surface normal, defined on each vertex.

## Face $\times$ Face intersection


### Do faces intersect at all


#### Spacial hashing

If we use a spacial hashing to store the faces, we can 

#### Convex hulls - the GKJ algorithm

If we have a polyhedron defined that gives a convex hull for each face, we can quickl

### Finding all intersection curves - algebraic curves shenanigans?

### Evaluation of curves along the intersection



## Quality Mesh Generation

We can create a quality mesh of a b-rep using [Chew's
algorithm](./chew_quality_mesh.pdf). The algorithm is fairly simple

1. Starting with an intial distribution of points on the b-rep create a constrained Delauny triangulation (CST). For each triangle grade it's 'badness' with some scale such as the local curvature of the surface, and place them into a priority queue. Pop off the worst triangle and add a new vertex either at a circumcenter or along an edge.

### Midpoint between points along an edge

Given two points $p_a$ and $p_b$ along a curve $C(t)$ such that $C(t_a) = p_a$ and $C(t_b) = p_b$ where, without loss of generality $t_a < t_b$, find a new point $C(t_c) = p_c$ such that $p_c$ lies on the plane that bisects the line between points $p_a$ and $p_b$.
