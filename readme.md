# WARNING
**Project still under active development - does not work**

# PyScad

Basic idea: make an OpenSCAD like package for doing 3-d solid design in python.
 
Design for solid objects will be based on constructive solid geometry or CSG. A
basic object will either be a 3-d primitive, like a sphere or cube, or made from
an extrusion of a 2-d shape or the rotation of a 2-D curve. Those basic objects
can then be combined together with operations of union, intersection, or
difference to make compound objects. All objects, basic or compound, can be
manipulated with transformations such as translations, rotations, and scalings.
A pyscad object will be represented internally in python as a tree, where every
internal node is a operator and every leaf node is a basic object and every node
has a list of transformations applied.

There will be no direct support of viewing the object, however it will be
possible to exporting the object as .obj for viewing or to an .stl for printing.
My idea for construction is to either use existing viewers or create a new
viewer using three.js for .obj files that work in a jupyter notebook. That way
the solid objects can be created in an interactive manner in a jupyter notebook.

A long term goal will be to support saving the objects as STEP files that can be
opened directly in other CAD programs such as SolidWorks or CATIA. That way the
3-d objects created with this method could be shared with shops that will do CNC
machining to produce the custom designed parts.

## **Why**

The big question here is: why am I making a new 'programmers' CAD system?
Doesn't OpenSCAD already do everything you want to do?

First off - that's two questions, not one. To answer the later: Doesn't OpenSCAD
already do everything you want to do? OpenSCAD is amazing and I recommend
everybody try it at least once. I use it almost exclusively for designing parts
to 3D print. However, there are two parts of how it works that I wish worked
differently.
1. OpenSCAD represents the surface positions with floats, so it's possible to
get strange behaviors where surfaces that should be coincident are not. I end up
defining an small distance epsilon, and moving the surfaces up or down by the
epsilon to make sure unions and differences are handled correctly.
2. OpenSCAD instantly meshes curved shapes such as spheres or cylinders based on
a fineness parameter (`$fn`). There are advantages to this, but I would like a
cylinder to be based on a circle and allow me to do all the transformations and
manipulations first and only mesh the object when exporting. 

To answer the former: why am I making a new 'programmers' CAD system? Mostly
because I can. I need a hobby, and I enjoy learning about new things. This
project gives me something to work on in my free time AND lets me address the
two issues I have with OpenSCAD.

## **How will it work**

To address the first issue, instead of using floating point numbers internal for
the representation of the geometry, we will instead use rounded floating point
intervals for all the geometric properties. This treats the values as small
intervals, and we can guarantee that the 'true' value will be somewhere in the
interval. This allows us to add one third three times and get a value that
overlaps with one.

For the second issue, I will represent a solid object as a CSG tree and only
worry about translating that tree into a mesh as a as a final step. The path for
producing these outputs will involve turning the CSG tree into a boundary
representation or b-rep where all of the surfaces are represented by NURBS
surfaces. The advantage of NURBS surfaces is that they can exactly represent
conic sections, such as spheres or ellipsoids, and the transformations simply
act on the control points with no additional complications. The surfaces in the
b-rep can then be triangulated into a mesh using standard triangulation methods
that put some limit on triangle quality based on properties such as surface
curvature, or max deviation from the surface. The .obj can then be constructed
using triangles with vertex and normal data, and the .stl will only have
triangles.

## To do

There is a lot to do before this project is ready for use for anything. Below is
a to-do list of what has already been done, and what is yet to be done.

- [x] Rounded floating point numbers [100% (ish)]

I have written a python/numpy extension in C and published it as separate PyPI
package. The small amount of testing I have done so far seems to work OK. It
will fails in some NumPy and SciPy routines because some of the math functions
that interact between floating points and integers, such as `ceil`, `floor` and
`round` are not implemented for the rounded floating point data type.

- [ ] CSG [10%]

I have started laying out the implementation of the CSG data structure.

- [ ] b-rep [10%]

I have started the idea-work for how to structure the objects themselves. I
believe I know how I'm going to lay out the objects, objects, and which methods
they will need. However, I'm not exactly sure how I'm going to implement all the
methods. There is one method in particular I am struggling with: How can (a)
find and (b) definte for efficient calcuation the curves that result from
intersecting two faces?

- [x] Parametric curves and surfaces [100% (ish)]

I have defined an abstract base class for both and defined many useful methods
using these base classes, and I have implemented b-spline and NURBS versions for
both curves and surfaces. I think I will also need to define 'ruled' and
'rotated' surfaces as well to support extrusions and solids of rotation.

