# PyScad
Basic idea: make an OpenSCAD like package for doing solid design in python.
 
Design for solid objects will be based on constructive solid geometry. A basic object in
constructive solid geometry will either be a 3-d primitive, like a sphere or cube, or
made from an extrusion of a 2-d shape or the rotation of a 2-D curve.

When rendering an object, or exporting to an .stl for printing the CSG tree will be
turned into a boundary representation where all of the surfaces are represented by NURBS
surfaces. The surfaces can then be triangulated into a mesh using the b-rep.

## To do
[_] floating point interval
    [x] flint arithmetic
        [x] tests
    [_] flint elementary functions
[_] curves and surfaces
    [x] polynomials
        [x] tests
    [x] b-spline evaluations
        [_] tests
    [x] b-spline derivatives
        [_] tests
    [x] nurbs curve evaluation
        [_] tests
    [x] nurbs curve derivatives
        [_] tests
    [ ] space curve arc length
    [ ] space curve properties
    [ ] nurbs surface evaluation
    [ ] nurbs surface evaluation
    [ ] surface curve properties
