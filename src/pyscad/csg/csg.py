## @file csg.py CSG structures
"""\
Contains the constructive solid geometry CSG structures
"""
# Include copyright statement

from .geo import *

class Csg:
    """Constructive Solid Geometry (CSG) abstract base class"""
    t: list[Transform] # All CSG objects can have their own list of transforms
    # ToDo: Add all expected methods

    def scale(self, size):
        """Apply a scaling transformation"""
        self.t.append(Scale(size))

    def trans(self, dx):
        """Apply a translation transformation"""
        self.t.append(Translate(dx))

    def rot(self, axis, angle):
        """Apply a rotation transformation around arbitrary axis"""
        self.t.append(Rotate(axis, angle))
    
    def rotx(self, angle):
        """Apply a rotation transformation around the x-axis"""
        self.t.append(Rotate((1,0,0), angle))

    def roty(self, angle):
        """Apply a rotation transformation around the y-axis"""
        self.t.append(Rotate((0,1,0), angle))

    def rotz(self, angle):
        """Apply a rotation transformation around the z-axis"""
        self.t.append(Rotate((0,0,1), angle))

    def rotzxz(self, alpha, beta, gamma):
        """Apply rotation transformations using ZXZ euler angles"""
        self.t.extend([
            Rotate((0,0,1), alpha),
            Rotate((1,0,0), beta),
            Rotate((0,0,1), gamma)
        ])

class Op(Csg):
    """CSG Operator abstract base class"""
    c: list[Csg]
 
    def __init__(self, *children):
        """Create a new Op object"""
        for child in children:
            if not isinstance(child, Csg):
                raise ValueError(f'CSG operators only act CSG objects')
            self.c.append(child)

class Union(Op):
    """A CSG Union operator"""
    ...

class Diff(Op):
    """A CSG Difference operator"""
    ...

class IntXn(Op):
    """A CSG Intersection operator"""
    ...


class Prim(Csg):
    """CSG Primitive abstract base class"""

class Sphere(Prim):
    """A CSG Sphere primitive"""

    def __init__(self, radius = 1, position = (0,0,0)):
        """Create a new Sphere object"""
        self.t = [Scale(radius), Translate(position)]

class Box(Prim):
    """A CSG Box primitive"""

    def __init__(self, size = 1, position = (0,0,0)):
        """Create a new Box object"""
        self.t = [Scale(size), Translate(position)]

class Cyl(Prim):
    """A CSG Cylinder primitive"""

    def __init__(self, height = 1, radius = 1, position = (0,0,0)):
        self.t = [
            Scale(radius, radius, height),
            Translate(position)
        ]

class Cone(Prim):
    """A CSG Cone primitive"""

    def __init__(self, height = 1, radius = 1, position = (0,0,0)):
        self.t = [
            Scale(radius, radius, height),
            Translate(position)
        ]
