## @file serde.py
"""\
Contains the deserialization method for constructive solid geometry objects
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of pyscad.
#
# pyscad is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pyscad is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pyscad. If not, see <https://www.gnu.org/licenses/>.
from typing import Union
import importlib

import numpy as np
from flint import flint

from .types import *
from .trans import Transform
from .csg import Csg

class ParamSerde:
    """Serialize and Deserialize routines for generic parameters"""

    @classmethod
    def ser(cls, p: Union[str, Num, npt.NDArray], allow_str: bool = True) -> Union[str, float, dict, list]:
        """Convert either a number or numpy array into a serializable object
        @param p A generic number (int, float or flint) or numpy array
        @return A serializable object
        """
        if isinstance(p, str) and allow_str:
            # Strings and simple numbers are already serializable
            return p
        if isinstance(p, (int, float, *np_numbers)):
            # Strings and simple numbers are already serializable
            return p
        elif isinstance(p, flint):
            # We turn flints into dict with the member data
            return {
                'a': float(p.a), 
                'b': float(p.b), 
                'v': float(p.v)
            }
        elif isinstance(p, np.ndarray) and p.dtype in (*np_numbers, flint):
            # Arrays are recursively handled a row at a time
            return [cls.ser(x, allow_str=False) for x in p]
        else:
            raise TypeError("Only number of numpy arrays can be serialized as parameters")

    @classmethod
    def deser(cls, a: Union[float, dict, list]) -> Union[str, Num, npt.NDArray]:
        """Convert a serializable object into string, number, or array parameter
        @param a The serializable object
        @return The parameter as a string, number, or numpy array
        """
        if isinstance(a, (str, int, float, *np_numbers)):
            # Simply numbers and strings are unchanged
            return a
        elif isinstance(a, dict):
            # Build a new flint if we get a dict
            if len(a) != 3 or 'a' not in a.keys() or 'b' not in a.keys() or 'v' not in a.keys():
                raise ValueError("Non supported type found")
            f = flint(0)
            f.interval = a['a'], a['b'], a['v']
            return f
        elif isinstance(a, list):
            # Build a multi-level list to recursively build the arrays
            l = [cls.deser(x) for x in a]
            if isinstance(l[0], (int, float, *np_numbers)):
                # Confirm all types are the same
                for x in l[1:]:
                    if not isinstance(x, (int, float, *np_numbers)):
                        raise ValueError("Arrays must be uniform types")
                return np.array(l, dtype=np.float64)
            elif isinstance(l[0], flint):
                # Confirm all types are the same
                for x in l[1:]:
                    if not isinstance(x, flint):
                        raise ValueError("Arrays must be uniform types")
                return np.array(l, dtype=flint)
            elif isinstance(l[0], np.ndarray):
                # Confirm uniform shape and type
                for x in l[1:]:
                    if not isinstance(x, np.ndarray) or x.shape != l[0].shape or x.dtype != l[0].dtype:
                        raise ValueError("Arrays must be uniform shapes and types")
                return np.stack(l)
        else:
            raise TypeError("Can only deserialize numbers or arrays")


class TransSerde:
    """Serialize and Deserialize routines for transforms"""

    @staticmethod
    def ser(obj: Transform) -> dict:
        """Build a python dict for JSON serialization
        @param obj The transform object
        @return A serializable object
        """
        if not isinstance(obj, Transform):
            raise TypeError("Can only serialize Transform objects")
        state = dict()
        state['__module__'] = obj.__module__
        state['__class__'] = obj.__class__.__name__
        state['m'] = ParamSerde.ser(obj.m)
        state['v'] = ParamSerde.ser(obj.v)
        return state

    @staticmethod
    def deser(ser: dict) -> Transform:
        """Build a transform from a serialized dict object
        @param ser The serializable object (a python dict)
        @return A Transform object
        """
        if not isinstance(ser, dict):
            raise TypeError("Can only deserialized from a python dict object")
        if '__module__' not in ser.keys() or '__class__' not in ser.keys():
            raise ValueError("Dict object must a have '__module__' and '__class__' attribute")
        mod_ = importlib.import_module(ser['__module__'])
        cls_ = getattr(mod_, ser['__class__'])
        m = ParamSerde.deser(ser['m'])
        v = ParamSerde.deser(ser['v'])
        t = cls_.from_arrays(m, v)
        return t


class CsgSerde:
    """Serialize and Deserialize routines for CSG objects"""

    @staticmethod
    def ser(obj: Csg) -> dict:
        """Build a python dict for JSON serialization
        @param obj The transform object
        @return A serializable object
        """
        if not isinstance(obj, Csg):
            raise TypeError("Can only serialize Csg objects")
        # Start off with object type
        state = dict()
        state['__module__'] = obj.__module__
        state['__class__'] = obj.__class__.__name__
        # Then Include the list of transforms
        state['trans'] = [TransSerde.ser(t) for t in obj.trans]
        # Next add all object specific parameters
        params = {}
        for param_name in obj.params:
            param_val = getattr(obj, param_name)
            if isinstance(param_val, np.ndarray):
                param_ser = array_ser(param_val)
            else:
                param_ser = num_ser(param_val)
            params[p] = ParamSerde.ser(param_val)
        state['params'] = params
        # Finally include meta-data
        state['meta'] = {}
        for k, v in obj.meta.items():
            try:
                enc = json.JSONEncoder()
                ser_v = enc.default(v)
                state['meta'][k] = ser_v
            except TypeError:
                warnings.warn(f"CSG meta-data ({v}) could not be serialized and was discarded")
        return state

    @staticmethod
    def deser(ser: dict) -> Csg:
        """Build a CSG object from a serialized dict object
        @param ser The serializable object (a python dict)
        @return A CSG object
        """
        if not isinstance(ser, dict):
            raise TypeError("Can only deserialized from a python dict object")
        if '__module__' not in ser.keys() or '__class__' not in ser.keys():
            raise ValueError("Dict object must a have '__module__' and '__class__' attribute")
        mod_ = importlib.import_module(ser['__module__'])
        cls_ = getattr(mod_, ser['__class__'])
        # Instantiate the CSG object
        csg = cls_._noargs()
        # Add the object parameters
        for param_name, param_val in ser['params'].items():
            setattr(csg, param_name, param_deser(param_val))
        # For CSG operators, we have to add children using recursion
        if 'children' in ser.keys():
            setattr(csg, 'children', [csg_deser(child) for child in ser['children']])
        # Attach all the transforms
        csg.trans = [trans_deser(t) for t in ser['trans']]
        # Attach all serialized meta-data
        # csg.
        return csg 
