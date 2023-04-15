import importlib

from .csg import Csg
from ..trans import Transform

def csg_deser(ser: dict) -> Csg:
    """Build a CSG object from a serialized dict object"""
    if not isinstance(ser, dict):
        raise TypeError("Can only deserialized from a python dict object")
    if '__module__' not in ser.keys() or '__class__' not in ser.keys():
        raise ValueError("Dict object must a have '__module__' and '__class__' attribute")
    mod_ = importlib.import_module(ser['__module__'])
    cls_ = getattr(mod_, ser['__class__'])
    # Build the parameters used to instantiate the objects
    params = {}
    for param_name, param_ser in ser['params'].items():
        if isinstance(param_ser, list):
            param_val = array_deser(param_ser)
        else:
            param_val = num_deser(param_ser)
        params[param_name] = param_val
    # For CSG operators, we have to recursively call this function
    if 'children' in ser.keys():
        params['children'] = [csg_deser(child) for child in ser['children']]
    # Instantiate the CSG object
    csg = cls_(**params)
    # Attach all the transforms
    csg.trans = [Transform.deser(t) for t in ser['trans']]
    return csg 
