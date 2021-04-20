import json

from . import bounds, parameter, plugin
SCHEMA_SOURCES = bounds, parameter

from dataclasses import is_dataclass, fields
from typing import List, Tuple, Union
from types import GeneratorType
import traceback
import warnings

def get_dataclass_defs(sources = SCHEMA_SOURCES):
    class_defs = {}
    for source in sources:
        names = dir(source)
        dataclasses = dict([(n, getattr(source, n)) for n in names if is_dataclass(getattr(source, n))])
        class_defs.update(dataclasses)
    return class_defs

class Deserializer(object):
    def __init__(self, sources=SCHEMA_SOURCES):
        self.class_defs = get_dataclass_defs(sources)
        self.refs = {}
        self.deferred = {}

    def rehydrate(self, obj):
        if isinstance(obj, dict):
            obj = obj.copy()
            has_type = 'type' in obj
            t = obj.pop('type', None)
            for key,value in obj.items():
                obj[key] = self.rehydrate(value)
                #print(key)
            if not has_type:
                return obj
            elif t in self.class_defs:
                hydrated = self.instantiate(t, obj)
                return hydrated
            else:
                # there is a type, but it is not found...
                raise ValueError("type %s not found!" % t, obj)
                
        elif isinstance(obj, list):
            return [self.rehydrate(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.rehydrate(v) for v in obj)
        else:
            return obj

    def instantiate(self, typename, serialized):
        s = serialized.copy()
        id = s.get("id", None) # will id be a required schema element?
        #print('rehydrating: ', typename)
        if id is None or not id in self.refs:
            class_factory = self.class_defs.get(typename)
            if hasattr(class_factory, 'from_dict'):
                class_factory = class_factory.from_dict
            try:
                hydrated = class_factory(**s)
            except Exception as e:
                print(class_factory, s, typename)
                raise e
            if id is not None:
                self.refs[id] = hydrated
        else:
            hydrated = self.refs[id]
        return hydrated

def from_dict(serialized):
    oasis = Deserializer(SCHEMA_SOURCES + plugin.SCHEMA_SOURCES)
    return oasis.rehydrate(serialized)

def load(filename):
    with open(filename, 'r') as fid:
        return from_dict(json.loads(fid.read()))

import copy
import numpy as np

def to_dict(obj, field_info=None):
    if is_dataclass(obj):
        return dict([(f.name, to_dict(getattr(obj, f.name), field_info=f)) for f in fields(obj)])
        # result = [('type', obj.__class__.__name__)]
        # for f in fields(obj):
        #     if f.name != "type":
        #         value = to_dict(getattr(obj, f.name))
        #         result.append((f.name, value))
        # return dict(result)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(v) for v in obj)
    elif isinstance(obj, GeneratorType):
        return list(to_dict(v) for v in obj)
    # elif isinstance(obj, GeneratorType) and issubclass(obj_type, Tuple):
    #     return tuple(to_dict(v) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((to_dict(k), to_dict(v))
                          for k, v in obj.items())
    elif isinstance(obj, np.ndarray) and obj.dtype.kind in ['f', 'i']:
        #return dict(type="numpy.ndarray", dtype=obj.dtype.name, values=obj.tolist())
        return obj.tolist()
    elif isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
        return to_dict(obj.tolist())
    elif isinstance(obj, float):
        return str(obj) if np.isinf(obj) else obj
    elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
        return obj
    else:
        raise ValueError("obj %s is not serializable" % str(obj))

def save(filename, problem):
    try:
        p = to_dict(problem)
        with open(filename, 'w') as fid:
            json.dump(p, fid)
    except Exception:
        traceback.print_exc()
        warnings.warn(f"failed to create JSON file {filename} for fit problem")