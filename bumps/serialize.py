import json

from . import bounds, parameter, plugin

import numpy as np
from dataclasses import is_dataclass, fields, dataclass
from importlib import import_module
from typing import Dict, List, Tuple, Union
from types import GeneratorType
import traceback
import warnings
import asyncio
from collections import defaultdict

DEBUG = True
MISSING = object()
REFERENCE_TYPE = "Reference"

@dataclass
class Reference:
    id: str

class Deserializer(object):
    def __init__(self, loop=None):
        loop = loop if loop is not None else asyncio.get_running_loop()
        self.loop = loop
        self.refs = {}
        self.lock = asyncio.Lock()
        self.deferred: Dict[str, asyncio.Future] = defaultdict(self.loop.create_future)
    
    def get_class(self, module_name: str, class_name: str):
        return getattr(import_module(module_name), class_name)

    async def rehydrate(self, obj):
        if isinstance(obj, dict):
            obj = obj.copy()
            t: str = obj.pop('type', MISSING)
            if t == REFERENCE_TYPE:
                obj_id: str = obj.get("id", MISSING)
                if obj_id is MISSING:
                    raise ValueError("object id is required for Reference type")
                async with self.lock:
                    if DEBUG and not obj_id in self.deferred:
                        print(f"creating deferred value: {obj_id}")
                    fut = self.deferred[obj_id]
                return await fut

            if not t == 'np.ndarray':
                # skip processing values list in ndarray
                keys = obj.keys()
                values = await asyncio.gather(*[self.rehydrate(value) for value in obj.values()])
                for key,value in zip(keys, values):
                    obj[key] = value

            if t is MISSING:
                return obj
            elif t == 'np.ndarray':
                return self.ndarray(obj)
            else:
                try:
                    module_name, class_name = t.rsplit('.', 1)
                    klass = self.get_class(module_name, class_name)
                    hydrated = await self.instantiate(klass, t, obj)
                    return hydrated
                except Exception:
                    # there is a type, but it is not found...
                    raise ValueError("type %s not found!" % t, obj)

        elif isinstance(obj, list):
            return await asyncio.gather(*[self.rehydrate(v) for v in obj])
        elif isinstance(obj, tuple):
            values = await asyncio.gather(*[self.rehydrate(v) for v in obj])
            return tuple(values)
        else:
            return obj

    async def instantiate(self, klass: type, typename: str, serialized: dict):
        s = serialized.copy()
        obj_id = s.get("id", MISSING) # will id be a required schema element?
        #print('rehydrating: ', typename)
        if obj_id is MISSING or not obj_id in self.refs:
            # use from_dict to initialize the class, if that method exists:
            class_factory = getattr(klass, 'from_dict', klass)

            try:
                hydrated = class_factory(**s)
            except Exception as e:
                print(class_factory, s, typename)
                raise e
            if obj_id is not MISSING:
                self.refs[obj_id] = hydrated
                async with self.lock:
                    if DEBUG and not obj_id in self.deferred:
                        print(f"setting future value for {obj_id} to {hydrated}")
                    fut: asyncio.Future = self.deferred[obj_id]
                    fut.set_result(hydrated)
        else:
            print(f"loading from refs: {typename}, {obj_id}")
            hydrated = self.refs[obj_id]
        return hydrated

    def ndarray(self, obj: dict):
        return np.asarray(obj['values'], dtype=np.dtype(obj.get('dtype', float)))
    
    def resolve_references(self, obj):
        if isinstance(obj, Reference):
            return self.refs.get(obj.id)
        elif isinstance(obj, dict):
            for key,value in obj.items():
                    obj[key] = self.resolve_references(value)
            return obj                
        elif isinstance(obj, list):
            return [self.resolve_references(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.resolve_references(v) for v in obj)
        else:
            return obj
    
async def async_from_dict(serialized, loop=None):
    oasis = Deserializer(loop)
    return await oasis.rehydrate(serialized)

def from_dict(serialized):
    return asyncio.run(async_from_dict(serialized))

def load(filename):
    with open(filename, 'r') as fid:
        return from_dict(json.loads(fid.read()))


class Serializer:
    def __init__(self):
        self.refs = {}

    def dataclass_to_dict(self, obj):
        return dict([(f.name, self.to_dict(getattr(obj, f.name))) for f in fields(obj)])

    def to_dict(self, obj):
        if hasattr(obj, '__bumps_schema__') and hasattr(obj, 'id') and hasattr(obj, 'type'):
            if obj.id in self.refs:
                return dict(id=obj.id, type=REFERENCE_TYPE)
            else:
                return self.refs.setdefault(obj.id, self.dataclass_to_dict(obj))
        if is_dataclass(obj):
            return self.dataclass_to_dict(obj)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.to_dict(v) for v in obj)
        elif isinstance(obj, GeneratorType):
            return list(self.to_dict(v) for v in obj)
        # elif isinstance(obj, GeneratorType) and issubclass(obj_type, Tuple):
        #     return tuple(to_dict(v) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)((self.to_dict(k), self.to_dict(v))
                            for k, v in obj.items())
        elif isinstance(obj, np.ndarray) and obj.dtype.kind in ['f', 'i']:
            return dict(type="np.ndarray", dtype=obj.dtype.name, values=obj.tolist())
        elif isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
            return to_dict(obj.tolist())
        elif isinstance(obj, float):
            return str(obj) if np.isinf(obj) else obj
        elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
            return obj
        else:
            raise ValueError("obj %s is not serializable" % str(obj))


def to_dict(obj):
    return Serializer().to_dict(obj)

def save(filename, problem):
    try:
        p = to_dict(problem)
        with open(filename, 'w') as fid:
            json.dump(p, fid)
    except Exception:
        traceback.print_exc()
        warnings.warn(f"failed to create JSON file {filename} for fit problem")
