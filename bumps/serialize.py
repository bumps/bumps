import json

from . import bounds, parameter, plugin

import numpy as np
from dataclasses import is_dataclass, fields, dataclass
from importlib import import_module
from typing import Dict, List, Literal, Optional, Tuple, Union
from types import GeneratorType
import traceback
import warnings
import asyncio
from collections import defaultdict

from .util import SCHEMA_ATTRIBUTE_NAME, schema, NumpyArray

DEBUG = False
MISSING = object()
REFERENCE_TYPE = "Reference"

@dataclass
class Reference:
    id: str
    type: Literal[REFERENCE_TYPE]

class Deserializer(object):
    def __init__(self, loop=None):
        self.loop = loop if loop is not None else asyncio.get_running_loop()
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

            if t == 'bumps.util.NumpyArray':
                # skip processing values list in ndarray
                return self.ndarray(obj)
            else:
                keys = obj.keys()
                values = await asyncio.gather(*[self.rehydrate(value) for value in obj.values()])
                for key,value in zip(keys, values):
                    obj[key] = value

            if t is MISSING:
                # no "type" provided, so no class to instantiate
                return obj
            elif t == 'bumps.util.NumpyArray':
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
        obj_id: str = s.get("id", MISSING) # will id be a required schema element?
        #print('rehydrating: ', typename)
        fut: Optional[asyncio.Future] = None
        if obj_id is not MISSING:
            async with self.lock:
                # self.deferred is a defaultdict, so a new Future is returned if none
                # already exists for that obj_id
                fut = self.deferred[obj_id]
                if fut.done():
                    return fut.result()

        # if klass provides 'from_dict' method, use it - 
        # otherwise use klass.__init__ directly.
        class_factory = getattr(klass, 'from_dict', klass)
        try:
            hydrated = class_factory(**s)
        except Exception as e:
            print(class_factory, s, typename)
            raise e

        if fut is not None:
            if DEBUG:
                print(f"setting future value for {obj_id} to {hydrated}")
            async with self.lock:
                fut.set_result(hydrated)
            
        return hydrated

    def ndarray(self, obj: dict):
        return np.asarray(obj['values'], dtype=np.dtype(obj.get('dtype', float)))


async def async_from_dict(serialized, loop=None):
    oasis = Deserializer(loop)
    return await oasis.rehydrate(serialized)

def from_dict(serialized):
    return asyncio.run(async_from_dict(serialized))

def from_dict_threaded(serialized):
    # use this wrapper when another loop might be running
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(from_dict, serialized)
        return future.result()

async def async_load(filename):
    """ use in e.g. Jupyter notebooks """
    with open(filename, 'r') as fid:
        return await async_from_dict(json.loads(fid.read()))
    
def load(filename):
    return asyncio.run(async_load(filename))

class Serializer:
    def __init__(self, use_refs=True):
        self.refs = {}
        self.use_refs = use_refs

    def dataclass_to_dict(self, dclass, include=None, exclude=None):
        all_fields = fields(dclass)
        if include is not None:
            all_fields = [f for f in all_fields if f.name in include]
        elif exclude is not None:
            all_fields = [f for f in all_fields if not f.name in exclude and not f.name.startswith('_')]
        else:
            all_fields = [f for f in all_fields if not f.name.startswith('_')]
        cls = dclass.__class__
        fqn = f"{cls.__module__}.{cls.__qualname__}"
        output = dict([(f.name, self.to_dict(getattr(dclass, f.name))) for f in all_fields])
        output["type"] = fqn
        return output

    def to_dict(self, obj):
        if hasattr(obj, SCHEMA_ATTRIBUTE_NAME):
            schema_opts = getattr(obj, SCHEMA_ATTRIBUTE_NAME)
            include = schema_opts.get("include")
            exclude = schema_opts.get("exclude")
            if self.use_refs and hasattr(obj, 'id') and obj.id in self.refs:
                return dict(id=obj.id, type=REFERENCE_TYPE)
            else:
                output = self.dataclass_to_dict(obj, include=include, exclude=exclude)
                if self.use_refs and hasattr(obj, 'id'):
                    self.refs.setdefault(obj.id, output)
                return output
        elif is_dataclass(obj):
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
        elif isinstance(obj, np.ndarray) and obj.dtype.kind in ['f', 'i', 'U']:
            return dict(type="bumps.util.NumpyArray", dtype=str(obj.dtype), values=obj.tolist())
        elif isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
            return self.to_dict(obj.tolist())
        elif isinstance(obj, float):
            return str(obj) if np.isinf(obj) else obj
        elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
            return obj
        else:
            raise ValueError("obj %s is not serializable" % str(obj))


def to_dict(obj, use_refs=True):
    return Serializer(use_refs=use_refs).to_dict(obj)

def save(filename, problem):
    try:
        p = to_dict(problem)
        with open(filename, 'w') as fid:
            json.dump(p, fid)
    except Exception:
        traceback.print_exc()
        warnings.warn(f"failed to create JSON file {filename} for fit problem")
