

from enum import Enum
import numpy as np
from dataclasses import is_dataclass, fields, dataclass
import graphlib
import json
from importlib import import_module
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, TypedDict, Union
from types import GeneratorType
import traceback
import warnings
import asyncio
from collections import defaultdict

from . import bounds, parameter, plugin
from .util import SCHEMA_ATTRIBUTE_NAME, schema, NumpyArray

DEBUG = False

class SCHEMA_VERSIONS(str, Enum):
    REFL1D_DRAFT_O1 = "refl1d-draft-01"
    REFL1D_DRAFT_02 = "refl1d-draft-02"

SCHEMA = SCHEMA_VERSIONS.REFL1D_DRAFT_02
REFERENCES_KEY = "references"
REFERENCE_IDENTIFIER = "$ref"
MISSING = object()
REFERENCE_TYPE_NAME = "Reference"
REFERENCE_TYPE = Literal["Reference"]

@dataclass
class Reference:
    id: str
    type: REFERENCE_TYPE

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None 

class SerializedObject(TypedDict, total=True):
    schema: str
    object: JSON
    references: dict[str, JSON]

def deserialize(serialized: SerializedObject):
    """ rehydrate all items in serialzed['references'] then 
     - reydrate all objects in serialized['object']
     - replacing `Reference` types with python objects from `references`
    """

    serialized_references = serialized[REFERENCES_KEY]
    references = {}

    dependency_graph = {}
    for ref_id, ref_obj in serialized_references.items():
        dependencies = set()
        _find_ref_dependencies(ref_obj, dependencies)
        dependency_graph[ref_id] = dependencies

    sorter = graphlib.TopologicalSorter(dependency_graph)
    for ref_id in sorter.static_order():
        # deserialize and put all references into self.references
        references[ref_id] = _rehydrate(serialized_references[ref_id], references)
    # references is now full of deserialized objects,
    # and we're ready to rehydrate the entire tree...

    return _rehydrate(serialized['object'], references)

#### deserializer helpers:
def _rehydrate(obj, references: dict[str, object]):
    if isinstance(obj, dict):
        obj = obj.copy()
        t: str = obj.pop('type', MISSING)
        if t == REFERENCE_TYPE_NAME:
            obj_id: str = obj.get("id", MISSING)
            if obj_id is MISSING:
                raise ValueError("object id is required for Reference type")
            # requires that self.references is populated with rehydrated objects:
            return references[obj_id]
        elif t == 'bumps.util.NumpyArray':
            # skip processing values list in ndarray
            return _to_ndarray(obj)
        else:
            for key in obj:
                obj[key] = _rehydrate(obj[key], references)
            if t is MISSING:
                # no "type" provided, so no class to instantiate: return hydrated object
                return obj
                # obj values are now rehydrated: instantiate the class from 'type'
            else:
                try:
                    module_name, class_name = t.rsplit('.', 1)
                    # print(module_name, class_name)
                    klass = getattr(import_module(module_name), class_name)
                    hydrated = _instantiate(klass, t, obj)
                    return hydrated
                except Exception as e:
                    # there is a type, but it is not found...
                    raise ValueError("type %s not found!, error: %s" % (t,e), obj)
    elif isinstance(obj, list):
        # rehydrate all the items
        return [_rehydrate(v, references) for v in obj]
    else:
        # it's a bare value - just return
        return obj

def _instantiate(klass: type, typename: str, serialized: dict):
    s = serialized.copy()
    # if klass provides 'from_dict' method, use it - 
    # otherwise use klass.__init__ directly.
    class_factory = getattr(klass, 'from_dict', klass)
    try:
        hydrated = class_factory(**s)
    except Exception as e:
        print(class_factory, s, typename)
        raise e            
    return hydrated

def _to_ndarray(obj: dict):
        return np.asarray(obj['values'], dtype=np.dtype(obj.get('dtype', float)))

def _find_ref_dependencies(obj, dependencies: set):
    if isinstance(obj, dict):
        if obj.get('type', None) == REFERENCE_TYPE:
            dependencies.add(obj['id'])
        else:
            for v in obj.values():
                _find_ref_dependencies(v, dependencies)    
    elif isinstance(obj, list):
        for v in obj:
            _find_ref_dependencies(v, dependencies)

#### end deserializer helpers

def serialize(obj, use_refs=True):
    references = {}

    def make_ref(obj_id: str):
        return dict(id=obj_id, type=REFERENCE_TYPE_NAME)

    def dataclass_to_dict(dclass, include=None, exclude=None):
        all_fields = fields(dclass)
        if include is not None:
            all_fields = [f for f in all_fields if f.name in include]
        elif exclude is not None:
            all_fields = [f for f in all_fields if not f.name in exclude and not f.name.startswith('_')]
        else:
            all_fields = [f for f in all_fields if not f.name.startswith('_')]
        cls = dclass.__class__
        fqn = f"{cls.__module__}.{cls.__qualname__}"
        output = dict([(f.name, obj_to_dict(getattr(dclass, f.name))) for f in all_fields])
        output["type"] = fqn
        return output

    def obj_to_dict(obj):
        if hasattr(obj, SCHEMA_ATTRIBUTE_NAME):
            schema_opts = getattr(obj, SCHEMA_ATTRIBUTE_NAME)
            include = schema_opts.get("include")
            exclude = schema_opts.get("exclude")
            use_ref = use_refs and hasattr(obj, 'id')
            if (not use_ref) or (obj.id not in references):
                # only calculate dict if it's not already in refs, or if not using refs
                obj_dict = dataclass_to_dict(obj, include=include, exclude=exclude)
                if use_ref:
                    references.setdefault(obj.id, obj_dict)
            return make_ref(obj.id) if use_ref else obj_dict
        elif is_dataclass(obj):
            return dataclass_to_dict(obj)
        elif isinstance(obj, (list, tuple, GeneratorType)):
            return list(obj_to_dict(v) for v in obj)
        # elif isinstance(obj, GeneratorType) and issubclass(obj_type, Tuple):
        #     return tuple(to_dict(v) for v in obj)
        elif isinstance(obj, dict):
            return type(obj)((obj_to_dict(k), obj_to_dict(v)) for k, v in obj.items())
        elif isinstance(obj, np.ndarray) and obj.dtype.kind in ['f', 'i', 'U']:
            return dict(type="bumps.util.NumpyArray", dtype=str(obj.dtype), values=obj.tolist())
        elif isinstance(obj, np.ndarray) and obj.dtype.kind == 'O':
            return obj_to_dict(obj.tolist())
        elif isinstance(obj, Enum):
            return obj_to_dict(obj.value)
        elif isinstance(obj, float):
            return str(obj) if np.isinf(obj) else obj
        elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
            return obj
        else:
            raise ValueError("obj %s is not serializable" % str(obj))

    serialized = {
        "$schema": SCHEMA,
        "object": obj_to_dict(obj),
        REFERENCES_KEY: references
    }
    return serialized


def save_file(filename, problem):
    try:
        p = serialize(problem)
        with open(filename, 'w') as fid:
            json.dump(p, fid)
    except Exception:
        traceback.print_exc()
        warnings.warn(f"failed to create JSON file {filename} for fit problem")

def load_file(filename):
    with open(filename, 'r') as fid:
        serialized: SerializedObject = json.loads(fid.read())
        return deserialize(serialized)
    
#### MIGRATIONS



def migrate(serialized: dict, from_version: Optional[SCHEMA_VERSIONS] = None, to_version: Optional[SCHEMA_VERSIONS] = SCHEMA):
    """
    Migrate a serialized object from one version to another 
    By default, the `from_version` is determined by inspection of the serialized object.
    This is overriden by setting the `from_version` keyword argument to a member of `SCHEMA_VERSIONS`

    Also by default, the target version is the current schema, which can be overriden with
    the `to_version` keyword argument
    """
    pass