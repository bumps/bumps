from copy import deepcopy
from enum import Enum
import numpy as np
from dataclasses import is_dataclass, fields, dataclass
import graphlib
import json
from importlib import import_module
from typing import Dict, List, Literal, Optional, TypedDict, Union
from types import GeneratorType
import traceback
import warnings

from . import plugin
from .util import SCHEMA_ATTRIBUTE_NAME, get_libraries

DEBUG = False


class SCHEMA_VERSIONS(str, Enum):
    BUMPS_DRAFT_O1 = "bumps-draft-01"
    BUMPS_DRAFT_02 = "bumps-draft-02"
    BUMPS_DRAFT_03 = "bumps-draft-03"


SCHEMA = SCHEMA_VERSIONS.BUMPS_DRAFT_03
REFERENCES_KEY = "references"
REFERENCE_IDENTIFIER = "$ref"
MISSING = object()
REFERENCE_TYPE_NAME = "Reference"
REFERENCE_TYPE = Literal["Reference"]
TYPE_KEY = "__class__"


@dataclass
class Reference:
    id: str
    type: REFERENCE_TYPE


JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]


class SerializedObject(TypedDict, total=True):
    schema: str
    object: JSON
    references: Dict[str, JSON]


def deserialize(serialized: SerializedObject, migration: bool = True):
    """rehydrate all items in serialzed['references'] then
    - reydrate all objects in serialized['object']
    - replacing `Reference` types with python objects from `references`
    """

    if migration:
        # first apply built-in migrations:
        _, serialized = migrate(serialized)
        # then apply plugin migrations:
        serialized = plugin.migrate_serialized(serialized)

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

    return _rehydrate(serialized["object"], references)


#### deserializer helpers:
def _rehydrate(obj, references: Dict[str, object]):
    if isinstance(obj, dict):
        obj = obj.copy()
        t: str = obj.pop(TYPE_KEY, MISSING)
        if t == REFERENCE_TYPE_NAME:
            obj_id: str = obj.get("id", MISSING)
            if obj_id is MISSING:
                raise ValueError("object id is required for Reference type")
            # requires that self.references is populated with rehydrated objects:
            return references[obj_id]
        elif t == "bumps.util.NumpyArray":
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
                    module_name, class_name = t.rsplit(".", 1)
                    # print(module_name, class_name)
                    klass = getattr(import_module(module_name), class_name)
                    hydrated = _instantiate(klass, t, obj)
                    return hydrated
                except Exception as e:
                    # there is a type, but it is not found...
                    raise ValueError("type %s not found!, error: %s" % (t, e), obj)
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
    class_factory = getattr(klass, "from_dict", klass)
    try:
        hydrated = class_factory(**s)
    except Exception as e:
        print(class_factory, s, typename)
        raise e
    return hydrated


def _to_ndarray(obj: dict):
    return np.asarray(obj["values"], dtype=np.dtype(obj.get("dtype", float)))


def _find_ref_dependencies(obj, dependencies: set):
    if isinstance(obj, dict):
        if obj.get(TYPE_KEY, None) == REFERENCE_TYPE:
            dependencies.add(obj["id"])
        else:
            for v in obj.values():
                _find_ref_dependencies(v, dependencies)
    elif isinstance(obj, list):
        for v in obj:
            _find_ref_dependencies(v, dependencies)


#### end deserializer helpers


def serialize(obj, use_refs=True, add_libraries=True):
    references = {}

    def make_ref(obj_id: str):
        return {"id": obj_id, TYPE_KEY: REFERENCE_TYPE_NAME}

    def dataclass_to_dict(dclass, include=None, exclude=None):
        all_fields = fields(dclass)
        if include is not None:
            all_fields = [f for f in all_fields if f.name in include]
        elif exclude is not None:
            all_fields = [f for f in all_fields if not f.name in exclude and not f.name.startswith("_")]
        else:
            all_fields = [f for f in all_fields if not f.name.startswith("_")]
        cls = dclass.__class__
        fqn = f"{cls.__module__}.{cls.__qualname__}"
        output = dict([(f.name, obj_to_dict(getattr(dclass, f.name))) for f in all_fields])
        output[TYPE_KEY] = fqn
        return output

    def obj_to_dict(obj):
        if hasattr(obj, SCHEMA_ATTRIBUTE_NAME):
            schema_opts = getattr(obj, SCHEMA_ATTRIBUTE_NAME)
            include = schema_opts.get("include", None)
            exclude = schema_opts.get("exclude", None)
            use_ref = use_refs and hasattr(obj, "id")
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
        elif isinstance(obj, np.ndarray) and obj.dtype.kind in ["f", "i", "U"]:
            return {TYPE_KEY: "bumps.util.NumpyArray", "dtype": str(obj.dtype), "values": obj.tolist()}
        elif isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            return obj_to_dict(obj.tolist())
        elif isinstance(obj, Enum):
            return obj_to_dict(obj.value)
        elif isinstance(obj, float):
            return str(obj) if np.isinf(obj) else obj
        elif isinstance(obj, int) or isinstance(obj, str) or obj is None:
            return obj
        else:
            raise ValueError("obj %s is not serializable" % str(obj))

    serialized = {"$schema": SCHEMA, "object": obj_to_dict(obj), REFERENCES_KEY: references}

    if add_libraries:
        serialized["libraries"] = get_libraries(obj)

    return serialized


def save_file(filename, problem):
    try:
        p = serialize(problem)
        with open(filename, "w") as fid:
            json.dump(p, fid)
    except Exception:
        traceback.print_exc()
        warnings.warn(f"failed to create JSON file {filename} for fit problem")


def load_file(filename):
    with open(filename, "r") as fid:
        serialized: SerializedObject = json.loads(fid.read())
        final_version, migrated = migrate(serialized)
        print("final version: ", final_version)
        return deserialize(migrated)


#### MIGRATIONS
def validate_version(version: str, variable_name="from_version"):
    if version not in list(SCHEMA_VERSIONS):
        raise ValueError(f"must choose a valid {variable_name} from this list: {[s.value for s in SCHEMA_VERSIONS]}")


def migrate(
    serialized: dict, from_version: Optional[SCHEMA_VERSIONS] = None, to_version: Optional[SCHEMA_VERSIONS] = SCHEMA
):
    """
    Migrate a serialized object from one version to another
    By default, the `from_version` is determined by inspection of the serialized object.
    This is overriden by setting the `from_version` keyword argument to a member of `SCHEMA_VERSIONS`

    Also by default, the target version is the current schema, which can be overriden with
    the `to_version` keyword argument
    """

    if from_version is None:
        from_version = serialized.get(
            "$schema", SCHEMA_VERSIONS.BUMPS_DRAFT_O1
        )  # fall back to first version if not specified

    validate_version(from_version, "from_version")
    validate_version(to_version, "to_version")

    current_version = from_version
    while current_version != to_version:
        print(f"migrating {current_version}...")
        current_version, serialized = MIGRATIONS[current_version](serialized)

    return current_version, serialized


def _migrate_draft_01_to_draft_02(serialized: dict):
    references = {}

    def rename_type(obj):
        if isinstance(obj, dict):
            t: str = obj.pop("type", obj.pop(TYPE_KEY, MISSING))
            # print(f"moving type to __class__ for id {obj.get('id', 'no id')}")
            if t is not MISSING:
                obj[TYPE_KEY] = t
            for v in obj.values():
                rename_type(v)
        elif isinstance(obj, list):
            for v in obj:
                rename_type(v)

    def build_references(obj):
        if isinstance(obj, dict):
            t: str = obj.get(TYPE_KEY, MISSING)
            obj_id: str = obj.get("id", MISSING)
            # if obj_id is not MISSING:
            #     print(f"building reference for id {obj_id}")
            if obj_id is not MISSING and not t in [MISSING, REFERENCE_TYPE_NAME]:
                if not obj_id in references:
                    references[obj_id] = deepcopy(obj)
                obj[TYPE_KEY] = REFERENCE_TYPE_NAME
                for k in list(obj.keys()):
                    if k not in [TYPE_KEY, "id"]:
                        del obj[k]
            for v in obj.values():
                build_references(v)
        elif isinstance(obj, list):
            for v in obj:
                build_references(v)

    rename_type(serialized)
    build_references(serialized)
    migrated = {
        "$schema": SCHEMA_VERSIONS.BUMPS_DRAFT_02.value,
        "object": serialized,
        "references": references,
    }
    return SCHEMA_VERSIONS.BUMPS_DRAFT_02, migrated


def _migrate_draft_02_to_draft_03(serialized: dict):
    # add migration code here
    def div_to_truediv(obj):
        # remove all 'div' operators and replace with 'truediv'
        if isinstance(obj, dict) and obj.get(TYPE_KEY, MISSING) == "bumps.parameter.Expression":
            if obj.get("op", MISSING) == "div":
                obj["op"] = "truediv"
            for v in obj.get("args", []):
                div_to_truediv(v)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                div_to_truediv(v)
        elif isinstance(obj, list):
            for v in obj:
                div_to_truediv(v)

    migrated = deepcopy(serialized)
    div_to_truediv(migrated)
    return SCHEMA_VERSIONS.BUMPS_DRAFT_03, migrated


MIGRATIONS = {
    SCHEMA_VERSIONS.BUMPS_DRAFT_O1: _migrate_draft_01_to_draft_02,
    SCHEMA_VERSIONS.BUMPS_DRAFT_02: _migrate_draft_02_to_draft_03,
}
