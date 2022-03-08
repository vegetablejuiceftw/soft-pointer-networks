"""A pydantic based data transfer object framework that provides.

- (de)serialization with nesting
- field validation from type-hints / Field class
- (some) immutability / lifecycle control

"""
from datetime import datetime
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import orjson
from pydantic import BaseModel, Field, PrivateAttr, root_validator, ValidationError, validator

from spn.dto.utils import make_immutable

T = TypeVar('T')


def _orjson_dumps(val, *, default):
    # faster and more standardized json implementation
    option = orjson.OPT_APPEND_NEWLINE | orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
    return orjson.dumps(val, default=default, option=option).decode()


class ImprovedJsonModel(BaseModel):
    """Improved json serialization base using orjson.

    - fastest Python library for JSON
    - more correct than the standard json library
    - serializes dataclass, datetime, numpy, and UUID instances natively.

    """

    class Config:
        json_loads = orjson.loads
        json_dumps = _orjson_dumps
        json_encoders = {
            MappingProxyType: dict,
        }

    # def dict(self, *args, **kwargs):
    #     return super().dict(*args, **{"by_alias": True, **kwargs})

    # def json(self, *args, indent=True, **kwargs):
    #     return super().json(*args, indent=indent, **{"by_alias": True, **kwargs})

    def serialize(self) -> Dict:
        return self.dict()

    @classmethod
    def deserialize(cls, data: dict):
        return cls.parse_obj(data)

    # TODO: parse_raw, inflate, flatten


class ExcludeEmptyMixin(BaseModel):
    """Exclude `false` values from serialization.

    I.e None, [], "", 0, {}, etc

    """

    class Config:
        exclude_empty_keys = []

    def dict(self, *args, **kwargs):
        exclude = kwargs.get("exclude", dict()) or dict()
        if not isinstance(exclude, dict):
            exclude = {k: ... for k in exclude}
        for k in self.Config.exclude_empty_keys:
            if not getattr(self, k, None):
                exclude[k] = ...

        return super().dict(*args, **{"by_alias": True, **kwargs, 'exclude': exclude})


class ImmutableModel(ExcludeEmptyMixin, BaseModel):
    """Immutability base model.

    - Automatically convert `dict` and `list` to immutable counterparts
    - `update` function required to modify model, all mutation create new instances
    - avoid using `copy`

    """
    update_reasons: Tuple[str, ...] = Field(default_factory=tuple)

    class Config(ExcludeEmptyMixin.Config):
        # do not allow `instance.variable = value` assignments
        allow_mutation = False
        run_validation_on_update = True  # turning this off will give performance boost
        skip_update_validation = ['timeline']  # turning this off will give performance boost
        exclude_empty_keys = ExcludeEmptyMixin.Config.exclude_empty_keys + ['update_reasons']

    def __str__(self) -> str:
        string = f"{self.__class__.__name__}: {super().__str__()}"
        return string.replace("update_reasons=() ", "").replace("meta={} metrics={} ", "")

    @validator('*')  # TODO: should be done on the whole field dict, not individual attributes
    def auto_make_immutable(cls, v):  # noqa
        """Auto map all dict and list to their immutable counterparts."""
        return make_immutable(v)

    @classmethod
    def make_immutable(cls, v, skip=True):
        """Auto map all dict and list to their immutable counterparts."""
        if skip:
            v: dict = v.copy()
            skipped = {k: v.pop(k) for k in cls.Config.skip_update_validation if k in v}
            v: MappingProxyType = make_immutable(v)
            return MappingProxyType({**skipped, **v})
        return make_immutable(v)

    @property
    def version(self):
        return len(self.update_reasons)

    @property
    def reason(self):
        return self.update_reasons[-1] if self.update_reasons else None

    def update(self: T, **kwargs) -> T:
        """Initiate a copy of the current DTO with new values."""
        reason: str = kwargs.get("reason", f"update-{self.version + 1}")
        kwargs = self.make_immutable(kwargs, skip=True)

        if self.Config.run_validation_on_update:
            # slower, but maybe more accurate
            # https://github.com/samuelcolvin/pydantic/issues/418#issuecomment-974980947
            copy = self.copy(update={
                "update_reasons": (*self.update_reasons, reason),
                **kwargs,
            })
            # noqa # pylint: disable=protected-access
            return self.validate(dict(copy._iter(to_dict=False, by_alias=False, exclude_unset=True)))
            # return self.validate(
            #     {
            #         "update_reasons": (*self.update_reasons, reason),
            #         **self.dict(),
            #         **kwargs,
            #     }
            # )

        # this might be faster than validating all attributes on each update / copy
        try:
            k = {k: v for k, v in kwargs.items() if k not in self.Config.skip_update_validation}
            self.validate(k)
        except ValidationError as e:
            errors = e.errors()
            if any(error['type'] != 'value_error.missing' for error in errors):
                raise e

        return self.copy(update={
            "update_reasons": (*self.update_reasons, reason),
            **kwargs,
        })


class TimestampedModel(ImmutableModel):
    _time_loaded: str = PrivateAttr(default_factory=lambda: datetime.now().isoformat())
    _time_updated: Optional[str] = PrivateAttr(default=None)

    class Config:
        underscore_attrs_are_private = True

    @property
    def time_loaded(self):
        return self._time_loaded

    @property
    def time_updated(self):
        return self._time_updated

    def update(self: T, **kwargs) -> T:
        result = super().update(**kwargs)
        # noqa # pylint: disable=protected-access
        result._time_updated = datetime.now().isoformat()
        # noqa # pylint: disable=protected-access
        result._time_loaded = self._time_loaded
        return result


class HashableModel(BaseModel):
    """Adds key and hash capability to models, but:

    - Model should have the attribute/property named `__key__`
    - HashableModel should be first in inheritance declaration (last in resolution as per python syntax)

    This will be used for hashing and equality checks.
    The `__key__` value for a document could be the filename
    and for layer elements, such as words and chapters, their start-end timestamp tuple.
    For experiment results add the model and run name string.

    """

    class Config:
        # short keys to abbreviate results in tables and such
        # not used for equality and hashing
        # TODO: current collision likelihood 1 in 16 million, could be replaced by b64 for better odds
        short_key_size: int = 6
        unique: List[str] = []  # field names which need to be unique validated
        sorted: List[str] = []  # field names which need to be ordered

    @property
    def key(self):
        return self.__key__  # noqa

    @root_validator
    @classmethod
    def unique_validation(cls, values):
        for key in cls.Config.unique:
            if key in values:
                arr: List['HashableModel'] = values[key]
                a, b = len(arr), len(set(arr))
                if a != b:
                    raise ValueError(f'{a} != {b}: field "{key}" values must be unique!')
        return values

    @root_validator
    @classmethod
    def sorted_modifier(cls, values):
        for key in cls.Config.sorted:
            if key in values:
                arr: List['HashableModel'] = values[key]
                values[key] = sorted(arr, key=cls.sorting_key)
        return values

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.short_key}>"

    def __eq__(self, other):
        return self.key == other.key and isinstance(other, self.__class__)
        # return isinstance(other, self.__class__) and HashableModel.__hash__(self) == HashableModel.__hash__(other)

    def __hash__(self):
        return hash(self.key)

    def sorting_key(self):
        # this can also be used as a key method for `sorting` calls:
        # sorted(array, key=HashableModel.sorting_key) # as per python language features
        return self.key

    @classmethod
    def make_short_key(cls, key: str):
        """Short keys to abbreviate results in tables and such."""
        return sha256(str(key).encode()).hexdigest()[:cls.Config.short_key_size]

    @property
    def short_key(self):
        return self.make_short_key(self.key)


class ExperimentMixin(ImmutableModel):
    meta: Dict = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config(ImmutableModel.Config):
        exclude_empty_keys = ImmutableModel.Config.exclude_empty_keys + ['meta', 'metrics']

    def update_meta(self, meta: Dict[str, Any], reason="update meta", full=False):
        """Convenience function to update meta field."""
        return self.update(reason=reason, meta=meta if full else {**self.meta, **meta})

    def update_metrics(self, metrics: Dict[str, Any], reason="update metrics", full=False):
        """Convenience function to update metrics field."""
        return self.update(reason=reason, metrics=metrics if full else {**self.metrics, **metrics})


class DataTransferObject(HashableModel, ExperimentMixin, ImprovedJsonModel, ImmutableModel):
    class Config(
        HashableModel.Config,
        ExperimentMixin.Config,
        ImprovedJsonModel.Config,
        ImmutableModel.Config,
    ):
        pass
