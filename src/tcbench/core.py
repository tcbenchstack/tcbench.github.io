from __future__ import annotations

import polars as pl

import multiprocessing
import functools
import pathlib

from typing import Any, Callable, Iterable
from enum import Enum

from tcbench import fileutils

class StringEnum(Enum):
    @classmethod
    def from_str(cls, text) -> StringEnum:
        for member in cls.__members__.values():
            if member.value == text:
                return member
        raise ValueError(f"Invalid enumeration {text}")

    @classmethod
    def values(cls):
        return [x.value for x in list(cls)]

    def __str__(self):
        return self.value

class Pool1N:
    def __init__(
        self, 
        processes: int,
        maxtasksperchild: int | None = None
    ):
        self.processes = processes
        self._pool = None
        if processes > 1:
            self._pool = multiprocessing.get_context("spawn").Pool(
                processes=processes,
                maxtasksperchild=maxtasksperchild,
            )

    def __enter__(self) -> Any:
        if self._pool is not None:
            return self._pool.__enter__()
        return self

    def __exit__(self, *args) -> Any:
        if self._pool:
            return self._pool.__exit__(*args)

    def imap_unordered(self, func: Callable, data: Iterable) -> Any:
        if self._pool is None:
            for items in data:
                res = func(items)
                yield res
        else:
            for res in self._pool.imap_unordered(func, data):
                yield res


def to_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    return df

def save_params(
    path_param: str, 
    split_index_param: str = "", 
    *,
    echo: bool = True
):
    def _save(*args, **kwargs):
        save_to = kwargs.get(path_param, None) 
        split_index = kwargs.get(split_index_param, None)
        if save_to is None:
            return 

        save_to = pathlib.Path(save_to)
        if split_index is not None:
            save_to /= f"split_{split_index:02d}"

        data = kwargs.copy()
        if len(args) > 0:
            data["args"] = args
        del(data[path_param])
        fileutils.save_yaml(data, save_to / "params.yml", echo)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _save(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator
