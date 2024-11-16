from __future__ import annotations

import polars as pl

import multiprocessing

from typing import Any, Callable, Iterable
from enum import Enum

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
