from __future__ import annotations

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

class MultiprocessingWorkerKWArgs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
