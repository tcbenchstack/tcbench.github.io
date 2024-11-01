from __future__ import annotations

import rich_click as click

from typing import List, Dict, Any

import functools

from tcbench.core import StringEnum
from tcbench.datasets import (
    DATASET_NAME,
    DATASET_TYPE,
)
from tcbench.modeling import (
    MODELING_METHOD_NAME, 
    MODELING_INPUT_REPR_TYPE
)


def _create_choice(enumeration:StringEnum) -> click.Choice:
    return click.Choice(enumeration.values(), case_sensitive=False)

def _parse_enum_from_str(command: str, parameter:str, value:str, enumeration:StringEnum) -> StringEnum:
    return enumeration.from_str(value)

def _parse_str_to_int(command: str, parameter: str, value: str) -> int:
    return int(value)


CHOICE_DATASET_NAME = _create_choice(DATASET_NAME)
parse_dataset_name = functools.partial(_parse_enum_from_str, enumeration=DATASET_NAME)

CHOICE_DATASET_TYPE = _create_choice(DATASET_TYPE)
parse_dataset_type = functools.partial(_parse_enum_from_str, enumeration=DATASET_TYPE)

CHOICE_MODELING_METHOD_NAME = _create_choice(MODELING_METHOD_NAME)
parse_modeling_method_name = functools.partial(_parse_enum_from_str, enumeration=MODELING_METHOD_NAME)


def _parse_range(text: str) -> List[Any]:
    parts = list(map(float, text.split(":")))
    if len(parts) == 1:
        return parts
    
    import numpy as np
    return np.arange(*parts).tolist()


def parse_raw_text_to_list(command: str, parameter: str, value: Tuple[str]) -> Tuple[Any]:
    """Parse a coma separated text string into the associated list of values.
       The list can be a combination of string, numeric or range values
       in the format first:last or first:last:step. In the latter two cases,
       the range are expanded into the associated formats.

       Examples:
        "1,a"       -> (1.0, "a")
        "0:3,a"     -> (0.0, 1.0, 2.0, "a")
        "0:2:0.5,a" -> (0.0, 0.5, 1.0, 1.5, "a")
    """
    value = "".join(value)
    if value == "" or value is None:
        return None
    l = []
    for text in value.split(","):
        if text.isnumeric():
            func = float
            if '.' not in text:
                func = int
            l.append(func(text))
        elif ":" in text:
            l.extend(_parse_range(text))
        else:
            l.append(text)
    return tuple(l)


def parse_raw_text_to_list_int(command: str, parameter: str, value: Tuple[str]) -> Tuple[int]:
    data = parse_raw_text_to_list(command, parameter, value)
    if data is None:
        return None
    return tuple(map(int, data))


def parse_remainder(command: str, argument: str, value: Tuple[str]) -> Dict[str, Any]:
    opts = dict()
    for text in value:
        key, val = text.split("=")
        opts[key] = parse_raw_text_to_list(None, None, val)
    return opts

#CLICK_CHOICE_METHOD_NAME = _create_choice(MODELING_METHOD_TYPE)
#CLICK_PARSE_METHOD_NAME = functools.partial(_parse_enum_from_str, enumeration=MODELING_METHOD_TYPE)

#CLICK_CHOICE_INPUT_REPR = _create_choice(MODELING_INPUT_REPR_TYPE)
#CLICK_PARSE_INPUT_REPR = functools.partial(_parse_enum_from_str, enumeration=MODELING_INPUT_REPR_TYPE)

CLICK_PARSE_STRTOINT = _parse_str_to_int


def compose_help_string_from_list(items:List[str]) -> str:
    """Compose a string from a list"""
    return "\[" + f'{"|".join(items)}' + "]."


def convert_params_dict_to_list(params:Dict[str,Any], skip_params:List[str]=None) -> List[str]:
    """Convert a dictionary of parameters (name,value) pairs into a list of "--<param-name> <param-value>"""
    if skip_params is None:
        skip_params = set()

    l = []
    for par_name, par_value in params.items():
        if par_name in skip_params or par_value == False or par_value is None:
            continue
        par_name = par_name.replace("_", "-")
        if par_value == True:
            l.append(f"--{par_name}")
        else:
            l.append(f"--{par_name} {str(par_value)}")

    return l


def help_append_choices(help_string:str, values:List[str]) -> str:
    """Append to an help string a styled version of a list of values"""
    text = "|".join([f"[bold]{text}[/bold]" for text in values])
    return f"{help_string} [yellow]Choices: [{text}][/yellow]"
