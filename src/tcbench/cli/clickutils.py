from __future__ import annotations

import rich_click as click

from typing import List, Dict, Any, Callable, Sequence, Tuple

import functools
import pathlib
import sys

from tcbench import fileutils
from tcbench.core import StringEnum
from tcbench.datasets import (
    DATASET_NAME,
    DATASET_TYPE,
)
from tcbench.modeling import (
    MODELING_METHOD_NAME, 
    MODELING_INPUT_REPR_TYPE
)
from tcbench.modeling.enums import MODELING_FEATURE


def _create_choice(values: List[str]) -> click.Choice:
    return click.Choice(values, case_sensitive=False)

def _parse_enum_from_str(
    command: str, 
    parameter:str, 
    value: str | None, 
    from_str: Callable, 
) -> StringEnum | None:
    if value == "" or value is None:
        return None
    return from_str(value)

def _parse_str_to_int(command: str, parameter: str, value: str) -> int:
    return int(value)


CHOICE_DATASET_NAME = _create_choice(DATASET_NAME.values())
parse_dataset_name = functools.partial(
    _parse_enum_from_str, from_str=DATASET_NAME.from_str
)

CHOICE_DATASET_TYPE = _create_choice(DATASET_TYPE.values())
parse_dataset_type = functools.partial(
    _parse_enum_from_str, from_str=DATASET_TYPE.from_str
)

CHOICE_MODELING_METHOD_NAME = _create_choice(MODELING_METHOD_NAME.values())
parse_modeling_method_name = functools.partial(
    _parse_enum_from_str, from_str=MODELING_METHOD_NAME.from_str
)

CHOICE_MODELING_FEATURE = _create_choice(MODELING_FEATURE.values())
parser_modeling_feature = functools.partial(
    _parse_enum_from_str, from_str=MODELING_FEATURE.from_str
)



def _parse_range(text: str) -> List[Any]:
    parts = list(map(float, text.split(":")))
    if len(parts) == 1:
        return parts
    
    import numpy as np
    return np.arange(*parts).tolist()


def parse_raw_text_to_list(
    command: str | None, 
    parameter: str | None, 
    value: Tuple[str]
) -> Tuple[Any] | None:
    """Parse a coma separated text string into the associated list of values.
       The list can be a combination of string, numeric or range values
       in the format first:last or first:last:step. In the latter two cases,
       the range are expanded into the associated formats.

       Examples:
        "1,a"       -> (1.0, "a")
        "0:3,a"     -> (0.0, 1.0, 2.0, "a")
        "0:2:0.5,a" -> (0.0, 0.5, 1.0, 1.5, "a")
    """
    text = "".join(value)
    if text == "" or text is None:
        return None
    values = []
    for token in text.split(","):
        if token.isnumeric():
            func = float
            if '.' not in token:
                func = int
            values.append(func(token))
        elif ":" in token:
            values.extend(_parse_range(token))
        else:
            values.append(token)
    return tuple(values)


def parse_raw_text_to_list_int(
    command: str, 
    parameter: str, 
    value: Tuple[str]
) -> Tuple[int] | None:
    data = parse_raw_text_to_list(command, parameter, value)
    if data is None:
        return None
    return tuple(map(int, data))


def parse_remainder(
    command: str, 
    argument: str, 
    values: Tuple[str]
) -> Dict[str, Tuple[Any]]:
    params = dict()
    for text in values:
        if text == "--":
            continue
        param_name, param_value = text.split("=")
        params[param_name] = parse_raw_text_to_list(None, None, (param_value,))
    return params

#CLICK_CHOICE_INPUT_REPR = _create_choice(MODELING_INPUT_REPR_TYPE)
#CLICK_PARSE_INPUT_REPR = functools.partial(_parse_enum_from_str, enumeration=MODELING_INPUT_REPR_TYPE)

CLICK_PARSE_STRTOINT = _parse_str_to_int


def save_commandline(
    cli_option: str, 
    *,
    cli_skip_option: str = "", 
    echo: bool = True
) -> Any:
    """
    Decorator to save the command line retrieved from sys.argv.
    This is meant to be associated to Click command having an option
    for saving output to disk. As such, the decorator requires
    as input the name of such command line option.
    """
    def _save(save_to: pathlib.Path | None, echo: bool = True) -> None:
        if save_to is not None:
            save_to = pathlib.Path(save_to)
            cmd = " ".join(sys.argv) + "\n"
            fileutils.save_txt(cmd, save_to / "command.txt", echo)

    def _decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            skip = kwargs.get(cli_skip_option, False)
            if not skip:
                _save(kwargs.get(cli_option, None), echo)
            return func(*args, **kwargs)
        return wrapper

    return _decorator































def compose_help_string_from_list(message: str, items: Sequence[str]) -> str:
    """Compose a string from a list"""
    message = message.strip()
    if message[-1] != ".":
        message += "."
    return f"""{message}\nValues: [{"|".join(items)}]."""


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
