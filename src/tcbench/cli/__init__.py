import pathlib
import sys
import logging

from rich.console import Console, RenderableType
from rich.logging import RichHandler
from rich.protocol import rich_cast
from rich.theme import Theme

def get_rich_console(
    fname: pathlib.Path | None = None,    
    log_time: bool = False,
    record: bool = False,
) -> Console:
    curr_module = sys.modules[__name__]
    folder_module = pathlib.Path(curr_module.__file__).parent

    file=fname
    if fname is not None:
        fname = pathlib.Path(fname)
        if not fname.parent.exists():
            fname.parent.mkdir(parents=True)
        file = open(fname, "w")

    return Console(
        theme=Theme.read(str(folder_module / "rich.theme")),
        log_time=log_time,
        log_path=False,
        file=file,
        record=record,
    )

class ConsoleLogger:
    def __init__(
        self,
        fname: pathlib.Path | None  = None,
    ):
        self.console = get_rich_console(record=True)
        self._extra_consoles = dict()
        if fname:
            self._extra_consoles["GLOBAL"] = get_rich_console(
                fname, log_time=True
            )

    def log(
        self, 
        obj: str | RenderableType, 
        echo: bool = True,
        file_shortname: str | None = None
    ) -> None:
        if echo:
            self.console.print(obj)

        if len(self._extra_consoles) > 0:
            if file_shortname is None:
                file_shortname = "GLOBAL"
            console = self._extra_consoles.get(file_shortname, None)
            if console is not None:
                console.log(obj)
            elif file_shortname != "GLOBAL":
                raise RuntimeError(
                    f"Console file {file_shortname} not registered")

    def register_new_file(
        self, 
        path: pathlib.Path, 
        shortname: str,
        with_log_time: bool = True 
    ) -> None:
        if shortname in self._extra_consoles:
            raise RuntimeError(f"Console file '{shortname}' already registered")
        self._extra_consoles[shortname] = get_rich_console(
            path, log_time=with_log_time
        )

    def unregister_file(
        self,
        shortname: str
    ) -> None:
        console = self._extra_consoles.get(shortname, None)
        if console is None:
            raise RuntimeError(f"Console file '{shortname}' not found")
        if console.file is not None:
            console.file.flush()
            console.file.close()
        del(self._extra_consoles[shortname])
        

    def save_svg(self, save_as: pathlib.Path, title: str = "") -> None:
        self.console.save_svg(str(save_as), title=title)


logger = ConsoleLogger()
console = logger.console


def reset_logger(fname: pathlib.Path | None = None) -> ConsoleLogger:
    global logger
    global console
    logger = ConsoleLogger(fname)
    console = logger.console
    return logger
