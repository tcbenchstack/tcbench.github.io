from __future__ import annotations

import pandas as pd
import rich.progress as richprogress
import rich.columns as richcolumns
import rich.text as richtext

from rich.table import Table
from rich.panel import Panel
from rich import box
from typing import List

import sys

from tcbench import cli

console = cli.console

PDB_DETECTED="pdb" in sys.modules

BULLET = "•"



def _rich_table_from_series(ser:pd.Series, columns:List[str], with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas Series"""
    table = Table()
    table.add_column(columns[0])
    table.add_column(columns[1], justify="right")
    for index, value in zip(ser.index, ser.values):
        table.add_row(str(index), str(value))
    if with_total:
        table.add_section()
        table.add_row("__total__", str(ser.values.sum()))
    return table


def _rich_table_from_series_multiindex(ser:pd.Series, with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas Series with MultiIndex"""
    table = Table()
    for col in ser.index.names:
        table.add_column(col)
    table.add_column("samples", justify="right")

    for level0 in ser.index.levels[0]:
        tmp = ser.loc[level0]
        for level1, value in zip(tmp.index, tmp.values):
            table.add_row(str(level0), str(level1), str(value))
            level0 = ""
        if with_total:
            table.add_row("", "__total__", str(tmp.values.sum()))
        table.add_section()
    return table


def _rich_table_from_dataframe(df:pd.DataFrame, with_total:bool=False) -> rich.table.Table:
    """Compose a rich Table from a pandas DataFrame"""
    table = Table()
    table.add_column(df.index.name)
    for col in df.columns:
        table.add_column(col, justify="right")
    for idx in range(df.shape[0]):
        table.add_row(df.index[idx], *list(df.iloc[idx].values.astype(str)))
    if with_total:
        table.add_section()
        table.add_row("__total__", *list(map(str, df.sum().values)))
    return table


def rich_samples_count_report(obj:pd.Series | pd.DataFrame, columns:List[str]=None, title:str=None, with_total:bool=True) -> rich.table.Table:
    """Compute and format into a table the per-class samples count"""
    if columns is None:
        columns = ["app", "samples"]
    if isinstance(obj, pd.Series):
        if isinstance(obj.index, pd.MultiIndex):
            table = _rich_table_from_series_multiindex(obj, with_total=with_total)
        else:
            table = _rich_table_from_series(obj, columns, with_total=with_total)
    else:
        table = _rich_table_from_dataframe(obj, with_total=with_total)

    if title is not None:
        console.print(title)
    console.print(table)


def rich_splits_report(df:pd.DataFrame, df_split:pd.DataFrame, split_index:int=None, title:str=None, with_total:bool=True) -> rich.table.Table:
    min_split_idx = 0
    max_split_idx = df_split["split_index"].max()
    _title = f"split_index={min_split_idx} to {max_split_idx}"
    if split_index is not None:
        min_split_idx = int(split_index)
        _title = f"split_index={split_index}"

    df = df.copy()
    df = df.set_index("row_id", drop=False)
    ser_split = df_split[df_split["split_index"] == min_split_idx]
    df_tmp = pd.DataFrame(
        [
            df.loc[ser_split["train_indexes"].values[0]]["app"].value_counts(),
            df.loc[ser_split["val_indexes"].values[0]]["app"].value_counts(),
            df.loc[ser_split["test_indexes"].values[0]]["app"].value_counts(),
        ],
        index=["train_samples", "val_samples", "test_samples"],
    ).T
    df_tmp = df_tmp.assign(all_samples=df_tmp.sum(axis=1))
    if title:
        _title = title
    rich_samples_count_report(df_tmp, title=title, with_total=with_total)


def rich_packets_report(
    df: pd.DataFrame, 
    packets_colname: str = "packets", 
    title: str = None
) -> rich.table.Table:
    """Compute and reports stats for number of packets per flow"""
    ser = df[packets_colname].describe().round(2)
    rich_samples_count_report(
        ser, columns=["stat", "value"], title=title, with_total=False
    )


def rich_label(text:str, extra_new_line:bool=False) -> None:
    """Output on the console a formatted label"""
    if extra_new_line:
        console.print()
    console.print(Panel(text, box=box.ROUNDED, expand=False, padding=0))


class SpinnerProgress(richprogress.Progress):
    def __init__(self, description: str = "", visible: bool = True):
        if description:
            description = f"{BULLET} {description}"
        self.description = description
        self.visible = visible
        self.task_id = None

        if self.visible:
            super().__init__(
                richprogress.SpinnerColumn(),
                richprogress.TimeElapsedColumn(),
                richprogress.TextColumn(
                    "[progress.description]{task.description}"
                ),
                transient=False,
                console=console,
            )
            self.task_id = self.add_task(description=description)

    def __enter__(self):
        if self.visible and not PDB_DETECTED:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            return False
        if self.visible and not PDB_DETECTED:
            description = ""
            if self.description is not None:
                description = f"{self.description} Done!"
            # remove Spinner and MofN columns
            self.columns = self.columns[1:]
            self.update(self.task_id, description=description, refresh=True)
            self.stop()
        return True

    def update(self, *args, **kwargs):
        kwargs["refresh"] = True
        super().update(*args, **kwargs)


class MofNCompleteColumn(richprogress.MofNCompleteColumn):
    def render(self, task) -> richtext.Text:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return richtext.Text(
            f"({completed + 1}{self.separator}{total})",
            style="progress.download",
        )

class SpinnerAndCounterProgress(richprogress.Progress):
    def __init__(
        self, 
        total:int, 
        description: str = "", 
        steps_description: List[str] = None, 
        visible: bool = True, 
        newline_after_update: bool = False
    ):
        self.visible = visible
        self.description = description
        self.steps_description = steps_description
        self.task_id = None
        self._col_inner_text = None
        self._col_mofn = None
        self.newline_after_update = newline_after_update

        if self.visible: 
            self._col_inner_text = richprogress.TextColumn("")
            #self._col_mofn = richprogress.MofNCompleteColumn()
            self._col_mofn = MofNCompleteColumn()
            super().__init__(
                richprogress.SpinnerColumn(),
                richprogress.TimeElapsedColumn(),
                richprogress.TextColumn(BULLET),
                richprogress.TextColumn("[progress.description]{task.description}"),
                self._col_mofn,
                self._col_inner_text,
                transient=False,
                console=console,
            )
            self.task_id = self.add_task(description=description, total=total)

    def __enter__(self) -> SpinnerAndCounterProgress:
        if self.visible and not PDB_DETECTED:
            self.start()
            if self.steps_description:
                self._col_inner_text.text_format = self.steps_description.pop(0)
                super().update(self.task_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            return False
        if self.visible and not PDB_DETECTED:
            description = self.description
            if description:
                self._col_inner_text.text_format = "Done!"
                # remove Spinner and MofN columns
                self.columns = (*self.columns[1:4], *self.columns[5:])
                super().update(self.task_id, description=description)
            self.stop()
        return True    

    def update_total(self, total: int) -> None:
        if self.visible and not PDB_DETECTED:
            super().update(self.task_id, total=total, refresh=True)

    def update(self, *args, **kwargs) -> None:
        if self.visible and not PDB_DETECTED:
            if self.newline_after_update:
                table = list(self.get_renderables())[0]
                renderables = [
                    col._cells[0]
                    for col in table.columns[1:]
                ]
                console.print(*renderables)
            if self.steps_description:
                self._col_inner_text.text_format = self.steps_description.pop(0)
            super().advance(self.task_id, *args, **kwargs)


class FileDownloadProgress(richprogress.Progress):
    def __init__(self, totalbytes: int, visible: bool = True): 
        super().__init__(
            richprogress.BarColumn(),
            richprogress.FileSizeColumn(),
            richprogress.TextColumn("/"),
            richprogress.TotalFileSizeColumn(),
            richprogress.TextColumn("eta"),
            richprogress.TimeRemainingColumn(),
            richprogress.TextColumn(f"{BULLET} Downloading..."),
            console=console,
        )
        self.visible = visible
        self.task_id = self.add_task(
            "", 
            total=totalbytes, 
        )

    def __enter__(self):
        if self.visible and not PDB_DETECTED:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            return False
        if self.visible and not PDB_DETECTED:
            self.stop()
        return True

    def update(self, *args, **kwargs):
        #kwargs["refresh"] = True
        if not PDB_DETECTED:
            super().advance(self.task_id, *args, **kwargs)

class Progress(richprogress.Progress):
    def __init__(self, total: int, description: str = "", visible: bool = True):
        if description:
            description = f"{BULLET} {description}"
        self.visible = visible
        self.description = description
        self.task_id = None

        if self.visible:
            super().__init__(
                richprogress.BarColumn(),
                richprogress.MofNCompleteColumn(),
                richprogress.TimeElapsedColumn(),
                richprogress.TextColumn("eta"),
                richprogress.TimeRemainingColumn(),
                richprogress.TextColumn("[progress.description]{task.description}"),
                console=console,
            )
            self.task_id = self.add_task(description, total=total)

    def __enter__(self):
        if self.visible and not PDB_DETECTED:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            return False
        if self.visible and not PDB_DETECTED:
            if self.description:
                self.columns = (self.columns[2], self.columns[5])
                description = f"{self.description} Done!"
                super().update(self.task_id, description=description, refresh=True)
            self.stop()
        return True

    def update_description(self, description:str = "") -> None:
        if self.visible and not PDB_DETECTED and self.task_id is not None:
            self.description = description
            super().update(self.task_id, description=description, refresh=True)

    def update(self):
        if self.visible and not PDB_DETECTED:
            super().advance(self.task_id, advance=1)
