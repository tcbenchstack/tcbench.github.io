from __future__ import annotations

from rich import box
from rich.table import Table, Column
from typing import Any, List


SYMBOL_HIGHLIGHT_ROW = ":left_arrow:"
SYMBOL_ELIPSIS = "..."
SYMBOL_ARROW_UP = "↑"
SYMBOL_ARROW_DOWN = "↓"


class LiveTableColumnSummary:
    def __init__(self, name: str, fmt: str):
        self.name = name
        self.fmt = fmt
        self._state = None

    def format(self) -> str:
        text = f"({self.name})"
        text += "\n"
        if not self.fmt:
            return text + str(self._state)
        return text + format(self._state, self.fmt)

    def update(self, value: Any) -> str:
        self._state = value
        return self.format()


class LiveTableColumnSummaryMin(LiveTableColumnSummary):
    def __init__(self, fmt: str):
        super().__init__(name="min", fmt=fmt)

    def update(self, value: Any) -> str:
        if (
            self._state is None
            or self._state > value
        ):
            self._state = value
        return self.format()


class LiveTableColumnSummaryMax(LiveTableColumnSummary):
    def __init__(self, fmt: str):
        super().__init__(name="max", fmt=fmt)

    def update(self, value: Any) -> str:
        if (
            self._state is None
            or self._state < value
        ):
            self._state = value
        return self.format()


class LiveTableColumnSummaryAvg(LiveTableColumnSummary):
    def __init__(self, fmt: str):
        super().__init__(name="avg", fmt=fmt)
        self._state = 0
        self._cum = 0
        self._count = 0

    def update(self, value: Any) -> str:
        self._count += 1
        self._cum += value
        self._state = self._cum / self._count
        return self.format()


COLUMN_SUMMARY = {
    "min": LiveTableColumnSummaryMin,
    "max": LiveTableColumnSummaryMax,
    "avg": LiveTableColumnSummaryAvg,
}


class LiveTableColumn(Column):
    def __init__(
        self, 
        name: str,
        fmt: str | None = None,
        justify: str = "right",
        track: str | None = None,      # min, max
        summary: str | None = None,    # min, max, avg
        **kwargs
    ):
        self.name = name
        self.fmt = fmt
        self.track = track
        self.values = []
        self._last_tracked_idx = None
        self._last_summary = None

        self._column_summary = None
        if summary:
            self._column_summary = COLUMN_SUMMARY[summary](self.fmt)

        header = name
        if track:
            symbol = SYMBOL_ARROW_DOWN if track == "min" else SYMBOL_ARROW_UP
            header = symbol + header
            if justify == "left":
                header = header + symbol
        kwargs["header"] = header
        kwargs["justify"] = justify

        super().__init__(**kwargs)
        if self._column_summary:
            self.show_footer = True

    def format(self, value:Any) -> str:
        if self.fmt:
            return format(value, self.fmt) 
        return str(value)

    def _reformat_cell(self, cell_idx: int | None) -> None:
        if cell_idx is not None:
            self._cells[cell_idx] = self.format(self.values[cell_idx])

    def append(self, value: Any) -> str:
        self.values.append(value)
        text = self.format(value)

        if self.track and getattr(self, f"_is_latest_global_{self.track}"):
            text = f"[bold]{text}[/bold]"
            if self.justify == "right":
                text = ":arrow_forward: " + text
            else:
                text = text + " :arrow_backward:"
            self._reformat_cell(self._last_tracked_idx)
            self._last_tracked_idx = len(self.values) - 1

        if self._column_summary:
            self.footer = self._column_summary.update(value)
        return text

    @property
    def _is_latest_global_max(self) -> bool:
        return (
            self._last_tracked_idx is None
            or (self.values[-1] > self.values[self._last_tracked_idx])
        )

    @property
    def _is_latest_global_min(self) -> bool:
        return (
            self._last_tracked_idx is None
            or (self.values[-1] < self.values[self._last_tracked_idx])
        )

    @property
    def _is_lastest_tracked(self) -> bool:
        return self._last_tracked_idx == len(self.values) - 1

    @property
    def has_summary(self) -> bool:
        return self._column_summary is not None


class LiveTable(Table):
    def __init__(
        self, 
        columns: List[LiveTableColumn],
        show_last_rows: int | None = None,
    ):
        self.show_last_rows = show_last_rows
        self._perf_columns = None

        super().__init__(
            box=box.SIMPLE_HEAVY, 
            show_footer=any(col.has_summary for col in columns),
            show_edge=False,
            pad_edge=False,
            padding=(0, 1, 0, 0),
        )

        for idx, col in enumerate(columns):
            col._index = idx
            self.columns.append(col)

    def add_row(self, *args, **kwargs) -> None:
        renderables = []
        for col in self.columns:
            raw_col_value = kwargs.get(col.name, "")
            text = ""
            if raw_col_value != "":
                text = col.append(raw_col_value)
            renderables.append(text)
        super().add_row(*renderables)
