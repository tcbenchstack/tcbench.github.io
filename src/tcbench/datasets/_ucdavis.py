from __future__ import annotations

import polars as pl
import numpy as np

import pathlib
import multiprocessing
import functools

import tcbench
from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.datasets.core import (
    Dataset,
    BaseDatasetProcessingPipeline,
    SequentialPipelineStage,
    _remove_fields_from_schema,
)
from tcbench.datasets import (
    DATASET_NAME, 
    DATASET_TYPE,
    curation,
    catalog
)

PARTITION_PRETRAINING = "pretraining"
PARTITION_SCRIPT = "retraining-script-triggered"
PARTITION_HUMAN = "retraining-human-triggered"


def _parse_raw_txt_worker(
    path: pathlib.Path, 
    schema: pl.Schema
) -> pl.DataFrame:

    raw_schema = _remove_fields_from_schema(
        schema, 
        "fname", "folder", "fname_row_id"
    )
    return (
        pl.read_csv(
            path, 
            separator="\t", 
            schema=raw_schema
        )
        .with_columns(
            fname=pl.lit(path.name),
            folder=pl.lit("/".join(path.parts[-3:-1])),
        )
        .with_row_index("fname_row_id")
        .select(
            *list(schema.keys())
        )
    )


def load_raw_txt(
    path: pathlib.Path, 
) -> pl.DataFrame:
    schema = catalog.get_dataset_polars_schema(
        DATASET_NAME.UCDAVIS19,
        DATASET_TYPE.RAW
    )
    return _parse_raw_txt_worker(path, schema)


class RawTXTParser:
    def __init__(self, save_to: pathlib.Path, num_workers: int = 1):
        self.save_to = save_to
        self.num_workers = tcbench.validate_num_workers(num_workers)

    def run(self, *paths: Pathlib.Path) -> pl.DataFrame:
        schema = catalog.get_dataset_polars_schema(
            DATASET_NAME.UCDAVIS19,
            DATASET_TYPE.RAW
        )

        with (
            richutils.Progress(
                description="Parsing raw txt files...", 
                total=len(paths),
            ) as progress,
            (
                multiprocessing
                .get_context("spawn")
                .Pool(processes=self.num_workers) 
            ) as pool,
        ):
            func = functools.partial(_parse_raw_txt_worker, schema=schema)
            data = []
            for df in pool.imap_unordered(func, paths):
                data.append(df)
                progress.update()
            df = (
                pl
                .concat(data)
                .sort(
                    "folder",
                    "fname",
                    "fname_row_id",
                )
            )
        return df


class RawPostprocessingPipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            dataset_name=DATASET_NAME.UCDAVIS19,
            description="Postprocess raw...",
            save_to=save_to,
        )

        stages = [
            SequentialPipelineStage(
                self._compose_flows,
                "Compose flows",
            ),
            SequentialPipelineStage(
                self._rename_columns,
                "Rename columns",
            ),
            SequentialPipelineStage(
                self._adjust_direction,
                "Adjust series direction"
            ),
            SequentialPipelineStage(
                self._add_columns,
                "Adding columns",
            ),
            SequentialPipelineStage(
                self.compute_stats,
                "Computing stats",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self.write_parquet_files,
                ),
                "Writing parquet files",
            )
        ]
        self.clear()
        self.extend(stages)

    def _compose_flows(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df
            .group_by("folder", "fname", maintain_order=True)
            .agg(
                "unixtime",
                "timetofirst",
                "packet_size",
                "packet_dir",
            )
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "unixtime": "pkts_timestamp",
            "timetofirst": "pkts_timetofirst",
            "packet_size": "pkts_size",
            "packet_dir": "pkts_dir",
        })

    def _adjust_direction(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0)
                    .then(pl.lit(-1))
                    .otherwise(pl.lit(1))
                )
            )
        )

    def _add_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        def compute_duration(data, direction=1):
            indices = np.where(np.atleast_1d(data["pkts_dir"]) == direction)[0]
            if len(indices) < 2:
                return 0
            first_idx, last_idx = indices[0], indices[-1]
            arr = data["pkts_timestamp"]
            return arr[last_idx] - arr[first_idx]
        
        return (
            df
            .with_columns(
                bytes=pl.col("pkts_size").list.sum(),
                packets=pl.col("pkts_size").list.len(),
                duration=(
                    pl.col("pkts_timestamp").list.last() 
                    - pl.col("pkts_timestamp").list.first()
                ),
                proto=pl.lit("udp"),
                pkts_size_times_dir=curation.expr_pkts_size_times_dir(),
                # this is required to run the statistics
                # but is removed at the end of the curation
                is_valid_handshake=pl.lit(True),
                partition=(
                    pl.col("folder")
                    .str.split("/")
                    .list.get(0)
                    .str.replace(")", "", literal=True)
                    .str.replace("(", "-", literal=True)
                    .str.to_lowercase()
                ),
                app=(
                    pl.col("fname")
                    .str.split("-")
                    .list.get(0)
                    .str.to_lowercase()
                ),
            ).with_columns(
                packets_upload=(
                    pl.col("pkts_size_times_dir").list.eval(
                        pl.element() > 0
                    ).list.sum()
                ),
                packets_download=(
                    pl.col("pkts_size_times_dir").list.eval(
                        pl.element() < 0
                    ).list.sum()
                ),
                bytes_upload=(
                    pl.col("pkts_size_times_dir").list.eval(
                        pl.when(pl.element() > 0)
                        .then(pl.element())
                        .otherwise(pl.lit(0))
                    ).list.sum()
                ),
                bytes_download=(
                    pl.col("pkts_size_times_dir").list.eval(
                        pl.when(pl.element() < 0)
                        .then(-pl.element())
                        .otherwise(pl.lit(0))
                    ).list.sum()
                ),
                duration_upload=(
                    pl.struct(
                        "pkts_timestamp", 
                        "pkts_dir"
                    ).map_elements(
                        function=functools.partial(
                            compute_duration, 
                            direction=1
                        ),
                        return_dtype=pl.Float64
                    )
                ),
                duration_download=(
                    pl.struct(
                        "pkts_timestamp", 
                        "pkts_dir"
                    ).map_elements(
                        function=functools.partial(
                            compute_duration, 
                            direction=-1
                        ),
                        return_dtype=pl.Float64
                    )
                )
            )
            .with_row_index("row_id")
        )


class CuratePipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            dataset_name=DATASET_NAME.UCDAVIS19,
            description="Curate...",
            save_to=save_to,
        )

    def _drop_columns(self, df: pl.DataFrame, *args) -> Any:
        df = df.drop(
            "is_valid_handshake"
        )
        return (df, *args)

    def run(self, df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        schema = catalog.get_dataset_polars_schema(
            self.dataset_name,
            DATASET_TYPE.CURATE
        )

        columns = list(schema.keys())
        if "is_valid_handshake" not in columns:
            columns.append("is_valid_handshake")
        df = df.select(columns)

        df_pretrain = df.filter(pl.col("partition") == PARTITION_PRETRAINING)
        df_human = df.filter(pl.col("partition") == PARTITION_HUMAN)
        df_script = df.filter(pl.col("partition") == PARTITION_SCRIPT)
        res = dict()

        self.clear()
        self.name = "Curate pretrain partition..."
        self.extend((
            SequentialPipelineStage(
                self.compute_stats,
                "Compute stats",
            ),
            SequentialPipelineStage(
                self._drop_columns,
                "Remove columns",
            ),
            SequentialPipelineStage(
                self.compute_splits,
                "Compute splits",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self.write_parquet_files,
                    fname_prefix=self.dataset_name
                ),
                "Write parquet files",
            ),
        ))
        _, df_stats, df_splits = super().run(df_pretrain)
        res["df"] = df_pretrain
        res["df_stats"] = df_stats
        res["df_splits"] = df_splits

        for df_curr, suffix in zip(
            (df_human, df_script), ("human", "script")
        ):
            self.clear()
            self.name = f"Curate {suffix} partition..."
            self.extend((
                SequentialPipelineStage(
                    self.compute_stats,
                    "Compute stats",
                ),
                SequentialPipelineStage(
                    self._drop_columns,
                    "Remove columns",
                ),
                SequentialPipelineStage(
                    functools.partial(
                        self.write_parquet_files,
                        fname_prefix=f"{self.dataset_name}_{suffix}"
                    ),
                    "Write parquet files",
                ),
            ))
            _1, df_stats, _2 = super().run(df_curr)
            res[f"df_{suffix}"] = df_curr
            res[f"df_{suffix}_stats"] = df_stats
        
        return res


class UCDavis19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.UCDAVIS19)
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

    @property
    def _list_raw_txt_files(self):
        return list(self.folder_raw.rglob("*.txt"))

    def _parse_raw(self) -> pd.DataFrame:
        paths = self._list_raw_txt_files
        with (
            richutils.Progress(
                description="Parsing raw txt files...", 
                total=len(paths),
            ) as progress,
            multiprocessing.get_context("spawn").Pool(processes=2) as pool,
        ):
            schema = self.get_schema(DATASET_TYPE.RAW).to_polars()
            func = functools.partial(_parse_raw_txt_worker, schema=schema)
            data = []
            for df in pool.imap_unordered(func, paths):
                data.append(df)
                progress.update()
            return pl.concat(data)

    def raw(self, num_workers: int = 1) -> pl.DataFrame:
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

        self.df = (
            RawTXTParser(self.folder_raw, num_workers)
            .run(*self._list_raw_txt_files)
        )
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                self.df, 
                self.folder_raw / f"{self.name}.parquet", 
                echo=False
            )
        return self.df
  
    def _raw_postprocess(self) -> pl.DataFrame:
        self.load(DATASET_TYPE.RAW)
        df, *_ = RawPostprocessingPipeline(
            save_to=self.folder_raw
        ).run(self.df)
        return df
        
    def curate(self, recompute_intermediate: bool = False) -> pl.DataFrame:
        fname = self.folder_raw / f"_postprocess.parquet"
        if not fname.exists() or recompute_intermediate:
            df = self._raw_postprocess()

        with richutils.SpinnerProgress(
            description=f"Load {self.name}/raw postprocess..."
        ):
            df = fileutils.load_parquet(fname, echo=False)

        res = CuratePipeline(save_to=self.folder_curate).run(df)
        self.df = res["df"]
        self.df_stats = res["df_stats"]
        self.df_splits = res["df_splits"]
        self.df_human = res["df_human"]
        self.df_human_stats = res["df_human_stats"]
        self.df_script = res["df_script"]
        self.df_script_stats = res["df_script_stats"]
        return self.df

    def load(self, dataset_type: DATASET_TYPE, *args, **kwargs) -> Dataset:
        self.df = None
        self.df_stats = None
        self.df_splits = None
        self.df_human = None
        self.df_human_stats = None
        self.df_script = None
        self.df_script_stats = None

        super().load(dataset_type, *args, **kwargs)

        if dataset_type == DATASET_TYPE.CURATE:
            self.df_human = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_human.parquet",
                echo=False
            )
            self.df_human_stats = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_human_stats.parquet",
                echo=False
            )
            self.df_script = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_script.parquet",
                echo=False
            )
            self.df_script_stats = fileutils.load_if_exists(
                self.folder_curate / f"{self.name}_script_stats.parquet",
                echo=False
            )
        return self
