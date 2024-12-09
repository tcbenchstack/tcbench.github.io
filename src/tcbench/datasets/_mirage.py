from __future__ import annotations

from collections import deque, OrderedDict
from typing import Dict, Any, Tuple, Iterable

import pathlib
import json
import multiprocessing
import tempfile
import functools
import math
import os
import shutil

import polars as pl
import numpy as np

import tcbench
from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.datasets.core import (
    Dataset,
    SequentialPipelineStage,
    DatasetSchema,
    BaseDatasetProcessingPipeline,
)
from tcbench.datasets import (
    DATASET_NAME,
    DATASET_TYPE,
    curation,
    catalog
)
from tcbench.datasets._constants import (
    DATASETS_RESOURCES_FOLDER,
    APP_LABEL_BACKGROUND,
)


def _reformat_json_entry(
    json_entry: Dict[str, Any], 
    fields_order: List[str] = None,
) -> Dict[str, Any]:
    """Process a JSON nested structure by chaining partial names via "_" """
    data = OrderedDict()
    if fields_order is None:
        fields_order = []
    for field_name in fields_order:
        data[field_name] = None

    queue = deque(json_entry.items())
    while len(queue):
        key, value = queue.popleft()
        if not isinstance(value, dict):
            if (
                value is not None 
                and not isinstance(value, (list, str)) 
                and math.isnan(value)
            ):
                value = "null"
            data[key] = value
            continue
        for inner_key, inner_value in value.items():
            queue.append((f"{key}_{inner_key}", inner_value))
    return data


def _json_entry_to_dataframe(
    json_entry: Dict[str, Any], 
    dataset_schema: DatasetSchema
) -> pl.DataFrame:
    """Create a DataFrame by flattening the JSON nested structure
    chaining partial names via "_"
    """
    json_entry = _reformat_json_entry(json_entry, dataset_schema.fields)
    for key, value in json_entry.items():
        if value == "NaN":
            value = np.nan
        # Note: enforce values to be list for pl.DataFrame conversion
        json_entry[key] = [value]
    return pl.DataFrame(json_entry, schema=dataset_schema.to_polars())


def _parse_raw_json_worker(
    fname: pathlib.Path, 
    dataset_schema: DatasetSchema, 
    folder_json: pathlib.Path,
    folder_parquet: pathlib.Path,
) -> None:
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    new_fname_json = f"{fname.parent.name}__{fname.name}"
    with open(folder_json / new_fname_json, "w") as fout:
        for idx, (flow_id, json_entry) in enumerate(data.items()):
            # adding a few extra columns after parsing raw data
            json_entry = _reformat_json_entry(
                json_entry, 
                dataset_schema.fields
            )
            src_ip, src_port, dst_ip, dst_port, proto_id = \
                flow_id.split(",")
            json_entry["src_ip"] = src_ip
            json_entry["src_port"] = int(src_port)
            json_entry["dst_ip"] = dst_ip
            json_entry["dst_port"] = int(dst_port)
            json_entry["proto_id"] = int(proto_id)
            json_entry["fname"] = fname.stem
            json_entry["fname_row_idx"] = idx
            json_entry["parent_folder"] = fname.parent.name
            json.dump(json_entry, fout)
            fout.write("\n")

    (
        pl.read_ndjson(
            folder_json / new_fname_json,
            schema=dataset_schema.to_polars()
        )
        .write_parquet(
            (folder_parquet / new_fname_json).with_suffix(".parquet")
        )
    )

    os.unlink(str(folder_json / new_fname_json))


def load_raw_json(
    fname: pathlib.Path, 
    dataset_name: DATASET_NAME
) -> pl.DataFrame:
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    dataset_schema = catalog.get_dataset_schema(
        dataset_name, 
        DATASET_TYPE.RAW
    )

    l = []
    for idx, (flow_id, json_entry) in enumerate(data.items()):
        json_entry = _reformat_json_entry(json_entry, dataset_schema.fields)
        src_ip, src_port, dst_ip, dst_port, proto_id = \
            flow_id.split(",")
        json_entry["src_ip"] = src_ip
        json_entry["src_port"] = int(src_port)
        json_entry["dst_ip"] = dst_ip
        json_entry["dst_port"] = int(dst_port)
        json_entry["proto_id"] = int(proto_id)
        json_entry["fname"] = fname.stem
        json_entry["fname_row_idx"] = idx
        if dataset_name == DATASET_NAME.MIRAGE19:
            json_entry["parent_folder"] = fname.parent.name
        l.append(_json_entry_to_dataframe(json_entry, dataset_schema))
    return pl.concat(l)


def _rename_columns(columns: List[str]) -> Dict[str, str]:
    rename = dict()
    for col in columns:
        new_name = col.lower()
        if col.startswith("packet_data_"):
            new_name = (
                new_name.replace("packet_data", "pkts")
                .replace("packet_dir", "dir")
                .replace("ip_packet_bytes", "l3_size")
                .replace("ip_header_bytes", "l3_header_size")
                .replace("l4_payload_bytes", "l4_size")
                .replace("l4_header_bytes", "l4_header_size")
                .replace("l4_raw_payload", "raw_payload")
            )
        elif col.startswith("flow_metadata"):
            new_name = (
                new_name.replace("flow_metadata_", "")
                .replace("bf_", "")
                .replace("num_packets", "packets")
                .replace("ip_packet_bytes", "bytes")
                .replace("l4_payload_bytes", "bytes_payload")
            )
            if "uf" in new_name:
                new_name = new_name.replace("uf_", "") + "_upload"
            elif "df" in new_name:
                new_name = new_name.replace("df_", "") + "_download"
        elif col.startswith("flow_features_"): 
            new_name = (
                new_name
                .replace("flow_features_", "")
                .replace("packet_length", "packet_size")
                .replace("_biflow", "")
            ) 
            if new_name.endswith("percentile"):
                _1, q, _2 = new_name.rsplit("_", 2)
                new_name = new_name.replace(f"{q}_percentile", f"q{q}")
            if "upstream_flow" in new_name:
                new_name = new_name.replace("_upstream_flow", "") + "_upload"
            elif "downstream_flow" in new_name:
                new_name = new_name.replace("_downstream_flow", "") + "_download"
        rename[col] = new_name
    return rename


class ParserRawJSON:
    def __init__(self, name: DATASET_NAME, num_workers: int = -1):
        self.name = name
        self.num_workers = tcbench.validate_num_workers(num_workers)
        self.dataset_schema = catalog.get_dataset_schema(
            name, 
            DATASET_TYPE.RAW
        )

    def _parse_raw_json(
        self, 
        files: Iterable[pathlib.Path],
        save_to: pathlib.Path,
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        if sort_by is None:
            sort_by = (
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            )

        save_to = pathlib.Path(save_to)
        if (save_to / "__tmp__").exists():
            shutil.rmtree(save_to / "__tmp__", ignore_errors=True)
        (save_to / "__tmp__").mkdir(parents=True)

        with tempfile.TemporaryDirectory(
            dir=save_to/"__tmp__"
        ) as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            folder_json = tmp_folder / "json"
            folder_parquet = tmp_folder / "parquet"
            folder_json.mkdir(parents=True)
            folder_parquet.mkdir(parents=True)
            func = functools.partial(
                _parse_raw_json_worker, 
                dataset_schema=self.dataset_schema, 
                folder_json=folder_json,
                folder_parquet=folder_parquet,
            )
            with (
                richutils.Progress(
                    description="Parse JSON files...", 
                    total=len(files)
                ) as progress,
                (
                    multiprocessing
                    .get_context("spawn")
                    .Pool(processes=self.num_workers)
                ) as pool,
            ):
                for _ in pool.imap_unordered(func, files):
                    progress.update()
            
            with richutils.SpinnerProgress(
                description="Write parquet files...",
            ):
                # Note: the two steps with intermediate file are
                # clearly less efficient than doing all operations
                # in one go, but turned out to be the only way
                # to process mirage24 on Intel MacBook with 16GB of memory.
                # and without using the lazy loading
                # the code was crashing on a Linux server with 64GB
                # (even if the loaded DataFrame was 40GB only)
                (
                    pl.scan_parquet(folder_parquet)
                    .sort(sort_by)
                    .sink_parquet(
                        tmp_folder / f"{self.name}.parquet"
                    )
                )

        shutil.rmtree(save_to / "__tmp__", ignore_errors=True)
        return df

    def run(
        self, 
        files: Iterable[pathlib.Path], 
        save_to: pathlib.Path,
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        return self._parse_raw_json(files, save_to=save_to, sort_by=sort_by)


class BaseRawPostprocessingPipeline(BaseDatasetProcessingPipeline):
    def __init__(
        self, 
        df_app_metadata: pl.DataFrame,
        dataset_name: DATASET_NAME,
        save_to: pathlib.Path, 
    ):
        super().__init__(
            description="Postprocess raw...",
            dataset_name=dataset_name,
            save_to=save_to,
            progress=True
        )
        self.df_app_metadata = df_app_metadata

        stages = [
            SequentialPipelineStage(
                self.rename_columns,
                name="Rename columns",
            ),
            SequentialPipelineStage(
                self.add_other_columns,
                name="Add columns", 
            ),
            SequentialPipelineStage(
                self.add_android_package_name,
                name="Add Android package name",
            ),
            SequentialPipelineStage(
                self.add_app_and_background,
                name="Add metadata",
            ),
            SequentialPipelineStage(
                self.compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self.write_parquet_files,
                    fname_prefix="_postprocess"
                ),
                name="Write parquet files",
            ),
        ]

        self.clear()
        self.extend(stages)

    def rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(_rename_columns(df.columns))

    def add_android_package_name(self, df: pl.DataFrame) -> pl.DataFrame:
        # Note: the syntax of the labeling is inconsistent between
        # datasets. For MIRAGE19, MIRAGE20 and MIRAGE22, flow_metadata_BF_label
        # represents the android package name, but for MIRAGE24 is a mix
        # between the app name and the android package name. At the same
        # time, for MIRAGE20, MIRAGE22 and MIRAGE24 flow_metadata_BF_sublabel
        # is the android package name (with some extra :<suffix>) but
        # this column is missing for MIRAGE19
        if self.dataset_name == DATASET_NAME.MIRAGE19:
            return df.with_columns(
                android_package_name=pl.col("label")
            )
        return df.with_columns(
            android_package_name=(
                pl.col("sublabel")
                .str
                .to_lowercase()
                .str
                .split(":")
                .list
                .first()
            )
        )

    def add_app_and_background(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            # add app column using static metadata
            .join(
                self.df_app_metadata,
                left_on="android_package_name",
                right_on="android_package_name",
                how="left",
            )
            .with_columns(
                # flows without a recognized label are re-labeled as background
                app=(pl.col("app").fill_null(APP_LABEL_BACKGROUND))
            )
            .with_columns(
                # force to background flows with UDP packets of size zero
                app=(
                    pl.when(
                        (pl.col("proto_id") == 17).and_(
                            pl.col("pkts_size").list.min() == 0
                        )
                    )
                    .then(pl.lit(APP_LABEL_BACKGROUND))
                    .otherwise(pl.col("app"))
                )
            )
        )
        return df

    def add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # add column: convert proto_id to string (6->tcp, 17->udp)
        df = df.with_columns(
            proto=(
                pl.when(pl.col("proto_id").eq(6))
                .then(pl.lit("tcp"))
                .otherwise(pl.lit("udp"))
            ),
        )
        # add columns: ip addresses private/public
        df = curation.add_ip_column_flags(df)
        # add columns: check if tcp handshake is valid
        df = curation.add_is_valid_tcp_handshake_heuristic(
            df, tcp_handshake_size=0, direction_upload=0, direction_download=1
        )
        return (
            df
            # add a global row_id
            .with_row_index(name="row_id")
        )


class BaseCuratePipeline(BaseDatasetProcessingPipeline):
    def __init__(
        self, 
        dataset_name: DATASET_NAME,
        save_to: pathlib.Path,
    ):
        super().__init__(
            description="Curation...",
            dataset_name=dataset_name,
            save_to=save_to,
            dataset_schema=(
                catalog.get_dataset_schema(
                    dataset_name, 
                    DATASET_TYPE.CURATE
                )
            ), 
            progress=True
        )

        stages = [
            SequentialPipelineStage(
                self.drop_background, 
                name="Drop background flows"
            ),
            SequentialPipelineStage(
                self.drop_dns,
                name="Drop DNS traffic",
            ),
            SequentialPipelineStage(
                self.drop_invalid_ips,
                name="Drop flows with invalid IPs",
            ),
            SequentialPipelineStage(
                self.adjust_packet_series,
                name="Adjust packet series",
            ),
            SequentialPipelineStage(
                self.add_pkt_indices_columns,
                name="Add packet series indices"
            ),
            SequentialPipelineStage(
                self.add_more_columns,
                name="Add more columns",
            ),
            SequentialPipelineStage(
                self.drop_columns,
                name="Drop columns",
            ),
            SequentialPipelineStage(
                self.final_filter,
                name="Filter out flows",
            ),
            SequentialPipelineStage(
                self.compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self.compute_splits,
                    num_splits=10,
                    test_size=0.1,
                    y_colname="app",
                    index_colname="row_id",
                    seed=1,
                ),
                name="Compute splits",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self.write_parquet_files,
                    fname_prefix=str(self.dataset_name),
                    columns=self.dataset_schema.fields,
                ),
                name="Write parquet files",
            ),
        ]

        self.clear()
        self.extend(stages)

    def drop_background(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("app") != APP_LABEL_BACKGROUND)

    def drop_dns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            ~curation.expr_is_dns_heuristic()
        )

    def drop_invalid_ips(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            curation.expr_are_ips_valid()
        )

    def adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            # increase packets size to reflect the expected true size
            # for TCP, add 40 bytes
            # for UDP, add 28 bytes
            pkts_size=(
                pl.when(pl.col("proto") == "tcp")
                .then(pl.col("pkts_size").list.eval(pl.element() + 40))
                .otherwise(pl.col("pkts_size").list.eval(pl.element() + 28))
            ),
            # enforce direction (0/upload: 1, 1/download: -1)
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0).then(1).otherwise(-1)
                )
            ),
        )
        return df

    def add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            # series with the index of TCP acks packets
            pkts_ack_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(curation.expr_pkts_ack_idx(ack_size=40))
                # for UDP, packets are always larger then 0 bytes
                # so the following is selecting all indices
                .otherwise(curation.expr_pkts_ack_idx(ack_size=0))
            ),
            # series with the index of data packets
            pkts_data_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(curation.expr_pkts_data_idx(ack_size=40))
                # for UDP, packets are always larger then 0 bytes
                # so the following is selecting all indices
                .otherwise(curation.expr_pkts_data_idx(ack_size=0))
            ),
        )
        return df

    def add_more_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
                # length of all series
                pkts_len=(pl.col("pkts_size").list.len()),
                # flag to indicate if the packet sizes have all packets
                pkts_is_complete=(pl.col("pkts_size").list.len() == pl.col("packets")),
                # series pkts_size * pkts_dir
                pkts_size_times_dir=(curation.expr_pkts_size_times_dir()),
        )
        df = df.with_columns(
                # number of ack packets
                packets_ack=(pl.col("pkts_ack_idx").list.len()),
                # number of ack packets in upload
                packets_ack_upload=(
                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_ack_idx")
                ),
                # number of ack packets in download
                packets_ack_download=(
                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_ack_idx")
                ),
                # number of data packets
                packets_data=(pl.col("pkts_data_idx").list.len()),
                # number of ack packets in upload
                packets_data_upload=(
                    curation.expr_list_len_upload("pkts_size_times_dir", "pkts_data_idx")
                ),
                # number of ack packets in download
                packets_data_download=(
                    curation.expr_list_len_download("pkts_size_times_dir", "pkts_data_idx")
                ),
            )
        return df

    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            [
                "pkts_src_port",
                "pkts_dst_port",
                "pkts_raw_payload",
                "labeling_type",
                "proto_id",
            ]
        )

    def final_filter(
        self, 
        df: pl.DataFrame, 
        min_pkts: int = None
    ) -> pl.DataFrame:
        df_new = df.filter(
            # flows starting with a complete handshake
            pl.col("is_valid_handshake")
        )

        if min_pkts is not None:
            # flows with at least a specified number of packets
            df_new = df.filter(pl.col("packets") >= min_pkts)
        return df_new


class ExtendedRawPostprocessingPipeline(BaseRawPostprocessingPipeline):
    def __init__(
        self,
        dataset_name: DATASET_NAME,
        df_app_metadata: pl.DataFrame,
        save_to: pathlib.Path
    ):
        super().__init__(
            df_app_metadata=df_app_metadata,
            dataset_name=dataset_name,
            save_to=save_to
        )

        self.insert(
            1, 
            SequentialPipelineStage(
                self.drop_columns,
                name="Drop columns",
            ),
        )

    def rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "device": "device_id",
                "pkts_l3_size": "pkts_size",
            })
        )
        return df

    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_is_clear",
            "pkts_heuristic",
            "pkts_annotations",
            "label_source",
            "label_version_code",
            "label_version_name",
            "labeling_type",
            "pkts_l4_header_size",
            "pkts_l3_header_size",
            "pkts_raw_payload",
            "pkts_src_port",
            "pkts_dst_port",
        )

    def add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # add column: convert proto_id to string (6->tcp, 17->udp)
        df = df.with_columns(
            proto=(
                pl.when(pl.col("proto_id").eq(6))
                .then(pl.lit("tcp"))
                .otherwise(pl.lit("udp"))
            ),
        )

        # add columns: ip addresses private/public
        df = curation.add_ip_column_flags(df)
        # add columns: check if tcp handshake is valid
        df = curation.add_is_valid_tcp_handshake_from_flags(
            df, 
            "pkts_tcp_flags",
            "pkts_dir",
            "proto", 
            proto_udp="udp",
            direction_upload=0,
            direction_download=1, 
        )

        return (
            df
            # add a global row_id
            .with_row_index(name="row_id")
        )


class BaseMirageDataset(Dataset):
    def __init__(
        self,
        name: DATASET_NAME,
        subfolder_raw_json: pathlib.Path,
        raw_sort_by: Iterable[str] | None = None,
    ):
        super().__init__(name=name)
        self.df_app_metadata = (
            pl.read_csv(
                DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
            )
            .with_columns(
                pl.col("android_package_name")
                .str
                .to_lowercase()
            )
        )
        self._subfolder_raw_json = self.folder_raw / subfolder_raw_json
        if raw_sort_by is None:
            raw_sort_by = [
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            ]
        self._raw_sort_by = raw_sort_by

    @property
    def _list_raw_json_files(self):
        return list(self._subfolder_raw_json.rglob("*.json"))

    def raw(self, num_workers: int = -1) -> pl.DataFrame:
        return (
            ParserRawJSON(
                self.name,
                num_workers,
            )
            .run(
                self._list_raw_json_files,
                sort_by=self._raw_sort_by,
                save_to=self.folder_raw,
            )
        )

    def _raw_postprocess(self, num_workers: int = -1) -> Tuple[pl.DataFrame]:
        self.load(DATASET_TYPE.RAW, lazy=False)
        return (
            _factory_raw_postprocessing_pipeline(self)
            .run(self.df)
        )

    def curate(
        self, 
        recompute_intermediate: bool = False,
    ) -> pl.LazyFrame:
        fname = self.folder_raw / "_postprocess.parquet"
        if not fname.exists() or recompute_intermediate:
            self._raw_postprocess()

        df = self.df
        with richutils.SpinnerProgress(
            description=f"Load {self.name}/raw postprocess..."
        ):
            df = fileutils.load_parquet(fname, echo=False, lazy=False)

        self.df, self.df_stats, self.df_splits = (
            _factory_curate_pipeline(self)
            .run(df)
        )
        return df


class Mirage19RawPostprocessingPipeline(BaseRawPostprocessingPipeline):
    def __init__(
        self,
        df_app_metadata: pl.DataFrame,
        save_to: pathlib.Path,
    ):
        super().__init__(
            df_app_metadata,
            dataset_name=DATASET_NAME.MIRAGE19,
            save_to=save_to,
        )

    def rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "parent_folder": "device_id",
                "pkts_l4_size": "pkts_size",
            })
        )
        return df


class Mirage19(BaseMirageDataset):
    def __init__(self):
        super().__init__(
            name=DATASET_NAME.MIRAGE19,
            subfolder_raw_json=(
                pathlib.Path("MIRAGE-2019_traffic_dataset_downloadable")
            ),
            raw_sort_by=[
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            ],
        )


class Mirage22CuratePipeline(BaseCuratePipeline):
    def __init__(
        self,
        save_to: pathlib.Path,
        dataset_name: DATASET_NAME = DATASET_NAME.MIRAGE22
    ):
        super().__init__(
            dataset_name=dataset_name, 
            save_to=save_to,
        )

    def adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            # enforce direction (0/upload: 1, 1/download: -1)
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0).then(1).otherwise(-1)
                )
            )
        )
        return df

    def add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            # series with the index of TCP acks packets
            pkts_ack_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(
                    curation.expr_indices_list_value_equal_to(
                        "pkts_l4_size", 
                        value=0
                    )
                )
                # for UDP, there are not ACK
                .otherwise(
                    curation.expr_indices_list_value_lower_than(
                        "pkts_l4_size", 
                        value=0, 
                        inclusive=False
                    )
                )
            ),
            # series with the index of data packets
            pkts_data_idx=(
                pl.when(pl.col("proto") == "tcp")
                # for TCP, acks are enforced to 40 bytes
                .then(
                    curation.expr_indices_list_value_not_equal_to(
                        "pkts_l4_size", 
                        value=0
                    )
                )
                # for UDP, all packets are data packets
                .otherwise(
                    curation.expr_indices_list_value_greater_than(
                        "pkts_l4_size", 
                        value=0, 
                        inclusive=True
                    )
                )
            )
        )
        return df

    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_l4_size",
            "proto_id",
        )


class Mirage22(BaseMirageDataset):
    def __init__(self):
        super().__init__(
            name=DATASET_NAME.MIRAGE22,
            subfolder_raw_json=pathlib.Path("MIRAGE-COVID-CCMA-2022/Raw_JSON"),
            raw_sort_by=[
                "flow_metadata_BF_device",
                "fname", 
                "fname_row_idx"
            ],
        )


class Mirage20RawPostprocessingPipeline(ExtendedRawPostprocessingPipeline):
    def __init__(
        self,
        df_app_metadata: pl.DataFrame,
        save_to: pathlib.Path
    ):
        super().__init__(
            dataset_name=DATASET_NAME.MIRAGE20,
            df_app_metadata=df_app_metadata,
            save_to=save_to,
        )

    def drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "label_source",
            "label_version_code",
            "label_version_name",
            "labeling_type",
            "pkts_l4_header_size",
            "pkts_l3_header_size",
            "pkts_raw_payload",
            "pkts_src_port",
            "pkts_dst_port",
        )


class Mirage20(BaseMirageDataset):
    def __init__(self):
        super().__init__(
            name=DATASET_NAME.MIRAGE20,
            subfolder_raw_json=pathlib.Path("MIRAGE2020_video_public"),
            raw_sort_by=[
                "flow_metadata_BF_device",
                "fname", 
                "fname_row_idx"
            ],
        )


class Mirage24(BaseMirageDataset):
    def __init__(self):
        super().__init__(
            name=DATASET_NAME.MIRAGE24,
            subfolder_raw_json=pathlib.Path("MIRAGE-AppAct-2024"),
            raw_sort_by=[
                "fname", 
                "fname_row_idx"
            ],
        )


def _factory_raw_postprocessing_pipeline(
    dataset: BaseMirageDataset
) -> BaseDatasetProcessingPipeline:
    kwargs = dict(
        df_app_metadata=dataset.df_app_metadata, 
        save_to=dataset.folder_raw
    )
    if dataset.name == DATASET_NAME.MIRAGE19:
        cls = Mirage19RawPostprocessingPipeline
    elif dataset.name == DATASET_NAME.MIRAGE20:
        cls = Mirage20RawPostprocessingPipeline
    elif dataset.name in (
        DATASET_NAME.MIRAGE22,
        DATASET_NAME.MIRAGE24
    ):
        cls = ExtendedRawPostprocessingPipeline
        kwargs["dataset_name"] = dataset.name
    return cls(**kwargs)


def _factory_curate_pipeline(
    dataset: BaseMirageDataset
) -> BaseDatasetProcessingPipeline:
    kwargs = dict(
        dataset_name=dataset.name,
        save_to=dataset.folder_curate,
    )
    if dataset.name == DATASET_NAME.MIRAGE19:
        cls = BaseCuratePipeline
    elif dataset.name in (
        DATASET_NAME.MIRAGE20, 
        DATASET_NAME.MIRAGE22,
        DATASET_NAME.MIRAGE24,
    ):
        cls = Mirage22CuratePipeline
    return cls(**kwargs)
