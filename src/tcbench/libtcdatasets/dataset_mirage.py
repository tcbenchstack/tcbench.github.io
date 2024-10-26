from __future__ import annotations

from collections import deque, OrderedDict
from typing import Dict, Any

import pathlib
import json
import multiprocessing
import tempfile
import functools
import math
import ipaddress

from multiprocessing import get_context

import polars as pl
import numpy as np

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.libtcdatasets import curation
from tcbench.libtcdatasets.core import (
    Dataset,
    SequentialPipelineStage,
#    SequentialPipeline,
    DatasetSchema,
    BaseDatasetProcessingPipeline,
)
from tcbench.libtcdatasets.constants import (
    DATASET_NAME,
    DATASET_TYPE,
    DATASETS_RESOURCES_FOLDER,
    APP_LABEL_BACKGROUND,
    APP_LABEL_ALL,
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
    dset_schema: DatasetSchema
) -> pl.DataFrame:
    """Create a DataFrame by flattening the JSON nested structure
    chaining partial names via "_"
    """
    json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
    for key, value in json_entry.items():
        if value == "NaN":
            value = np.nan
        # Note: enforce values to be list for pl.DataFrame conversion
        json_entry[key] = [value]
    return pl.DataFrame(json_entry, schema=dset_schema.to_polars())


def _load_raw_json_worker(
    fname: pathlib.Path, 
    dset_schema: DatasetSchema, 
    save_to: pathlib.Path,
) -> None:
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    with open(save_to / f"{fname.parent.name}__{fname.name}", "w") as fout:
        for idx, (flow_id, json_entry) in enumerate(data.items()):
            # adding a few extra columns after parsing raw data
            json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
            src_ip, src_port, dst_ip, dst_port, proto_id = \
                flow_id.split(",")
            json_entry["src_ip"] = src_ip
            json_entry["src_port"] = int(src_port)
            json_entry["dst_ip"] = dst_ip
            json_entry["dst_port"] = int(dst_port)
            json_entry["proto_id"] = int(proto_id)
            json_entry["fname"] = fname.stem
            json_entry["fname_row_idx"] = idx
            if dset_schema.dataset_name == DATASET_NAME.MIRAGE19:
                json_entry["parent_folder"] = fname.parent.name
            json.dump(json_entry, fout)
            fout.write("\n")


def load_raw_json(fname: pathlib.Path, dataset_name: DATASET_NAME) -> pl.DataFrame:
    import tcbench
    fname = pathlib.Path(fname)
    with open(fname) as fin:
        data = json.load(fin)

    dset_schema = (
        tcbench
        .datasets_catalog()
        [dataset_name]
        .get_schema(DATASET_TYPE.RAW)
    )

    l = []
    for idx, (flow_id, json_entry) in enumerate(data.items()):
        json_entry = _reformat_json_entry(json_entry, dset_schema.fields)
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
        l.append(_json_entry_to_dataframe(json_entry, dset_schema))
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


class BaseParserRawJSON:
    def __init__(self, name: DATASET_NAME, dset_schema: DatasetSchema):
        self.name = name
        self.dset_schema = dset_schema

    def _parse_raw_json(
        self, 
        *files: Iterable[pathlib.Path],
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        if sort_by is None:
            sort_by = (
                "parent_folder", 
                "fname", 
                "fname_row_idx"
            )

        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            func = functools.partial(
                _load_raw_json_worker, 
                dset_schema=self.dset_schema, 
                save_to=tmp_folder,
            )
            with (
                richutils.Progress(
                    description="Parse JSON files...", 
                    total=len(files)
                ) as progress,
                multiprocessing.Pool(processes=2) as pool,
            ):
                for _ in pool.imap_unordered(func, files):
                    progress.update()
            
            with richutils.SpinnerProgress(description="Reload..."):
                df = (
                    pl.read_ndjson(
                        tmp_folder, 
                        schema=self.dset_schema.to_polars()
                    )
                    .sort(*sort_by)
                )
        return df

    def run(
        self, 
        *files: Iterable[pathlib.Path], 
        save_to: pathlib.Path = None,
        sort_by: Iterable[str] = None,
    ) -> pl.DataFrame:
        df = self._parse_raw_json(*files, sort_by=sort_by)
        if save_to is None:
            save_to = pathlib.Path(".")
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                df, 
                save_as=save_to/f"{self.name}.parquet", 
                echo=False
            )
        return df


class MirageBaseRawPostprocessingPipeline(BaseDatasetProcessingPipeline):
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
                self._rename_columns,
                name="Rename columns",
            ),
            SequentialPipelineStage(
                self._add_other_columns,
                name="Add columns", 
            ),
            SequentialPipelineStage(
                self._add_app_and_background,
                name="Add metadata",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._write_parquet_files,
                    fname_prefix="_postprocess"
                ),
                name="Write parquet files",
            ),
        ]

        self.clear()
        self.extend(stages)

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename(_rename_columns(df.columns))

    def _add_app_and_background(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            # add app column using static metadata
            .join(
                self.df_app_metadata,
                left_on="label",
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

    def _add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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

#    def _compute_stats(self, df):
#        df_stats = curation.get_stats(df)
#        return (df, df_stats)
#
#    def _write_parquet_files(self, df, df_stats):
#        df.write_parquet(
#            self.save_to / "_postprocess.parquet"
#        )
#        df_stats.write_parquet(
#            self.save_to / f"_postprocess_stats.parquet"
#        )
#        return df, df_stats


class MirageBaseCuratePipeline(BaseDatasetProcessingPipeline):
    def __init__(
        self, 
        dataset_name: DATASET_NAME,
        save_to: pathlib.Path,
        dset_schema: DatasetSchema,
    ):
        super().__init__(
            description="Curation...",
            dataset_name=dataset_name,
            save_to=save_to,
            dset_schema=dset_schema,
            progress=True
        )

        stages = [
            SequentialPipelineStage(
                self._rename_columns,
                name="Rename columns",
            ),
            SequentialPipelineStage(
                self._drop_background, 
                name="Drop background flows"
            ),
            SequentialPipelineStage(
                self._drop_dns,
                name="Drop DNS traffic",
            ),
            SequentialPipelineStage(
                self._drop_invalid_ips,
                name="Drop flows with invalid IPs",
            ),
            SequentialPipelineStage(
                self._adjust_packet_series,
                name="Adjust packet series",
            ),
            SequentialPipelineStage(
                self._add_pkt_indices_columns,
                name="Add packet series indices"
            ),
            SequentialPipelineStage(
                self._add_more_columns,
                name="Add more columns",
            ),
            SequentialPipelineStage(
                self._drop_columns,
                name="Drop columns",
            ),
            SequentialPipelineStage(
                self._final_filter,
                name="Filter out flows",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._compute_splits,
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
                    self._write_parquet_files,
                    fname_prefix=str(self.dataset_name),
                    columns=self.dset_schema.fields,
                ),
                name="Write parquet files",
            ),
        ]

        self.clear()
        self.extend(stages)

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "label": "android_package_name",
        })

    def _drop_background(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("app") != APP_LABEL_BACKGROUND)

    def _drop_dns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            ~curation.expr_is_dns_heuristic()
        )

    def _drop_invalid_ips(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            curation.expr_are_ips_valid()
        )

    def _adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _add_more_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            [
                "pkts_src_port",
                "pkts_dst_port",
                "pkts_raw_payload",
                "labeling_type",
                "proto_id",
            ]
        )

    def _final_filter(self, df: pl.DataFrame, min_pkts: int = None) -> pl.DataFrame:
        df_new = df.filter(
            # flows starting with a complete handshake
            pl.col("is_valid_handshake")
        )

        if min_pkts is not None:
            # flows with at least a specified number of packets
            df_new = df.filter(pl.col("packets") >= min_pkts)
        return df_new


class Mirage19RawPostprocessingPipeline(MirageBaseRawPostprocessingPipeline):
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

        self.replace_stage(
            "Rename columns",
            SequentialPipelineStage(
                self._rename_columns,
                "Rename columns",
            )
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "parent_folder": "device_id",
                "pkts_l4_size": "pkts_size",
            })
        )
        return df


class Mirage19CuratePipeline(MirageBaseCuratePipeline):
    def __init__(
        self,
        save_to: pathlib.Path,
    ):
        import tcbench
        dset_schema = (
            tcbench
            .datasets_catalog()
            [DATASET_NAME.MIRAGE19]
            .get_schema(DATASET_TYPE.CURATE)
        )
        super().__init__(
            dataset_name=DATASET_NAME.MIRAGE19,
            save_to=save_to,
            dset_schema=dset_schema
        )

        self.replace_stage(
            "Rename columns",
            SequentialPipelineStage(
                self._rename,
                "Rename columns",
            )
        )

    def _rename(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "label": "android_package_name",
        })


class Mirage19(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.MIRAGE19)
        self.df_app_metadata = pl.read_csv(
            DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json_files(self):
        return list(self.folder_raw.rglob("*.json"))

    def raw(self) -> pl.DataFrame:
        return (
            BaseParserRawJSON(
                self.name, 
                self.get_schema(DATASET_TYPE.RAW)
            )
            .run(
                *self._list_raw_json_files,
                sort_by=(
                    "parent_folder", 
                    "fname", 
                    "fname_row_idx"
                ),
                save_to=self.folder_raw,
            )
        )

    def _raw_postprocess(self) -> Tuple[pl.DataFrame]:
        self.load(DATASET_TYPE.RAW)
        return (
            Mirage19RawPostprocessingPipeline(
                self.df_app_metadata,
                save_to=self.folder_raw,
            )
            .run(self.df)
        )

    def curate(self, recompute_intermediate: bool = False) -> pl.DataFrame:
        fname = self.folder_raw / "_postprocess.parquet"
        if fname.exists() and not recompute_intermediate:
            with richutils.SpinnerProgress(
                description=f"Load {self.name}/raw postprocess..."
            ):
                df = fileutils.load_parquet(fname, echo=False)
        else:
            df, *_ = self._raw_postprocess()

        self.df, self.df_stats, self.df_splits = (
            Mirage19CuratePipeline(
                self.folder_curate
            )
            .run(df)
        )
        return df
        

class Mirage22RawPostprocessingPipeline(MirageBaseRawPostprocessingPipeline):
    def __init__(
        self,
        df_app_metadata: pl.DataFrame,
        save_to: pathlib.Path
    ):
        super().__init__(
            df_app_metadata,
            dataset_name=DATASET_NAME.MIRAGE22,
            save_to=save_to
        )

        self.replace_stage(
            "Rename columns",
            SequentialPipelineStage(
                self._rename_columns,
                name="Rename columns",
            ),
        )
        self.replace_stage(
            "Add columns", 
            SequentialPipelineStage(
                self._add_other_columns,
                name="Add columns", 
            ),
        )
        self.insert(
            1, 
            SequentialPipelineStage(
                self._drop_columns,
                name="Drop columns",
            ),
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df
            .rename(_rename_columns(df.columns))
            .rename({
                "device": "device_id",
                "pkts_l3_size": "pkts_size",
                #"label": "android_package_name",
            })
        )
        return df

    def _drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_is_clear",
            "pkts_heuristic",
            "pkts_annotations",
            "label_source",
            "sublabel",
            "label_version_code",
            "label_version_name",
            "labeling_type",
            "pkts_l4_header_size",
            "pkts_l3_header_size",
            "pkts_raw_payload",
            "pkts_src_port",
            "pkts_dst_port",
        )

    def _add_other_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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


class Mirage22CuratePipeline(MirageBaseCuratePipeline):
    def __init__(
        self,
        save_to: pathlib.Path
    ):
        import tcbench
        dset_schema = (
            tcbench.datasets_catalog()
            [DATASET_NAME.MIRAGE22]
            .get_schema(DATASET_TYPE.CURATE)
        )
        super().__init__(
            dataset_name=DATASET_NAME.MIRAGE22,
            save_to=save_to,
            dset_schema=dset_schema,
        )

        self.replace_stage(
            "Adjust packet series",
            SequentialPipelineStage(
                self._adjust_packet_series,
                name="Adjust packet series",
            ),
        )
        self.replace_stage(
            "Add packet series indices",
            SequentialPipelineStage(
                self._add_pkt_indices_columns,
                name="Add packet series indices"
            ),
        )
        self.replace_stage(
            "Drop columns",
            SequentialPipelineStage(
                self._drop_columns,
                name="Drop columns",
            )
        )

    def _adjust_packet_series(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            # enforce direction (0/upload: 1, 1/download: -1)
            pkts_dir=(
                pl.col("pkts_dir").list.eval(
                    pl.when(pl.element() == 0).then(1).otherwise(-1)
                )
            )
        )
        return df

    def _add_pkt_indices_columns(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def _drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "pkts_l4_size",
            "proto_id",
        )


class Mirage22(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.MIRAGE22)
        self.df_app_metadata = pl.read_csv(
           DATASETS_RESOURCES_FOLDER / f"{self.name}_app_metadata.csv"
        )

    @property
    def _list_raw_json_files(self) -> List[pathlib.Path]:
        return list(
            (
                self.folder_raw 
                / "MIRAGE-COVID-CCMA-2022" 
                / "Raw_JSON"
            ).rglob("*.json")
        )

    def raw(self) -> pl.DataFrame:
        parser = BaseParserRawJSON(
            self.name, 
            self.get_schema(DATASET_TYPE.RAW)
        )
        return parser.run(
            *self._list_raw_json_files,
            sort_by=(
                "flow_metadata_BF_device",
                "fname", 
                "fname_row_idx"
            ),
            save_to=self.folder_raw,
        )

    def _raw_postprocess(self) -> Tuple[pl.DataFrame]:
        _ = self.load(DATASET_TYPE.RAW)
        df = self.df
        return (
            Mirage22RawPostprocessingPipeline(
                self.df_app_metadata,
                save_to=self.folder_raw
            )
            .run(df)
        )

    def curate(self, recompute_intermediate: bool = True) -> pl.DataFrame:
        fname = self.folder_raw / "_postprocess.parquet"
        if recompute_intermediate or not fname.exists():
            self._raw_postprocess()

        with richutils.SpinnerProgress(
            description=f"Load {self.name}/raw postprocess..."
        ):
            df = fileutils.load_parquet(fname, echo=False)

        self.df, self.df_stats, self.df_splits = (
            Mirage22CuratePipeline(
                self.folder_curate
            )
            .run(df)
        )
        return self.df
