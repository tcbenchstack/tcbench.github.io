from __future__ import annotations

from typing import Tuple
from collections import OrderedDict

import polars as pl
import numpy as np

import pathlib
import multiprocessing
import functools
import shutil
import tempfile
import csv

from tcbench import fileutils
from tcbench.cli import richutils
from tcbench.libtcdatasets.core import (
    Dataset,
    DatasetSchema,
    BaseDatasetProcessingPipeline,
    SequentialPipelineStage,
    RawDatasetInstaller,
    _remove_fields_from_schema,
)
from tcbench.libtcdatasets.constants import DATASET_NAME, DATASET_TYPE
from tcbench.libtcdatasets import curation


def _fix_raw_csv(path: pathlib.Path) -> Iterable[Dict[str, str]]:
    with open(path) as fin:
        lines = []
        for row in csv.DictReader(fin):
            if "," not in row["ip.hdr_len"]:
                lines.append(row)
                continue
            row1 = row.copy()
            row2 = row.copy()
            for field in (
                "ip.hdr_len",
                "ip.dsfield.ecn",
                "ip.len",
                "ip.id",
                "ip.frag_offset",
                "ip.ttl",
                "ip.proto",
                "ip.checksum",
                "ip.src",
                "ip.dst",
                "tcp.hdr_len",
                "tcp.len",
                "tcp.srcport",
                "tcp.dstport",
                "tcp.seq",
                "tcp.ack",
                "tcp.flags.ns",
                "tcp.flags.fin",
                "tcp.window_size_value",
                "tcp.checksum",
                "tcp.urgent_pointer",
                "tcp.option_kind",
                "tcp.option_len",
                "tcp.options.timestamp.tsval",
                "tcp.options.timestamp.tsecr",
                "udp.srcport",
                "udp.dstport",
                "udp.length",
                "udp.checksum",
                "gquic.puflags.rsv",
                "gquic.packet_number",
            ):
                value = row[field]
                parts = value.split(",")
                if len(parts) == 1:
                    continue
                row1[field] = parts[0]
                row2[field] = parts[1]
            lines.append(row1)
            lines.append(row2)

        # wiping out L4 information for ICMP records
        for row in lines:
            if row["ip.proto"] != 1:
                continue
            for field in (
                "tcp.hdr_len"
                "tcp.len"
                "tcp.srcport"
                "tcp.dstport"
                "tcp.seq"
                "tcp.ack"
                "tcp.flags.ns"
                "tcp.flags.fin"
                "tcp.window_size_value"
                "tcp.checksum"
                "tcp.urgent_pointer"
                "tcp.option_kind"
                "tcp.option_len"
                "tcp.options.timestamp.tsval"
                "tcp.options.timestamp.tsecr"
                "udp.srcport"
                "udp.dstport"
                "udp.length"
                "udp.checksum"
                "gquic.puflags.rsv"
                "gquic.packet_number"
            ):
                row[field] = ""
    return lines
    

def _parse_raw_csv_worker(
    path: pathlib.Path, 
    schema: pl.Schema,
    save_to: pathlib.Path = None
) -> pl.DataFrame:
    path = pathlib.Path(path)
    schema = _remove_fields_from_schema(
        schema,
        # the csv does not have these two fields   
        "fname",
        "folder"
    )
    raw_schema = OrderedDict()
    for field_name, field_type in schema.items():
        # the first column of the CSV does not have a name
        if field_name == "index":
            field_name = ""
        raw_schema[field_name] = field_type

    rows = _fix_raw_csv(path)

    with tempfile.NamedTemporaryFile("w") as tmp_file:
        fields = list(raw_schema.keys())
        tmp_file.write(",".join(fields))
        tmp_file.write("\n")
        writer = csv.DictWriter(
            tmp_file, 
            fieldnames=fields,
        )
        for row in rows:
            writer.writerow(row)    
        tmp_file.flush()

        df = (
            pl
            .read_csv(tmp_file.name, schema=raw_schema)
            .rename({"": "index"})
            .with_columns(
                folder=pl.lit(str(path.parent.name)),
                fname=pl.lit(str(path.name)),
                location=(
                    pl.col("location")
                    .str
                    .to_lowercase()
                    .str
                    .strip_chars("\\")
                )
            )
        )

    if save_to is not None:
        df.write_parquet(
            save_to 
            / f"""str(path.parent.name).lower().replace(" ", "_")_{path.name}"""
        )
    return df


def load_raw_csv(path: pathlib.Path) -> pl.DataFrame:
    import tcbench
    path = pathlib.Path(path)
    schema = tcbench.get_dataset_polars_schema(
        DATASET_NAME.UTMOBILENET21,
        DATASET_TYPE.RAW,
    )
    return _parse_raw_csv_worker(path, schema)


class RawCSVParser:
    def __init__(self):
        import tcbench
        self.name = DATASET_NAME.UTMOBILENET21
        self.dset_schema = (
            tcbench.datasets_catalog()
            [DATASET_NAME.UTMOBILENET21]
            .get_schema(DATASET_TYPE.RAW)
        )

    def _parse_raw_csv(
        self, 
        *files: Iterable[pathlib.Path],
    ) -> pl.DataFrame:
        with tempfile.TemporaryDirectory() as tmp_folder:
            tmp_folder = pathlib.Path(tmp_folder)
            func = functools.partial(
                _parse_raw_csv_worker, 
                schema=self.dset_schema.to_polars(), 
                save_to=tmp_folder,
            )
            with (
                richutils.Progress(
                    description="Parse CSV files...", 
                    total=len(files)
                ) as progress,
                multiprocessing.Pool(processes=2) as pool,
            ):
                for _ in pool.imap_unordered(func, files):
                    progress.update()
            
            with richutils.SpinnerProgress(description="Reload..."):
                df = (
                    pl
                    .read_parquet(tmp_folder)
                    .sort(
                        "folder",
                        "fname",
                        "frame.number",
                    )
                )
        return df

    def run(
        self, 
        *files: Iterable[pathlib.Path], 
        save_to: pathlib.Path = None,
    ) -> pl.DataFrame:
        df = self._parse_raw_csv(*files)
        if save_to is None:
            save_to = pathlib.Path(".")
        with richutils.SpinnerProgress(description="Writing parquet files..."):
            fileutils.save_parquet(
                df, 
                save_as=save_to/f"{self.name}.parquet", 
                echo=False
            )
        return df


class RawPostorocessingPipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            dataset_name=DATASET_NAME.UCDAVIS19,
            description="Postprocess raw...",
            save_to=save_to,
        )

        stages = [
            SequentialPipelineStage(
                self._drop_columns,
                "Drop columns",
            ),
            SequentialPipelineStage(
                self._rename_columns,
                "Rename columns",
            ),
            SequentialPipelineStage(
                self._filter_broken_records,
                "Filter broken records",
            ),
            SequentialPipelineStage(
                self._filter_tcp_and_udp,
                "Filter TCP and UDP",
            ),
            SequentialPipelineStage(
                self._add_columns,
                "Add columns",
            ),
            SequentialPipelineStage(
                self._cast_types,
                "Cast types",
            ),
            SequentialPipelineStage(
                self._compose_flows,
                "Compose flows",
            ),
            SequentialPipelineStage(
                self._add_ip_columns,
                "Add IP columns",
            ),
            SequentialPipelineStage(
                self._add_pkts_dir,
                "Add packets direction",
            ),
            SequentialPipelineStage(
                self._is_valid_handshake_heuristic,
                "Adding TCP handshake check",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                "Computing stats",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._write_parquet_files,
                    fname_prefix="_postprocess"
                ),
                "Writing parquet files",
            )
        ]
        self.clear()
        self.extend(stages)

    def _drop_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop(
            "index",
            "frame.cap_len",
            "sll.pkttype",
            "sll.hatype",
            "sll.halen",
            "sll.src.eth",
            "sll.unused",
            "sll.etype",
            "ip.dsfield.ecn",
            "ip.id",
            "ip.frag_offset",
            "ip.ttl",
            "ip.checksum",
            "tcp.checksum",
            "tcp.urgent_pointer",
            "tcp.option_kind",
            "tcp.option_len",
            "tcp.options.timestamp.tsval",
            "tcp.options.timestamp.tsecr",
            "tcp.flags.ns",
            "tcp.flags.fin",
            "udp.checksum",
            "gquic.puflags.rsv",
            "gquic.packet_number",
        )

    def _rename_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            "frame.number": "fname_row_id",
            "frame.time": "timestamp",
            "frame.len": "l2_size",
            "ip.hdr_len": "l3_header_size",
            "ip.len": "l3_size",
            "ip.proto": "proto_id",
            "ip.src": "src_ip",
            "ip.dst": "dst_ip",
            "tcp.hdr_len": "tcp_header_size",
            "tcp.len": "tcp_size",
            "tcp.srcport": "tcp_src_port",
            "tcp.dstport": "tcp_dst_port",
            "tcp.seq": "tcp_seq",
            "tcp.ack": "tcp_ack",
            "tcp.window_size_value": "tcp_window_size",
            "udp.srcport": "udp_src_port",
            "udp.dstport": "udp_dst_port",
            "udp.length": "udp_size",
            "location": "partition",
        })

    def _filter_broken_records(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
                pl.col("src_ip").is_not_null() &
                pl.col("dst_ip").is_not_null() &
                pl.col("proto_id").is_not_null()
            )

    def _filter_tcp_and_udp(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            (pl.col("proto_id") == 6)
            .or_(
                pl.col("proto_id") == 17
            )
        )

    def _add_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            timestamp=(
                pl.col("timestamp")
                .str
                .to_datetime(
                    format="%b %d, %Y %H:%M:%S%.9f CDT"
                )
            ),
            proto=(
                pl.when(pl.col("proto_id") == 6)
                .then(pl.lit("tcp"))
                .otherwise(pl.lit("udp"))
            ),
            l4_size=(
                pl.when(pl.col("proto_id") == 6)
                .then(pl.col("tcp_size"))
                .otherwise(pl.col("udp_size"))
            ),
            is_ack=(
                (pl.col("proto_id") == 6)
                .and_(
                    pl.col("tcp_size") == 0
                )
            ),
            src_port=(
                pl.when(pl.col("proto_id") == 6)
                .then(pl.col("tcp_src_port"))
                .otherwise(pl.col("udp_src_port"))
            ),
            dst_port=(
                pl.when(
                    pl.col("proto_id") == 6
                )
                .then(
                    pl.col("tcp_dst_port")
                )
                .otherwise(
                    pl.col("udp_dst_port")
                )
            ),
            app=(
                pl.col("fname")
                .str
                .split("_")
                .list
                .first()
            ),
            action=(
                pl.col("fname")
                .str
                .extract("[^_]+_([^_]+)_.*")
            ),
        )

    def _cast_types(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("src_port").cast(pl.UInt32()),
            pl.col("dst_port").cast(pl.UInt32()),
            pl.col("l4_size").cast(pl.UInt32()),
            pl.col("l3_size").cast(pl.UInt32()),
            pl.col("tcp_window_size").cast(pl.UInt32()),
            pl.col("timestamp").cast(pl.UInt64()) / 1e9
        )

    def _compose_flows(self, df: pl.DataFrame) -> pl.DataFrame:
        def create_net_tuple(data):
            ip1 = data["src_ip"]
            port1 = str(data["src_port"])
            ip2 = data["dst_ip"]
            port2 = str(data["dst_port"])
            proto = data["proto"]
            
            if ip1 > ip2:
                ip1, ip2 = ip2, ip1
                port1, port2 = port2, port1
            return f"{ip1}:{port1}:{ip2}:{port2}:{proto}"

        return (
            df
            .with_columns(
                net_tuple=(
                    pl.struct(
                        "src_ip",
                        "src_port",
                        "dst_ip",
                        "dst_port",
                        "proto"
                    )
                    .map_elements(
                        function=create_net_tuple,
                        return_dtype=pl.String()
                    )
                )
            )
            .group_by(
                "folder",
                "fname",
                "partition",
                "net_tuple",
                "proto",
                "app",
                "action",
                maintain_order=True
            )
            .agg(
                pl.col("fname_row_id").alias("pkts_fname_row_id"),
                pl.col("timestamp").alias("pkts_timestamp"),
                pl.col("src_ip").alias("pkts_src_ip"),
                pl.col("dst_ip").alias("pkts_dst_ip"),
                pl.col("src_port").alias("pkts_src_port"),
                pl.col("dst_port").alias("pkts_dst_port"),
                pl.col("l3_size").alias("pkts_size"),
                pl.col("l4_size").alias("pkts_size_payload"),
                pl.col("is_ack").alias("pkts_is_ack"),
                pl.col("tcp_window_size").alias("pkts_tcp_window_size"),
            )
            .with_columns(
                src_ip=(
                    pl.col("pkts_src_ip").list.first()
                ),
                src_port=(
                    pl.col("pkts_src_port").list.first()
                ),
                dst_ip=(
                    pl.col("pkts_dst_ip").list.first()
                ),
                dst_port=(
                    pl.col("pkts_dst_port").list.first()
                ),
                packets=(
                    pl.col("pkts_size").list.len()
                ),
                bytes=(
                    pl.col("pkts_size").list.sum()
                ),
                bytes_payload=(
                    pl.col("pkts_size_payload").list.sum()
                ),
                duration=(
                    (
                        pl.col("pkts_timestamp").list.last() 
                        - pl.col("pkts_timestamp").list.first()
                    )
                )
            )
            .with_row_index("row_id")
        )

    def _add_ip_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            curation.add_ip_columns_validation(df)
            .with_columns(
                client_ip=(
                    pl.when(pl.col("src_ip_is_private"))
                    .then(pl.col("src_ip"))
                    .otherwise(pl.col("dst_ip"))
                )
            )
        )

    def _add_pkts_dir(self, df: pl.DataFrame) -> pl.DataFrame:
        def _define_direction(data):
            arr = data["pkts_src_ip"]
            value = data["client_ip"]
            return pl.Series(
                np.where(
                    np.atleast_1d(arr) == value,
                    1,
                    -1
                ).astype(np.int8)
            )

        def _reverse_direction_if_needed(data, direction, suffix):
            if data["src_ip"] != data["client_ip"]:
                if direction == "src":
                    direction = "dst"
                else:
                    direction = "src"
            field = f"{direction}_{suffix}"
            return data[field]
    
        df = (
            df
            .with_columns(
                pkts_dir=(
                    pl.struct(
                        "client_ip",
                        "pkts_src_ip",
                    )
                    .map_elements(
                        function=_define_direction,
                        return_dtype=pl.List(pl.Int8())
                    )
                ),
                ip_ports=pl.struct(
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "src_ip_is_private",
                    "src_ip_is_valid",
                    "dst_ip_is_private",
                    "dst_ip_is_valid",
                    "client_ip",
                )
            )
            # fix source/destination order (if needed)
            .with_columns(
                src_ip=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="src", 
                            suffix="ip"
                        ),
                        return_dtype=pl.String()
                    )
                ),
                src_port=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="src", 
                            suffix="port"
                        ),
                        return_dtype=pl.UInt32()
                    )
                ),
                dst_ip=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="dst", 
                            suffix="ip"
                        ),
                        return_dtype=pl.String()
                    )
                ),
                dst_port=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="dst", 
                            suffix="port"
                        ),
                        return_dtype=pl.UInt32()
                    )
                ),
                src_ip_is_private=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="src", 
                            suffix="ip_is_private"
                        ),
                        return_dtype=pl.Boolean()
                    )
                ),
                src_ip_is_valid=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="src", 
                            suffix="ip_is_valid"
                        ),
                        return_dtype=pl.Boolean()
                    )
                ),
                dst_ip_is_private=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="dst", 
                            suffix="ip_is_private"
                        ),
                        return_dtype=pl.Boolean()
                    )
                ),
                dst_ip_is_valid=(
                    pl.col("ip_ports")
                    .map_elements(
                        function=functools.partial(
                            _reverse_direction_if_needed, 
                            direction="dst", 
                            suffix="ip_is_valid"
                        ),
                        return_dtype=pl.Boolean()
                    )
                )
            )
            .drop("ip_ports")
        )
        return df

    def _is_valid_handshake_heuristic(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
                is_valid_handshake=(
                    pl.when(pl.col("proto") == "udp")
                    .then(pl.lit(True))
                    .otherwise(
                        (pl.col("packets") >= 3)
                        & (pl.col("pkts_is_ack").list.head(3).list.sum() == 3)
                        & (pl.col("pkts_dir").list.get(0, null_on_oob=True) == 1)
                        & (pl.col("pkts_dir").list.get(1, null_on_oob=True) == -1)
                        & (pl.col("pkts_dir").list.get(2, null_on_oob=True) == 1)
                    )
                )
            )


class CuratePipeline(BaseDatasetProcessingPipeline):
    def __init__(self, save_to: pathlib.Path):
        super().__init__(
            description="Curation...",
            dataset_name=DATASET_NAME.UTMOBILENET21,
            save_to=save_to,
            progress=True
        )

        stages = [
            SequentialPipelineStage(
                self._drop_not_valid_handshake,
                "Remove flows with invalid handshake",
            ),
            SequentialPipelineStage(
                self._drop_invalid_ip,
                "Remove flows with invalid IP address",
            ),
            SequentialPipelineStage(
                self._drop_dns,
                "Remove DNS",
            ),
            SequentialPipelineStage(
                self._add_columns,
                "Add columns",
            ),
            SequentialPipelineStage(
                self._compute_stats,
                "Compute stats",
            ),
            SequentialPipelineStage(
                self._compute_splits,
                "Compute splits",
            ),
            SequentialPipelineStage(
                functools.partial(
                    self._write_parquet_files,
                    fname_prefix=self.dataset_name
                ),
                "Write parquet files",
            ),
        ]
        self.clear()
        self.extend(stages)

    def _drop_not_valid_handshake(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            pl.col("is_valid_handshake")
        )

    def _drop_invalid_ip(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(curation.expr_are_ips_valid())
#            (~pl.col("src_ip_is_valid"))
#            .or_(~pl.col("dst_ip_is_valid"))
#            .or_(
#                pl.col("src_ip_is_private") 
#                .and_(pl.col("dst_ip_is_private"))
#            )
#        )
 
    def _drop_dns(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(
            ~curation.expr_is_dns_heuristic()
        )

    def _add_columns(self, df: pl.DataFrame):
        df = (
            df
            .with_columns(
                pkts_size_times_dir=(
                    curation.expr_pkts_size_times_dir(
                        "pkts_size",
                        "pkts_dir"
                    )
                ),
                pkts_size_times_dir_payload=(
                    curation.expr_pkts_size_times_dir(
                        "pkts_size_payload",
                        "pkts_dir"
                    )
                ),
            )
        )
        df = curation.add_flow_stats_by_direction(
            df,
            colname_pkts_size_times_dir_payload="pkts_size_times_dir_payload",
            colname_pkts_is_ack="pkts_is_ack",
        )
        return df

class UTMobilenet21(Dataset):
    def __init__(self):
        super().__init__(name=DATASET_NAME.UTMOBILENET21)

    def _install_raw(
        self, 
    ) -> pathlib.Path:
        url = self.metadata.raw_data_url_hidden[0]["UTMobileNet2021.zip"]
        UTMobilenet21DatasetInstaller(
            url=url,
            install_folder=self.install_folder,
            verify_tls=True,
            force_reinstall=True,
        )
        return self.install_folder

    @property
    def _list_raw_csv_files(self) -> Tuple[pathlib.Path]:
        return tuple(self.folder_raw.rglob("*.csv"))

    def raw(self) -> pl.DataFrame:
        return RawCSVParser().run(
            *self._list_raw_csv_files,
            save_to=self.folder_raw
        )

    def _raw_postprocess(self) -> pl.DataFrame:
        _ = self.load(DATASET_TYPE.RAW)
        df, *_ = (
            RawPostorocessingPipeline(
                save_to=self.folder_raw
            )
            .run(self.df)
        )
        return df

    def curate(self, recompute: bool = True) -> pl.DataFrame:
        fname = self.folder_raw / f"_postprocess.parquet"
        if not fname.exists() or recompute:
            df = self._raw_postprocess()
        else:
            with richutils.SpinnerProgress(
                description=f"Load {self.name}/raw postprocess..."
            ):
                df = fileutils.load_parquet(fname, echo=False)

        self.df, self.df_stats, self.df_splits = (
            CuratePipeline(
                save_to=self.folder_curate
            )
            .run(df)
        )
        return self.df
