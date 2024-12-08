from __future__ import annotations
import multiprocessing

import rich.table as richtable
import rich.box as richbox

import polars as pl

from typing import Dict, Any, Iterable, List, Tuple
from collections import UserDict, UserList, OrderedDict

import abc
import pathlib
import dataclasses
import rich.console

import tcbench
from tcbench import fileutils
from tcbench.datasets import (
    DATASET_NAME,
    DATASET_TYPE,
    curation,
    _constants,
)
from tcbench.cli import richutils


def get_dataset_folder(dataset_name: str | DATASET_NAME) -> pathlib.Path:
    """Returns the path where a specific datasets in installed"""
    return _constants.DATASETS_DEFAULT_INSTALL_ROOT_FOLDER / str(dataset_name)


def _from_schema_to_yaml(schema:pl.schema.Schema) -> Dict[str, Any]:
    data = dict()
    for field_name, field_dtype in schema.items():
        data[field_name] = dict(
            type=field_dtype._string_repr(),
            desc="",
            window="flow",
        )
    return data

def _remove_fields_from_schema(schema: pl.schema.Schema, *fields_to_remove: str) -> pl.schema.Schema:
    data = OrderedDict()
    for field_name, field_type in schema.items():
        if field_name in fields_to_remove:
            continue
        data[field_name] = field_type
    return pl.Schema(data)


@dataclasses.dataclass
class DatasetMetadata:
    name: DATASET_NAME
    desc: str = ""
    num_classes: int = -1
    url_paper: str = ""
    url_website: str = ""
    raw_data_url: str = ""
    raw_data_md5: dict = dataclasses.field(default_factory=dict)
    raw_data_url_hidden: dict = dataclasses.field(default_factory=dict)
    raw_data_size: str = ""
    curated_data_url: str = ""
    curated_data_md5: str = ""

    def __post_init__(self):
        self.folder_dset = tcbench.get_config().install_folder / str(self.name)
        self._schemas = dict()

        fname = _constants.DATASETS_RESOURCES_FOLDER / f"schema_{self.name}.yml"
        if fname.exists():
            data = fileutils.load_yaml(fname, echo=False)
            for dataset_type in DATASET_TYPE.values():
                if str(dataset_type) in data:
                    self._schemas[dataset_type] = DatasetSchema(
                        self.name,
                        DATASET_TYPE.from_str(dataset_type),
                        data[dataset_type]
                    )

    @property
    def folder_download(self):
        return self.folder_dset / "download"

    @property
    def folder_raw(self):
        return self.folder_dset / "raw"

    @property
    def folder_preprocess(self):
        return self.folder_dset / "preprocess"

    @property
    def folder_curate(self):
        return self.folder_dset / "curate"

    @property
    def is_raw(self) -> bool:
        return (self.folder_raw / f"{self.name}.parquet").exists()

    @property
    def is_curated(self) -> bool:
        return (self.folder_curate / f"{self.name}.parquet").exists()

    def get_schema(self, dataset_type: DATASET_TYPE) -> DatasetSchema:
        return self._schemas.get(str(dataset_type), None)

    def __rich__(self) -> richtable.Table:
        table = richtable.Table(show_header=False, box=richbox.SIMPLE_HEAD, show_footer=False, pad_edge=False)
        table.add_column("property")
        table.add_column("value", overflow="fold")

        table.add_row(":a: Description:", self.desc)
        table.add_row(":triangular_flag: Num. classes:", str(self.num_classes))
        table.add_row(
            ":link: Paper URL:", 
            f"[link={self.url_paper}]{self.url_paper}[/link]"
        )
        table.add_row(
            ":link: Website:", 
            f"[link={self.url_website}]{self.url_website}[/link]",
        )
        table.add_section()
        ###
        table.add_row(
            ":triangular_ruler: Raw data size:", 
            self.raw_data_size,
        )
        table.add_row(
            ":link: Raw data URL:", 
            f"[link={self.raw_data_url}]{self.raw_data_url}[/link]"
        )
        if len(self.raw_data_md5) == 1:
            table.add_row(
                ":heavy_plus_sign: Raw data MD5:", 
                list(self.raw_data_md5.values())[0]
            )
        else:
            table.add_row(
                ":heavy_plus_sign: Raw data MD5:", 
                "\n".join(
                    f"{name}: {md5}"
                    for name, md5 in self.raw_data_md5.items()
                )
            )
        table.add_section()
        ####
        if self.curated_data_url:
            table.add_row(
                ":link: Curated data URL:", 
                f"[link={self.curated_data_url}]{self.curated_data_url}[/link]"
                if self.curated_data_url else
                ""
            )
            table.add_row(
                ":heavy_plus_sign: Curated data MD5:", 
                self.curated_data_md5,
            )
            table.add_section()
        ###
        table.add_row(":file_folder: Root folder:", str(self.folder_dset))

        text = (":heavy_check_mark:" if self.is_raw else ":cross_mark:") + "Raw "
        text += (":heavy_check_mark:" if self.is_curated else ":cross_mark:") + "Curate"
        table.add_row(":arrow_down: Installed:", text)

        return table

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()





class RawDatasetInstaller:
    def __init__(
        self,
        url: str,
        install_folder: pathlib.Path = None,
        verify_tls: bool = True,
        force_reinstall: bool = False,
        #extra_unpack: Iterable[pathlib.Path] = None
    ):
        self.url = url
        self.install_folder = install_folder
        self.verify_tls = verify_tls
        self.force_reinstall = force_reinstall
        self.download_path = None

        if install_folder is None:
            self.install_folder = (
                _constants.DATASETS_DEFAULT_INSTALL_ROOT_FOLDER
            )

        self.install()

    def install(self) -> Tuple[pathlib.Path]:
        self.download_path = self.download()
        return self.unpack(self.download_path)

    def download(self) -> pathlib.Path:
        return fileutils.download_url(
            self.url,
            self.install_folder / "download",
            self.verify_tls,
            self.force_reinstall,
        )

    def _unpack(
        self, 
        path: pathlib.Path, 
        progress: bool = True,
        remove_dst: bool = True,
    ) -> pathlib.Path:
        func_unpack = None
        if path.suffix == ".zip":
            func_unpack = fileutils.unzip
        elif str(path).endswith(".tar.gz"):
            func_unpack = fileutils.untar
        else:
            raise RuntimeError(f"Unrecognized {path.suffix} archive")

        # do not change the destination folder
        # if path already under /raw 
        dst = self.install_folder / "raw"
        if str(path).startswith(str(self.install_folder / "raw")):
            dst = path.parent

        if (
            self.force_reinstall 
            or not dst.exists() 
            or len(list(dst.iterdir())) == 0
        ):
            return func_unpack(src=path, dst=dst, progress=progress, remove_dst=remove_dst)
        return dst

    def unpack(self, *paths: pathlib.Path) -> Tuple[pathlib.Path]:
        queue = []
        for path in paths:
            path = pathlib.Path(path)
            if fileutils.is_compressed_file(path):
                queue.append(path)
            elif path.is_dir():
                queue.extend(fileutils.list_compressed_files(path))

        res = []
        completed = set()
        with richutils.SpinnerAndCounterProgress(
            description="Unpack...",
            total=len(queue)
        ) as progress:
            while len(queue) > 0:
                # unpack one path
                path = queue.pop(0)
                res.append(
                    self._unpack(
                        path, progress=False, remove_dst=len(res) == 0
                    )
                )
                completed.add(path)
                # check if new archives have been appeared
                new_archives = (
                    set(fileutils.list_compressed_files(res[-1].parent)) 
                    - set(queue) 
                    - set(completed)
                )
                if new_archives:
                    queue.extend(list(new_archives))
                    progress.update_total(len(completed) + len(queue))
                progress.update()
        return tuple(res)


@dataclasses.dataclass
class DatasetSchemaField:
    name: str
    dtype_repr: str
    desc: str = ""
    window: str = ""
    lineage: str = ""

    def __post_init__(self):
        self._dtype = self._parse_dtype_repr(self.dtype_repr)

    def _parse_dtype_repr(self, text: str) -> Any:
        from polars.datatypes.convert import dtype_short_repr_to_dtype
        if "list" not in text:
            return dtype_short_repr_to_dtype(text)
        
        num_list = text.count("list")
        _, inner_dtype_repr = text[:-num_list].rsplit("[", 1)
        dtype = dtype_short_repr_to_dtype(inner_dtype_repr)
        while num_list:
            dtype = pl.List(dtype)
            num_list -= 1
        return dtype
        
    @property
    def dtype(self) -> Any:
        return self._dtype

class DatasetSchema:
    def __init__(
        self, 
        dataset_name: DATASET_NAME, 
        dataset_type: DATASET_TYPE,
        metadata: Dict[str, Any]
    ):
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.metadata = OrderedDict()
        self._schema = OrderedDict()
        for field_name, field_data in metadata.items():
            field = DatasetSchemaField(
                name=field_name,
                dtype_repr=field_data["type"],
                desc=field_data.get("desc", ""),
                window=field_data.get("window", ""),
                lineage=field_data.get("lineage", ""),
            )
            self.metadata[field_name] = field
            self._schema[field_name] = field.dtype

    @classmethod
    def from_dataframe(
        cls, 
        dataset_name: DATASET_NAME, 
        dataset_type: DATASET_TYPE, 
        df: pl.DataFrame
    ) -> DatasetSchema:
        metadata = OrderedDict()

        schema = None
        if isinstance(df, pl.DataFrame):
            schema = df.schema
        else:
            schema = df.collect_schema()

        for field_name, field_dtype in schema.items():
            metadata[field_name] = dict(
                type=field_dtype._string_repr(),
            )
        return DatasetSchema(dataset_name, dataset_type, metadata)

    @property
    def fields(self) -> List[str]:
        return list(self.metadata.keys())


    def to_yaml(self) -> Dict[str, Any]:
        data = dict()
        for field_name, field_data in self.metadata.items():
            data[field_name] = dict(
                type=field_data.dtype_repr,
                desc=field_data.desc,
                window=field_data.window
            )
        return {str(self.dataset_type): data}

    def to_polars(self) -> pl.Schema:
        return self._schema

    def __rich__(self) -> richtable.Table:
        import rich.markup

        has_any_lineage = any(
            field.lineage != "" 
            for field in self.metadata.values()
        )
        table = rich.table.Table(
            box=richbox.HORIZONTALS,
            show_header=True, 
            show_footer=False, 
            pad_edge=False
        )
        table.add_column("ID")
        table.add_column("Field", overflow="fold")
        table.add_column("Type")
        table.add_column("Window")
        table.add_column("Description", overflow="fold")
        if has_any_lineage:
            table.add_column("Lineage", overflow="fold")
        for idx, field in enumerate(self.metadata.values(), start=1):
            fields = [
                str(idx),
                field.name, 
                rich.markup.escape(field.dtype_repr),
                field.window,
                field.desc
            ]
            if has_any_lineage:
                fields.append(field.lineage)
            table.add_row(*fields)
        return table

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()


class Dataset:
    def __init__(self, name: DATASET_NAME):
        dataset_data = (
            fileutils.load_yaml(
                _constants.DATASETS_RESOURCES_METADATA_FNAME, 
                echo=False
            )
            .get(str(name), None)
        )
        if dataset_data is None:
            raise RuntimeError(f"Dataset {name} not recognized")
        if "raw_data_md5" in dataset_data:
            raw_data_md5 = dict()
            for item in dataset_data["raw_data_md5"]:
                raw_data_md5.update(item)
            dataset_data["raw_data_md5"] = raw_data_md5
        dataset_data["name"] = name
        self.name = name
        self.metadata = DatasetMetadata(**dataset_data)
        self.install_folder = tcbench.get_config().install_folder / str(self.name)
        self.y_colname = "app"
        self.index_colname = "row_id"
        self.df: pl.DataFrame | None = None
        self.df_stats: pl.DataFrame | None = None
        self.df_splits: pl.DataFrame | None = None
        self.metadata_schema = None

    @property
    def folder_download(self):
        return self.install_folder / "download"

    @property
    def folder_raw(self):
        return self.install_folder / "raw"

    @property
    def folder_preprocess(self):
        return self.install_folder / "preprocess"

    @property
    def folder_curate(self):
        return self.install_folder / "curate"

    @property
    def list_folder_raw(self) -> List[pathlib.Path]:
        return list(self.folder_raw.rglob("*"))

    def get_schema(self, dataset_type: DATASET_TYPE) -> DatasetSchema:
        return self.metadata.get_schema(dataset_type)

    def get_schema_polars(self, dataset_type: DATASET_TYPE) -> pl.Schema:
        return self.metadata.get_schema(dataset_type).to_polars()

    def install(
        self, 
        no_download:bool = False, 
        num_workers: int = -1,
    ) -> pathlib.Path:
        if not no_download:
            self._install_raw()

        self.raw(num_workers)
        self.curate()
        return self.install_folder

    def _install_raw(
        self, 
        #extra_unpack: Iterable[pathlib.Path]=None
    ) -> pathlib.Path:
        RawDatasetInstaller(
            url=self.metadata.raw_data_url,
            install_folder=self.install_folder,
            verify_tls=True,
            force_reinstall=True,
            #extra_unpack=extra_unpack,
        )
        return self.install_folder

    def compute_splits(
        self, 
        df: pl.DataFrame = None,
        num_splits: int = 10, 
        seed: int = 1, 
        test_size: float = 0.1,
    ) -> pl.DataFrame:
        from tcbench.modeling import splitting
        if df is None:
            df = self.df
        return splitting.split_monte_carlo(
            df,
            y_colname = self.y_colname,
            index_colname = self.index_colname, 
            num_splits = num_splits,
            seed = 1,
            test_size = 0.1,
        )

    def _load_schema(self, dataset_type: DATASET_TYPE) -> DatasetSchema:
        fname = _constants.DATASETS_RESOURCES_FOLDER / f"schema_{self.name}.yml"
        if not fname.exists():
            raise FileNotFoundError(fname)
        metadata = fileutils.load_yaml(fname, echo=False).get(str(dataset_type), None) 
        if metadata is None:
            raise RuntimeError(
                f"Dataset schema {self.name}.{dataset_type} not found"
            )

        return DatasetSchema(self.name, dataset_type, metadata)
        
    def load(
        self, 
        dataset_type: DATASET_TYPE, 
        n_rows: int | None = None, 
        min_packets:int | None = None,
        columns: Iterable[str] | None = None,
        lazy: bool = True,
        echo: bool = True,
    ) -> Dataset:
        folder = self.folder_curate
        if dataset_type == DATASET_TYPE.RAW:
            folder = self.folder_raw

        if min_packets is None or min_packets <= 0:
            min_packets = -1
        if columns is None:
            columns = [pl.col("*")]
        else:
            columns = list(map(str, columns))

        self.df: pl.DataFrame | pl.LazyFrame | None = None
        self.df_stats: pl.DataFrame | pl.LazyFrame | None = None
        self.df_splits: pl.DataFrame | pl.LazyFrame | None = None
        with richutils.SpinnerProgress(
            description=f"Loading {self.name}/{dataset_type}...",
            visible=echo,
        ):
            fname = folder / f"{self.name}.parquet"
            self.df = pl.scan_parquet(fname, n_rows=n_rows)
            if dataset_type != DATASET_TYPE.RAW:
                self.df = self.df.filter(
                    pl.col("packets") >= min_packets
                )
            self.df = self.df.select(*columns)
            if not lazy:
                self.df = self.df.collect()

            fname = folder / f"{self.name}_stats.parquet"
            if min_packets == -1 and fname.exists():
                self.df_stats = fileutils.load_parquet(
                    folder / f"{self.name}_stats.parquet",
                    echo=False,
                    lazy=lazy,
                )

            if (folder / f"{self.name}_splits.parquet").exists():
                self.df_splits = fileutils.load_parquet(
                    folder / f"{self.name}_splits.parquet",
                    echo=False,
                    lazy=lazy
                )

        self.metadata_schema = self._load_schema(dataset_type)
        return self

    @property
    def is_loaded(self) -> bool:
        return self.df is not None

    @abc.abstractmethod
    def raw(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def curate(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __rich__(self) -> richtable.Table:
        return self.metadata.__rich__()

    def __rich_console__(self,
        console: rich.console.Console,
        options: rich.console.ConsoleOptions,
    ) -> rich.console.RenderResult:
        yield self.__rich__()
        

class SequentialPipelineStage:
    def __init__(self, func, name:str = None, **kwargs):
        self.func = func
        self.name = name if name else ""
        self.run_kwargs = kwargs

    def run(self, *args) -> Any:
        return self.func(*args, **self.run_kwargs)
    
    def __repr__(self):
        return f"""{self.__class__.__name__}(name={self.name!r}, func={self.func.__name__}, run_kwargs_keys={", ".join(self.run_kwargs.keys())})"""


class SequentialPipelineStageRuntimeError(BaseException):
    """Exception raised when encountering a problem 
        when running a PipelineStage"""


class SequentialPipeline(UserList):
    def __init__(
        self, 
        *stages: SequentialPipelineStage,
        name: str = None,
        progress: bool = True
    ):
        super().__init__(stages)
        self.name = name if name is not None else ""
        self.progress = progress

    def remove_stage(self, name: str) -> bool:
        for idx, stage in enumerate(self):
            if stage.name == name:
                self.pop(idx)
                return True
        return False

    def replace_stage(self, name: str, new_stage: str) -> bool:
        for idx, stage in enumerate(self):
            if stage.name == name:
                self[idx] = new_stage
                return True
        return False

    def find_stage(self, name: str) -> SequentialPipelineStage:
        for idx in enumerate(self):
            if self[idx].name == name:
                return self[idx]
        return None

    def clear(self) -> None:
        self.data = []

    def run(self, *args) -> Any:
        names = [stage.name for stage in self]
        next_args = args
        with richutils.SpinnerAndCounterProgress(
            description=self.name,
            steps_description=names,
            total=len(self),
            visible=self.progress,
            newline_after_update=True,
        ) as progress:
            for idx, stage in enumerate(self.data):
                try:
                    next_args = stage.run(*next_args)
                except Exception as e:
                    msg = "Error at stage "
                    if self.name:
                        msg = f"Pipeline {self.name!r} @ stage "
                    msg += f"{idx+1}"
                    if stage.name:
                        msg += f" ({stage.name})"
                    raise SequentialPipelineStageRuntimeError(msg) from e
                if (
                    idx < len(self) - 1 
                    and not isinstance(next_args, (list, tuple))
                ):
                    next_args = (next_args,)
                progress.update()
        return next_args 

class BaseDatasetProcessingPipeline(SequentialPipeline):
    def __init__(
        self, 
        description: str,
        dataset_name: DATASET_NAME, 
        save_to: pathlib.Path,
        progress: bool = True,
        dataset_schema: DatasetSchema = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_schema = dataset_schema
        self.save_to = save_to

        stages = (
            SequentialPipelineStage(
                self.compute_stats,
                name="Compute statistics",
            ),
            SequentialPipelineStage(
                self.write_parquet_files,
                name="Write parquet files",
            ),
        )
        super().__init__(
            *stages, 
            name=description,
            progress=progress,
        )

    def compute_stats(self, df) -> Tuple[pl.DataFrame]:
        df_stats = curation.get_stats(df)
        return (df, df_stats)

    def write_parquet_files(
        self, 
        df: pl.DataFrame, 
        df_stats: pl.DataFrame = None, 
        df_splits: pl.DataFrame = None,
        fname_prefix: str = "_postprocess",
        columns: Iterable[str] = None,
    ) -> Tuple[pl.DataFrame]:

        if columns is None and self.dataset_schema:
            columns = self.dataset_schema.fields
        if columns:
            df = df.select(*columns)

        fileutils.save_parquet(
            df, 
            self.save_to / f"{fname_prefix}.parquet", 
            echo=False
        )
        if df_stats is not None:
            fileutils.save_parquet(
                df_stats,
                self.save_to / f"{fname_prefix}_stats.parquet",
                echo=False,
            )
        if df_splits is not None:
            fileutils.save_parquet(
                df_splits,
                self.save_to / f"{fname_prefix}_splits.parquet",
                echo=False,
            )
        return df, df_stats, df_splits
        
    def compute_splits(
        self, 
        df: pl.DataFrame, 
        *args: Any, 
        num_splits: int = 10,
        test_size: float = 0.1,
        y_colname: str = "app",
        index_colname: str = "row_id",
        seed: int = 1,
    ) -> Iterable[pl.DataFrame]:
        from tcbench.modeling import splitting
        df_splits = splitting.split_monte_carlo(
            df,
            y_colname=y_colname,
            index_colname=index_colname,
            num_splits=num_splits,
            seed=seed,
            test_size=test_size,
        )
        return (df, *args, df_splits)

# def install_ucdavis_icdm19(input_folder, num_workers=10, *args, **kwargs):
#    # moved here to speedup loading
#    from tcbench.libtcdatasets import ucdavis_icdm19_csv_to_parquet
#    from tcbench.libtcdatasets import ucdavis_icdm19_generate_splits
#
#    rich_label("unpack")
#    expected_files = [
#        "pretraining.zip",
#        "Retraining(human-triggered).zip",
#        "Retraining(script-triggered).zip",
#    ]
#
#    input_folder = pathlib.Path(input_folder)
#    for fname in expected_files:
#        path = input_folder / fname
#        if not path.exists():
#            raise RuntimeError(f"missing {path}")
#
#    dataset_folder = get_dataset_folder(DATASETS.UCDAVISICDM19)
#
#    # unpack the raw CSVs
#    raw_folder = dataset_folder / "raw"
#    if not raw_folder.exists():
#        raw_folder.mkdir(parents=True)
#    for fname in expected_files:
#        path = input_folder / fname
#        unzip(path, raw_folder)
#
#    # preprocess raw CSVs
#    rich_label("preprocess", extra_new_line=True)
#    preprocessed_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
#    args = ucdavis_icdm19_csv_to_parquet.cli_parser().parse_args(cmd.split())
#    ucdavis_icdm19_csv_to_parquet.main(args)
#
#    # generate data splits
#    rich_label("generate splits", extra_new_line=True)
#    ucdavis_icdm19_generate_splits.main(
#        dict(datasets={str(DATASETS.UCDAVISICDM19): preprocessed_folder})
#    )
#
#    verify_dataset_md5s(DATASETS.UCDAVISICDM19)
#
#
# def install_utmobilenet21(input_folder, num_workers=50):
#    # moved here to speed up loading
#    from tcbench.libtcdatasets import utmobilenet21_csv_to_parquet
#    from tcbench.libtcdatasets import utmobilenet21_generate_splits
#
#    # enforcing this to 50, attempting to replicate the
#    # original setting used to create the artifact
#    num_workers=50
#
#    rich_label("unpack")
#    expected_files = ["UTMobileNet2021.zip"]
#    input_folder = pathlib.Path(input_folder)
#
#    for fname in expected_files:
#        path = input_folder / fname
#        if not path.exists():
#            raise RuntimeError(f"missing {path}")
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'utmobilenet21'
#    dataset_folder = get_dataset_folder(DATASETS.UTMOBILENET21)
#
#    # unpack the raw CSVs
#    raw_folder = dataset_folder / "raw"
#    if not raw_folder.exists():
#        raw_folder.mkdir(parents=True)
#    for fname in expected_files:
#        path = input_folder / fname
#        unzip(path, raw_folder)
#
#    # preprocess raw CSVs
#    rich_label("preprocess", extra_new_line=True)
#    preprocessed_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --num-workers {num_workers} --output-folder {preprocessed_folder}"
#    args = utmobilenet21_csv_to_parquet.cli_parser().parse_args(cmd.split())
#    utmobilenet21_csv_to_parquet.main(args)
#
#    # generate data splits
#    rich_label("filter & generate splits", extra_new_line=True)
#    cmd = f"--config dummy_file.txt"
#    args = utmobilenet21_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.UTMOBILENET21): preprocessed_folder})
#    utmobilenet21_generate_splits.main(args)
#
#    #verify_dataset_md5s(DATASETS.UTMOBILENET21)
#
#
# def install_mirage22(input_folder=None, num_workers=30):
#    # moved here to speed up loading
#    from tcbench.libtcdatasets import mirage22_json_to_parquet
#    from tcbench.libtcdatasets import mirage22_generate_splits
#
#    rich_label("download & unpack")
#    expected_files = ["MIRAGE-COVID-CCMA-2022.zip"]
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage22'
#    dataset_folder = get_dataset_folder(DATASETS.MIRAGE22)
#
#    raw_folder = dataset_folder / "raw"
#    if input_folder:
#        _verify_expected_files_exists(input_folder, expected_files)
#        unzip(input_folder / expected_files[0], raw_folder)
#    else:
#        datasets_yaml = load_datasets_yaml()
#        url = datasets_yaml[str(DATASETS.MIRAGE22)]["data"]
#        with tempfile.TemporaryDirectory() as tmpfolder:
#            path = download_url(url, tmpfolder)
#            unzip(path, raw_folder)
#
#    # second unzip
#    files = [
#        "Discord.zip",
#        "GotoMeeting.zip",
#        "Meet.zip",
#        "Messenger.zip",
#        "Skype.zip",
#        "Slack.zip",
#        "Teams.zip",
#        "Webex.zip",
#        "Zoom.zip",
#    ]
#    raw_folder = raw_folder / "MIRAGE-COVID-CCMA-2022" / "Raw_JSON"
#    for fname in files:
#        path = raw_folder / fname
#        unzip(path, raw_folder)
#
#    rich_label("preprocess", extra_new_line=True)
#    preprocess_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder}"
#    args = mirage22_json_to_parquet.cli_parser().parse_args(cmd.split())
#    mirage22_json_to_parquet.main(args)
#
#    rich_label("filter & generate splits", extra_new_line=True)
#    # fooling the script believe there is a config.yml
#    cmd = f"--config dummy_file.txt"
#    args = mirage22_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.MIRAGE22): preprocess_folder})
#    mirage22_generate_splits.main(args)
#
#    verify_dataset_md5s(DATASETS.MIRAGE22)
#
#
# def install_mirage19(input_folder=None, num_workers=30):
#    from tcbench.libtcdatasets import mirage19_json_to_parquet
#    from tcbench.libtcdatasets import mirage19_generate_splits
#
#    rich_label("download & unpack")
#    expected_files = [
#        "MIRAGE-2019_traffic_dataset_downloadable_v2.tar.gz",
#    ]
#    # dataset_folder = _get_module_folder() / FOLDER_DATASETS / 'mirage19'
#    dataset_folder = get_dataset_folder(DATASETS.MIRAGE19)
#
#    raw_folder = dataset_folder / "raw"
#    if input_folder:
#        _verify_expected_files_exists(input_folder, expected_files)
#        untar(input_folder / expected_files[0], raw_folder)
#    else:
#        datasets_yaml = load_datasets_yaml()
#        url = datasets_yaml[str(DATASETS.MIRAGE19)]["data"]
#        with tempfile.TemporaryDirectory() as tmpfolder:
#            path = download_url(url, tmpfolder)
#            untar(path, raw_folder)
#
#    rich_label("preprocess", extra_new_line=True)
#    preprocess_folder = dataset_folder / "preprocessed"
#    cmd = f"--input-folder {raw_folder} --output-folder {preprocess_folder} --num-workers {num_workers}"
#    args = mirage19_json_to_parquet.cli_parser().parse_args(cmd.split())
#    mirage19_json_to_parquet.main(
#        args.input_folder, args.output_folder / "mirage19.parquet", args.num_workers
#    )
#
#    rich_label("filter & generate splits", extra_new_line=True)
#    # fooling the script believe there is a config.yml
#    cmd = f"--config dummy_file.txt"
#    args = mirage19_generate_splits.cli_parser().parse_args(cmd.split())
#    args.config = dict(datasets={str(DATASETS.MIRAGE19): preprocess_folder})
#    mirage19_generate_splits.main(args)
#
#    verify_dataset_md5s(DATASETS.MIRAGE19)
#
#
# def install(dataset_name, *args, **kwargs):
#    dataset_name = str(dataset_name).replace("-", "_")
#    curr_module = sys.modules[__name__]
#    func_name = f"install_{dataset_name}"
#    func = getattr(curr_module, func_name)
#    return func(*args, **kwargs)
#
#
# def get_dataset_parquet_filename(
#    dataset_name: str | DATASETS, min_pkts: int = -1, split: str = None, animation=False
# ) -> pathlib.Path:
#    """Returns the path of a dataset parquet file
#
#    Arguments:
#        dataset_name: The name of the dataset
#        min_pkts: the filtering rule applied when curating the datasets.
#            If -1, load the unfiltered dataset
#        split: if min_pkts!=-1, is used to request the loading of
#            the split file. For DATASETS.UCDAVISICDM19
#            values can be "human", "script" or a number
#            between 0 and 4.
#            For all other dataset split can be anything
#            which is not None (e.g., True)
#
#    Returns:
#        The pathlib.Path of a dataset parquet file
#    """
#    dataset_folder = get_dataset_folder(dataset_name) / "preprocessed"
#    path = dataset_folder / f"{dataset_name}.parquet"
#
#    if isinstance(split, int) and split < 0:
#        split = None
#
#    #    if isinstance(split, bool):
#    #        split = 0
#    #    elif isinstance(split, int):
#    #        split = str(split)
#    #
#    #    if min_pkts == -1 and (split is None or int(split) < 0):
#    #        return path
#    #
#    if isinstance(dataset_name, str):
#        dataset_name = DATASETS.from_str(dataset_name)
#
#    if dataset_name != DATASETS.UCDAVISICDM19:
#        if min_pkts != -1:
#            dataset_folder /= "imc23"
#            if split is None:
#                path = (
#                    dataset_folder
#                    / f"{dataset_name}_filtered_minpkts{min_pkts}.parquet"
#                )
#            else:
#                path = (
#                    dataset_folder
#                    / f"{dataset_name}_filtered_minpkts{min_pkts}_splits.parquet"
#                )
#    else:
#        #        if split is None:
#        #            raise RuntimeError('split cannot be None for ucdavis-icdm19')
#        #        dataset_folder /= 'imc23'
#        #        if split in ('human', 'script'):
#        #            path = dataset_folder / f'test_split_{split}.parquet'
#        #        else:
#        #            if split == 'train':
#        #                split = 0
#        #            path = dataset_folder / f'train_split_{split}.parquet'
#
#        if split is not None:
#            dataset_folder /= "imc23"
#            if split in ("human", "script"):
#                path = dataset_folder / f"test_split_{split}.parquet"
#            else:
#                if split == "train":
#                    split = 0
#                path = dataset_folder / f"train_split_{split}.parquet"
#
#    return path
#
#
# def load_parquet(
#    dataset_name: str | DATASETS,
#    min_pkts: int = -1,
#    split: str = None,
#    columns: List[str] = None,
#    animation: bool = False,
# ) -> pd.DataFrame:
#    """Load and returns a dataset parquet file
#
#    Arguments:
#        dataset_name: The name of the dataset
#        min_pkts: the filtering rule applied when curating the datasets.
#            If -1, load the unfiltered dataset
#        split: if min_pkts!=-1, is used to request the loading of
#            the split file. For DATASETS.UCDAVISICDM19
#            values can be "human", "script" or a number
#            between 0 and 4.
#            For all other dataset split can be anything
#            which is not None (e.g., True)
#        columns: A list of columns to load (if None, load all columns)
#        animation: if True, create a loading animation on the console
#
#    Returns:
#        A pandas dataframe and the related parquet file used to load the dataframe
#    """
#    path = get_dataset_parquet_filename(dataset_name, min_pkts, split)
#
#    import pandas as pd
#    from tcbench import cli
#
#    if animation:
#        with cli.console.status(f"loading: {path}...", spinner="dots"):
#            return pd.read_parquet(path, columns=columns)
#    return pd.read_parquet(path, columns=columns)
#
#
# def get_split_indexes(dataset_name, min_pkts=-1):
#    dataset_path = get_dataset_folder(dataset_name) / "preprocessed" / "imc23"
#    if str(dataset_name) == str(DATASETS.UCDAVISICDM19):  #'ucdavis-icdm19':
#        # automatically detect all split indexes
#        split_indexes = sorted(
#            [
#                int(split_path.stem.rsplit("_", 1)[1])
#                for split_path in dataset_path.glob("train_split*")
#            ]
#        )
#
#    # elif args.dataset in (str(DATASETS.MIRAGE19), str(DATASETS.MIRAGE22), str(DATASETS.UTMOBILENET21)): #'mirage19', 'mirage22', 'utmobilenet21'):
#    else:
#        #        prefix = f'{args.dataset}_filtered'
#        #        if args.dataset_minpkts != -1:
#        #            prefix = f'{prefix}_minpkts{args.dataset_minpkts}'
#        #        df_splits = pd.read_parquet(dataset_path / f'{prefix}_splits.parquet')
#        #        split_indexes = df_splits['split_index'].unique().tolist()
#
#        #    else:
#        #        df_splits = pd.read_parquet(dataset_path / f'{args.dataset}_filtered_splits.parquet')
#        #        df_splits = df_splits[df_splits['idx_inner_kfold'] == 0]
#        #        split_indexes = list(map(int, df_splits['split_index'].values))
#        df_splits = load_parquet(dataset_name, min_pkts=min_pkts, split=True)
#        split_indexes = list(map(int, df_splits["split_index"].values))
#
#    #    if args.max_train_splits == -1:
#    #        args.max_train_splits = len(split_indexes)
#    #
#    #    split_indexes = split_indexes[:min(len(split_indexes), args.max_train_splits)]
#
#    return split_indexes
#
# def import_dataset(dataset_name, path_archive):
#    data = load_datasets_yaml()
#    folder_datasets = _get_module_folder() #/ FOLDER_DATASETS
#
#    if dataset_name is None or str(dataset_name) not in data:
#        raise RuntimeError(f"Invalid dataset name {dataset_name}")
#
#    with tempfile.TemporaryDirectory() as tmpfolder:
#        if path_archive is None:
#            dataset_name = str(dataset_name)
#            if "data_curated" not in data[dataset_name]:
#                raise RuntimeError(f"The curated dataset cannot be downloaded (likely for licencing problems). Regenerate it using `tcbench datasets install --name {dataset_name}`")
#            url = data[dataset_name]["data_curated"]
#            expected_md5 = data[dataset_name]["data_curated_md5"]
#
#            try:
#                path_archive = download_url(url, tmpfolder)
#            except requests.exceptions.SSLError:
#                path_archive = download_url(url, tmpfolder, verify=False)
#
#            md5 = get_md5(path_archive)
#            assert md5 == expected_md5, f"MD5 check error: found {md5} while should be {expected_md5}"
#        untar(path_archive, folder_datasets)
#
# def verify_dataset_md5s(dataset_name):
#
#    def flatten_dict(data):
#        res = []
#        for key, value in data.items():
#            key = pathlib.Path(key)
#            if isinstance(value, str):
#                res.append((key, value))
#                continue
#            for inner_key, inner_value in flatten_dict(value):
#                res.append((key / inner_key, inner_value))
#        return res
#
#    dataset_name = str(dataset_name)
#    data_md5 = load_datasets_files_md5_yaml().get(dataset_name, None)
#    expected_files = flatten_dict(data_md5)
#
#    if dataset_name in (None, "") or data_md5 is None:
#        raise RuntimeError(f"Invalid dataset name {dataset_name}")
#
#    folder_dataset = _get_module_folder() / FOLDER_DATASETS / dataset_name
#    if not folder_dataset.exists():
#        raise RuntimeError(f"Dataset {dataset_name} is not installed. Run first \"tcbench datasets install --name {dataset_name}\"")
#
#    folder_dataset /= "preprocessed"
#
#    mismatches = dict()
#    for exp_path, exp_md5 in richprogress.track(expected_files, description="Verifying parquet MD5..."):
#        path = folder_dataset / exp_path
#        if not path.exists():
#            raise RuntimeError(f"File {path} not found")
#
#        found_md5 = get_md5(path)
#        fname = path.name
#        if found_md5 == exp_md5:
#            continue
#        mismatches[path] = (exp_md5, found_md5)
#
#    if mismatches:
#        console.print(f"Found {len(mismatches)}/{len(expected_files)} mismatches when verifying parquet files md5")
#        for path, (expected_md5, found_md5) in mismatches.items():
#            console.print()
#            console.print(f"path: {path}")
#            console.print(f"expected_md5: {expected_md5}")
#            console.print(f"found_md5: {found_md5}")
#    else:
#        console.print("All MD5 are correct!")
#
