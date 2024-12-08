from __future__ import annotations

import polars as pl

from typing import Any, Dict

import pickle
import yaml
import pathlib
import requests
import tarfile
import zipfile
import shutil
import sys

from tcbench import cli
from tcbench.cli import richutils

def _get_module_folder(name: str) -> pathlib.Path:
    module = sys.modules[name]
    folder = pathlib.Path(module.__file__).parent
    return folder

def _check_file_exists(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"FileNotFound: {path}")
    return path


def _convert_data_to_yaml(data: Dict[str, Any]) -> Dict[str, int|float|str|bool]:
    new_data = dict()
    for key, value in data.items():
        new_data[key] = value
        if value is not None:
            if isinstance(value, (int, float, bool, str)):
                continue
            if isinstance(value, pathlib.Path):
                new_data[key] = str(value)
                continue
            if isinstance(value, (set, tuple, list)):
                new_data[key] = list(map(str, value))
                continue
            if isinstance(value, dict):
                new_data[key] = _convert_data_to_yaml(value)
                continue
            else:
                new_data[key] = str(value)
    return new_data


def create_folder(folder: pathlib.Path, echo: bool = True) -> pathlib.Path:
    folder = pathlib.Path(folder)
    if not folder.exists():
        cli.logger.log(f"creating folder: {folder}", echo=echo)
        folder.mkdir(parents=True)
    return folder


def load_pickle(path: pathlib.Path, echo: bool = True) -> Any:
    path = _check_file_exists(path)
    cli.logger.log(f"reading: {path}", echo=echo)
    with open(path, "rb") as fin:
        data = pickle.load(fin)
    return data


def save_pickle(data: Any, save_as: pathlib.Path, echo: bool = True) -> Any:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    with open(save_as, "wb") as fout:
        pickle.dump(data, fout)
    return data


def load_yaml(path: pathlib.Path, echo: bool = True) -> Dict[Any, Any]:
    cli.logger.log(f"reading: {path}", echo=echo)
    with open(path) as fin:
        return yaml.safe_load(fin)


def save_yaml(data: Any, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    with open(save_as, "w") as fout:
        yaml.dump(_convert_data_to_yaml(data), fout, sort_keys=False)


def load_csv(path: pathlib.Path, echo: bool = True) -> pl.DataFrame:
    cli.logger.log(f"loading: {path}", echo=echo)
    return pl.read_csv(path)


def save_csv(df: pl.DataFrame, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    df.write_csv(save_as)


def load_parquet(
    path: pathlib.Path, 
    *,
    echo: bool = True,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    cli.logger.log(f"loading: {path}", echo=echo)
    if not lazy:
        return pl.read_parquet(path)
    return pl.scan_parquet(path)


def save_parquet(
    df: pl.DataFrame | pl.LazyFrame, 
    save_as: pathlib.Path, 
    echo: bool = True
) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent, echo=echo)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    if isinstance(df, pl.DataFrame):
        df.write_parquet(save_as)
    else:
        df.collect().write_parquet(save_as)


LOAD_FUNC_BY_SUFFIX = {
    ".yaml": load_yaml,
    ".pkl": load_pickle,
    ".csv": load_csv,
    ".parquet": load_parquet,
}


def load_if_exists(path: pathlib.Path, echo: bool = True, error_policy: str = "return") -> Any:
    path = pathlib.Path(path)
    if not path.exists():
        if error_policy == "return":
            return None
        raise RuntimeError(f"FileNotFound: {path}")

    func = LOAD_FUNC_BY_SUFFIX.get(path.suffix, None)
    if not func:
        raise RuntimeError(f"Unrecognized suffix {path.suffix}")
    return func(path, echo=echo)


def save_txt(text: str, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    if text[-1] != "\n":
        text += "\n"
    cli.logger.log(f"saving: {save_as}", echo=echo)
    save_as.write_text(text)


def load_txt(path: pathlib.Path, echo: bool = True) -> str:
    path = _check_file_exists(path)
    cli.logger.log(f"loading: {path}", echo=echo)
    return path.read_text()


def save_sys_argv(save_to: pathlib.Path, echo: bool = True) -> None:
    save_to = pathlib.Path(save_to)
    cmd = " ".join(sys.argv) + "\n"
    save_txt(cmd, save_to / "command.txt", echo)



def _download_via_requests(
    url: str,
    save_as: pathlib.Path,
    verify_tls: bool = True
) -> pathlib.Path:
    resp = requests.get(url, stream=True, verify=verify_tls)
    totalbytes = int(resp.headers.get("content-length", 0))
    if not save_as.parent.exists():
        save_as.parent.mkdir(parents=True)
    with (
        richutils.FileDownloadProgress(totalbytes=totalbytes) as progress,
        open(str(save_as), "wb") as fout,
    ):
        for data in resp.iter_content(chunk_size=64 * 1024):
            size = fout.write(data)
            progress.update(advance=size)
    return save_as


def _download_via_gdown(
    url: str,
    save_to: pathlib.Path,
    verify_tls: bool = True,
) -> pathlib.Path:
    import gdown

    func = gdown.download
    if "/folders/" in url:
        func = gdown.download_folder

    with richutils.SpinnerProgress(description="Downloading..."):
        func(url, output=str(save_to), quiet=True, verify=verify_tls)
    return save_to
    

def download_url(
    url: str,
    save_to: pathlib.Path,
    verify_tls: bool = True,
    force_redownload: bool = False,
) -> pathlib.Path:
    """Download content via URL.

    Args:
        url: the object URL.
        save_to: the destination folder.
        verify_tls: if False, skip TLS verification when downloading.
        force_redownload: if False, return with no action if the destination folder
            already contains a file with the expected name

    Returns:
        the path of the downloaded file
    """
    # from tcbench.cli import get_rich_console
    save_to = pathlib.Path(save_to)

    fname = pathlib.Path(url).name
    save_as = save_to / fname
    if save_as.exists() and not force_redownload:
        return save_as

    if str(url).startswith("https://drive.google.com"):
        return _download_via_gdown(url, save_to)
    return _download_via_requests(url, save_as, verify_tls)


def _verify_expected_files_exists(folder, expected_files):
    for fname in expected_files:
        path = folder / fname
        if not path.exists():
            raise RuntimeError(f"missing {path}")


def is_compressed_file(path: pathlib.Path) -> bool:
    path = pathlib.Path(path)
    return path.suffix in (".zip", ".gz", ".tar")


def list_compressed_files(path: pathlib.Path) -> List[pathlib.Path]:
    res = []
    for suffix in (".zip", ".gz", ".tar"):
        res.extend(list(path.rglob(f"*{suffix}")))
    return res


def unzip(
    src: str | pathlib.Path, 
    dst: str | pathlib.Path | None = None, 
    progress:bool=True,
    remove_dst: bool = True
) -> pathlib.Path:
    """Unpack a zip archive.

    Arguments:
        src: path of the .zip archive
        dst: destination folder. If None, the archive is
            unpacked in the same folder of src

    Returns:
        the destination folder
    """
    src = pathlib.Path(src)
    if dst is None:
        dst = src.parent
    else:
        dst = pathlib.Path(dst)

    if dst.exists() and remove_dst:
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)

    with (
        richutils.SpinnerProgress("Unpacking...", visible=progress), 
        zipfile.ZipFile(src) as fzipped,
    ):
        fzipped.extractall(dst)
    return dst


def untar(
    src: pathlib.Path, 
    dst: pathlib.Path | None = None, 
    progress: bool = True,
    remove_dst: bool = True,
) -> pathlib.Path:
    """Unpack a tarball archive.

    Arguments:
        src: path of the .zip archive
        dst: destination folder. If None, the archive is
            unpacked in the same folder of src

    Returns:
        the destination folder
    """
    src = pathlib.Path(src)
    if dst is None:
        dst = src.parent
    else:
        dst = pathlib.Path(dst)

    if dst.exists() and remove_dst:
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True)

    with (
        richutils.SpinnerProgress("Unpacking...", visible=progress), 
        tarfile.open(src, "r:gz") as ftar
    ):
        ftar.extractall(dst)
    return dst
