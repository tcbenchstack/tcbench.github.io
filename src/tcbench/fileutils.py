from __future__ import annotations

import polars as pl

from typing import Any, Dict

import pickle
import yaml
import pathlib
import requests
import tarfile
import zipfile
import os
import shutil

from tcbench import cli
from tcbench.cli import richutils

def _check_file_exists(path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    if not path.exists():
        raise RuntimeError(f"FileNotFound: {path}")
    return path

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
        yaml.dump(data, fout)

def load_csv(path: pathlib.Path, echo: bool = True) -> pl.DataFrame:
    cli.logger.log(f"loading: {path}", echo=echo)
    return pl.read_csv(path)

def save_csv(df: pl.DataFrame, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    df.write_csv(save_as)

def load_parquet(path: pathlib.Path, echo: bool = True) -> pl.DataFrame:
    cli.logger.log(f"loading: {path}", echo=echo)
    return pl.read_parquet(path)

def save_parquet(df: pl.DataFrame, save_as: pathlib.Path, echo: bool = True) -> None:
    save_as = pathlib.Path(save_as)
    create_folder(save_as.parent)
    cli.logger.log(f"saving: {save_as}", echo=echo)
    df.write_parquet(save_as)


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

    resp = requests.get(url, stream=True, verify=verify_tls)
    totalbytes = int(resp.headers.get("content-length", 0))

    if not save_as.parent.exists():
        save_as.parent.mkdir(parents=True)

    with (
        richutils.FileDownloadProgress(totalbytes=totalbytes) as progressbar,
        open(str(save_as), "wb") as fout,
    ):
        for data in resp.iter_content(chunk_size=64 * 1024):
            size = fout.write(data)
            progressbar.update(advance=size)

    return save_as


def _verify_expected_files_exists(folder, expected_files):
    for fname in expected_files:
        path = folder / fname
        if not path.exists():
            raise RuntimeError(f"missing {path}")


def unzip(src: str | pathlib.Path, dst: str | pathlib.Path = None) -> pathlib.Path:
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

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)

    with richutils.SpinnerProgress(f"Unpacking..."), zipfile.ZipFile(src) as fzipped:
        fzipped.extractall(dst)
    return dst


def untar(src: pathlib.Path, dst: pathlib.Path = None) -> pathlib.Path:
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

    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)

    with richutils.SpinnerProgress(f"Unpacking..."), tarfile.open(src, "r:gz") as ftar:
        ftar.extractall(dst)
    return dst
