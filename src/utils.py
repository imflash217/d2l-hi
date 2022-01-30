import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = f"http://d2l-data.s3-accelerate.amazonaws.com/"


def download(name: str, cache_dir: str = os.path.join("../", "data")):
    """
    Downloads the file inserte dinto DATA_HUB.
    Returns:
        the local filename
    """
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exists_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  ## return the cached file
    print(f"Downloaading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def download_extract(fname: str, folder: str = None):
    """Download & Extract a zip/tar file"""
    fname = download(fname)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "File type not supported. Only zip/tar files can be extracted"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else base_dir


def download_all():
    """Downloads all the datasets stored in DATA_HUB"""
    for name in DATA_HUB:
        download(name)
