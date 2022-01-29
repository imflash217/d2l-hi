import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = f"http://d2l-data.s3-accelerate.amazonaws.com/"

def download(name:str, cache_dir:str=os.path.join("../../../", "data")):
    """
    Downloads the file inserte dinto DATA_HUB.
    Returns:
        the local filename
    """
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.maakedirs(cache_dir, exists_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])

