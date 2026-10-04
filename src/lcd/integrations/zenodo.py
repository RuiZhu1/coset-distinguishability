"""Fetch single files from Google's published QEC datasets on Zenodo without downloading whole archives.

The archives are zip files of several GB; ``ZipOverHttp`` reads only the central directory and the requested members
with HTTP range requests (standard library only).

    from lcd.integrations.zenodo import willow_dem
    dem = willow_dem(distance=5, basis="Z", rounds=10, prior="rl_optimized")   # about 0.4 MB

Data: Google Quantum AI, "Quantum error correction below the surface code threshold" (Nature 2025),
Zenodo 10.5281/zenodo.13273331, licence CC-BY 4.0.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

WILLOW_105Q = "https://zenodo.org/api/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip/content"


class _HttpFile(io.RawIOBase):
    def __init__(self, url: str):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


class ZipOverHttp:
    """A read-only zipfile.ZipFile on a remote archive, fetching byte ranges on demand."""

    def __init__(self, url: str = WILLOW_105Q):
        self.zip = zipfile.ZipFile(io.BufferedReader(_HttpFile(url), buffer_size=1 << 20))

    def names(self) -> list[str]:
        return self.zip.namelist()

    def read(self, name: str) -> bytes:
        return self.zip.read(name)


def willow_dem(distance: int, basis: str = "Z", rounds: int = 10, prior: str = "rl_optimized", patch: str | None = None,
               archive: ZipOverHttp | None = None):
    """The detector error model Google published with its correlated-matching decoder for one Willow experiment.

    ``prior`` is "rl_optimized" (fitted to the device's data) or "si1000" (a uniform model). ``patch`` defaults to the
    first patch of the given distance in the archive.
    """
    import stim
    z = archive or ZipOverHttp()
    names = z.names()
    patches = sorted({n.split("/")[1] for n in names if f"/d{distance}_at_" in n})
    if not patches:
        raise ValueError(f"no distance-{distance} patch in the archive")
    patch = patch or patches[0]
    member = (f"google_105Q_surface_code_d3_d5_d7/{patch}/{basis}/r{rounds:02d}/decoding_results/"
              f"correlated_matching_decoder_with_{prior}_prior/error_model.dem")
    if member not in names:
        raise ValueError(f"not in the archive: {member}")
    return stim.DetectorErrorModel(z.read(member).decode())
