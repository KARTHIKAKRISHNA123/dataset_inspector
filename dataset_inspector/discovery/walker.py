"""DatasetWalker — streaming recursive discovery that yields fully-populated FileMeta objects.

Memory: O(directory depth) for the DFS stack + one FileMeta at a time (it's a generator). Time:
O(#entries) plus an O(1) prefix read per file. Archives are enumerated (central directory only),
never extracted. This is the same streaming discipline as the rest of the tool.
"""

from __future__ import annotations

import hashlib
import os
import stat as stat_mod
from typing import Iterator, List

from ..config import Config
from ..core.io import read_head_bytes
from ..core.mime import guess_mime
from ..core.models import Compression, FileMeta, FileType
from ..logging_utils import get_logger
from . import detectors as D


class DatasetWalker:
    def __init__(self, config: Config):
        self.cfg = config
        self.log = get_logger("discovery")
        d = config.section("discovery")
        self.data_exts: List[str] = list(d.get("data_extensions", []))
        self.comp_exts: List[str] = list(d.get("compression_extensions", []))
        self.ignore_dirs = set(d.get("ignore_dirs", []))
        self.expand_archives = bool(d.get("expand_archives", True))
        self.quick_hash = bool(d.get("quick_hash", True))
        self.record_perms = bool(d.get("record_permissions", True))
        self.sample_bytes = int(config.get("io.detection_sample_bytes", 65536))
        self.enc_fallback = str(config.get("io.encoding_fallback", "utf-8"))

    # ------------------------------------------------------------------
    def walk(self, root: str) -> Iterator[FileMeta]:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            self.log.error("input is not a directory: %s", root)
            return
        stack: List[str] = [root]
        while stack:
            cur = stack.pop()
            try:
                entries = list(os.scandir(cur))
            except (PermissionError, FileNotFoundError, OSError) as exc:
                self.log.warning("cannot list %s: %s", cur, exc)
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if e.name not in self.ignore_dirs:
                            stack.append(e.path)
                        continue
                    if not e.is_file(follow_symlinks=False):
                        continue
                except OSError as exc:
                    self.log.warning("stat failed %s: %s", e.path, exc)
                    continue
                yield from self._file_metas(e.path, root)

    # ------------------------------------------------------------------
    def _is_candidate(self, path: str) -> bool:
        e = D.ext_of(path)
        return e in self.comp_exts or e in self.data_exts or e == ""

    def _file_metas(self, path: str, root: str) -> Iterator[FileMeta]:
        if not self._is_candidate(path):
            return
        try:
            st = os.stat(path)
        except OSError as exc:
            self.log.warning("cannot stat %s: %s", path, exc)
            return
        head = read_head_bytes(path, self.sample_bytes)
        compression = D.detect_compression(path, head)

        if compression is Compression.ZIP and self.expand_archives:
            yield from self._zip_members(path, root, st)
            return

        ftype = D.detect_file_type(path, compression, head)
        if ftype is FileType.UNKNOWN and D.ext_of(path) == "":
            return  # extensionless + unrecognised → not data

        if ftype in (FileType.PARQUET, FileType.DOCX, FileType.GGUF):
            encoding, conf = "binary", 1.0
        else:
            enc_sample = D._decompressed_head(path, compression, head)
            encoding, conf = D.detect_encoding(enc_sample, self.enc_fallback)

        yield self._build(path, root, st, ftype, compression, encoding, conf, head,
                          archive_path=None, member=None, member_size=None)

    def _zip_members(self, archive: str, root: str, st: os.stat_result) -> Iterator[FileMeta]:
        members = D.list_zip_members(archive, self.data_exts)
        if not members:
            self.log.warning("empty/unreadable zip skipped: %s", archive)
            return
        for member, msize in members:
            ftype = D.detect_file_type(member, Compression.NONE, b"")
            yield self._build(member, root, st, ftype, Compression.ZIP, self.enc_fallback, 0.0,
                              b"", archive_path=archive, member=member, member_size=msize)

    # ------------------------------------------------------------------
    def _build(self, path, root, st, ftype, compression, encoding, conf, head,
               archive_path, member, member_size) -> FileMeta:
        # For archive members the physical file is the archive; the logical name is the member.
        display_path = archive_path if archive_path else path
        rel = os.path.relpath(display_path, root)
        dataset = rel.split(os.sep, 1)[0] if rel not in (".", "..") else "(root)"
        name = os.path.basename(member) if member else os.path.basename(path)
        size = member_size if member_size is not None else st.st_size
        perms = oct(stat_mod.S_IMODE(st.st_mode)) if self.record_perms else None
        ext = D.ext_of(member or path)

        return FileMeta(
            file_name=name,
            absolute_path=os.path.abspath(display_path),
            relative_path=rel,
            parent_dir=os.path.dirname(os.path.abspath(display_path)),
            dataset=dataset,
            source=D.detect_source(display_path),
            extension=ext,
            file_type=ftype,
            mime_type=guess_mime(member or path, ext),
            compression=compression,
            size_bytes=size,
            mtime=st.st_mtime,
            encoding=encoding,
            encoding_confidence=round(conf, 3),
            sha1_quick=(hashlib.sha1(head).hexdigest() if (self.quick_hash and head) else None),
            archive_path=archive_path,
            archive_member=member,
            permissions=perms,
        )
