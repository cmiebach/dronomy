"""Offline tests for the resumable/integrity-checked downloader (data/fetch.py).

No network: the HTTP layer is injected via the `opener` seam, so we exercise the
byte logic (resume offset, Range honoured vs ignored, size verification, atomic
commit, idempotency) against an in-memory source.
"""
import pytest

from dronomy_loc.data.fetch import download_file, human_bytes


def make_opener(full: bytes, *, force_status=None, chunk=7, calls=None):
    """Fake opener serving `full` from a start offset. force_status=200 models a
    server that ignores Range and resends the whole body from byte 0."""
    def opener(url, start):
        if calls is not None:
            calls.append(start)
        if force_status == 200:
            status, body = 200, full
        elif start > 0:
            status, body = 206, full[start:]
        else:
            status, body = 200, full
        headers = {"Content-Length": str(len(body))}
        chunks = [body[i:i + chunk] for i in range(0, len(body), chunk)] or [b""]
        return status, headers, iter(chunks)
    return opener


def test_full_download_writes_dest_and_removes_part(tmp_path):
    data = b"dronomy-payload-" * 100
    dest = tmp_path / "video.mp4"
    out = download_file(data and "http://x", dest, expected_bytes=len(data),
                        opener=make_opener(data))
    assert out == dest
    assert dest.read_bytes() == data
    assert not dest.with_suffix(".mp4.part").exists()


def test_resume_from_partial_appends_only_remainder(tmp_path):
    data = bytes(range(256)) * 20
    dest = tmp_path / "f.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(data[:1000])           # a prior interrupted transfer
    calls = []
    download_file("http://x", dest, expected_bytes=len(data),
                  opener=make_opener(data, calls=calls))
    assert dest.read_bytes() == data
    assert calls == [1000]                   # asked the server to resume at 1000


def test_server_ignoring_range_restarts_clean(tmp_path):
    data = b"abcdefgh" * 50
    dest = tmp_path / "f.bin"
    dest.with_suffix(".bin.part").write_bytes(b"GARBAGE-partial")
    download_file("http://x", dest, expected_bytes=len(data),
                  opener=make_opener(data, force_status=200))
    assert dest.read_bytes() == data         # stale partial discarded, not appended


def test_size_mismatch_raises_and_keeps_part(tmp_path):
    data = b"twelve bytes"                    # 12 bytes actually served
    dest = tmp_path / "f.bin"
    with pytest.raises(IOError):
        download_file("http://x", dest, expected_bytes=9999, opener=make_opener(data))
    assert not dest.exists()                  # never promoted to final
    assert dest.with_suffix(".bin.part").exists()  # left for a future resume


def test_idempotent_when_dest_already_complete(tmp_path):
    data = b"already here"
    dest = tmp_path / "f.bin"
    dest.write_bytes(data)
    calls = []
    download_file("http://x", dest, expected_bytes=len(data),
                  opener=make_opener(data, calls=calls))
    assert calls == []                        # no fetch attempted


def test_complete_part_is_committed_without_fetch(tmp_path):
    data = b"finished but unrenamed"
    dest = tmp_path / "f.bin"
    dest.with_suffix(".bin.part").write_bytes(data)
    calls = []
    download_file("http://x", dest, expected_bytes=len(data),
                  opener=make_opener(data, calls=calls))
    assert dest.read_bytes() == data
    assert calls == []


def test_human_bytes():
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(1536) == "1.5 KB"
    assert human_bytes(3_761_828_779).endswith("GB")
