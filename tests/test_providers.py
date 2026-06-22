"""Esri / PNOA provider tests — fully offline: `requests` is monkeypatched inside
each provider module and fake responses carry a real in-memory PNG."""
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dronomy_loc.reference import esri as esri_mod  # noqa: E402
from dronomy_loc.reference import pnoa as pnoa_mod  # noqa: E402
from dronomy_loc.reference.geo import haversine_m, mercator_bbox_around  # noqa: E402

LAT, LON = 43.521955, -5.624290  # the Asturias flight prior
SPAN, PIXELS = 1500.0, 8


def _png_bytes(size: int = PIXELS) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (40, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status_code=200, content_type="image/png", content=b"", text=""):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.content = content
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeRequests:
    """Stands in for the `requests` module binding inside a provider module.
    Serves the queued responses in order and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.responses.pop(0)


def _ok_png():
    return FakeResponse(content=_png_bytes())


# ── Esri ──────────────────────────────────────────────────────────────
def test_esri_happy_path(monkeypatch):
    fake = FakeRequests([_ok_png()])
    monkeypatch.setattr(esri_mod, "requests", fake)

    geo = esri_mod.EsriProvider().fetch(LAT, LON, SPAN, PIXELS)

    bbox = mercator_bbox_around(LON, LAT, SPAN)
    assert geo.bbox == bbox
    assert geo.image.shape == (PIXELS, PIXELS, 3)

    assert len(fake.calls) == 1
    p = fake.calls[0]["params"]
    assert p["bbox"] == ",".join(f"{v:.6f}" for v in bbox)  # minx,miny,maxx,maxy
    assert p["bboxSR"] == "3857"
    assert p["imageSR"] == "3857"
    assert p["f"] == "image"
    assert p["size"] == f"{PIXELS},{PIXELS}"


def test_esri_json_error_with_200(monkeypatch):
    body = '{"error":{"code":400,"message":"Invalid bbox"}}'
    fake = FakeRequests([FakeResponse(content_type="application/json", text=body)])
    monkeypatch.setattr(esri_mod, "requests", fake)

    with pytest.raises(RuntimeError, match="Invalid bbox"):
        esri_mod.EsriProvider().fetch(LAT, LON, SPAN, PIXELS)


def test_esri_retries_on_500_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(esri_mod, "time", types.SimpleNamespace(sleep=sleeps.append))
    fake = FakeRequests([FakeResponse(status_code=500), FakeResponse(status_code=500), _ok_png()])
    monkeypatch.setattr(esri_mod, "requests", fake)

    geo = esri_mod.EsriProvider().fetch(LAT, LON, SPAN, PIXELS)
    assert geo.image.shape == (PIXELS, PIXELS, 3)
    assert len(fake.calls) == 3
    assert sleeps == [1.0, 1.5]  # 1.5x backoff between attempts


def test_esri_gives_up_after_three_500s(monkeypatch):
    monkeypatch.setattr(esri_mod, "time", types.SimpleNamespace(sleep=lambda s: None))
    fake = FakeRequests([FakeResponse(status_code=500)] * 3)
    monkeypatch.setattr(esri_mod, "requests", fake)

    with pytest.raises(requests.HTTPError):
        esri_mod.EsriProvider().fetch(LAT, LON, SPAN, PIXELS)
    assert len(fake.calls) == 3


# ── PNOA ──────────────────────────────────────────────────────────────
def test_pnoa_happy_path(monkeypatch):
    fake = FakeRequests([_ok_png()])
    monkeypatch.setattr(pnoa_mod, "requests", fake)

    geo = pnoa_mod.PNOAProvider().fetch(LAT, LON, SPAN, PIXELS)

    bbox = mercator_bbox_around(LON, LAT, SPAN)
    assert geo.bbox == bbox
    assert geo.image.shape == (PIXELS, PIXELS, 3)

    p = fake.calls[0]["params"]
    assert p["VERSION"] == "1.3.0"
    assert p["CRS"] == "EPSG:3857"
    assert p["LAYERS"] == "OI.OrthoimageCoverage"
    assert p["WIDTH"] == str(PIXELS)
    assert p["HEIGHT"] == str(PIXELS)
    assert p["BBOX"] == ",".join(f"{v:.6f}" for v in bbox)  # minx,miny,maxx,maxy


def test_pnoa_xml_error_with_200(monkeypatch):
    body = '<?xml version="1.0"?><ServiceExceptionReport>layer not queryable</ServiceExceptionReport>'
    fake = FakeRequests([FakeResponse(content_type="text/xml", text=body)])
    monkeypatch.setattr(pnoa_mod, "requests", fake)

    with pytest.raises(RuntimeError, match="ServiceException"):
        pnoa_mod.PNOAProvider().fetch(LAT, LON, SPAN, PIXELS)


# ── Shared pixel<->geo contract ───────────────────────────────────────
@pytest.mark.parametrize("provider_mod,provider_cls", [
    (esri_mod, "EsriProvider"),
    (pnoa_mod, "PNOAProvider"),
])
def test_tile_center_roundtrips_within_1m(monkeypatch, provider_mod, provider_cls):
    fake = FakeRequests([_ok_png()])
    monkeypatch.setattr(provider_mod, "requests", fake)

    geo = getattr(provider_mod, provider_cls)().fetch(LAT, LON, SPAN, PIXELS)
    clon, clat = geo.pixel_to_lonlat(PIXELS / 2, PIXELS / 2)
    assert haversine_m(LAT, LON, clat, clon) < 1.0
