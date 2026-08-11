from PIL import Image
from bench.prompt import build_prompt, load_refs, RETRY_SUFFIX

BRANDS = [
    {"name": "adidas", "description": "adidas logo", "refs": []},
    {"name": "delay", "description": "Delay Sports crest", "refs": []},
]


def test_prompt_contains_brands_and_contract():
    p = build_prompt(BRANDS)
    assert '"adidas"' in p and '"delay"' in p
    assert "0-1000" in p
    assert '"detections"' in p
    assert "small|medium|large" in p
    assert "foreground|background" in p
    assert "conf" in p


def test_prompt_mentions_refs_only_when_present():
    assert "REFERENCE" not in build_prompt(BRANDS)
    p = build_prompt(BRANDS, ref_labels=["a labeled reference sheet of all brand marks",
                                         'a reference crop of the "delay" mark'])
    assert "first 2 images" in p.lower()
    assert "reference sheet" in p
    assert 'crop of the "delay" mark' in p
    assert "TARGET" in p


def test_prompt_zoom_instructions_only_when_enabled():
    assert '"zoom"' not in build_prompt(BRANDS)
    p = build_prompt(BRANDS, zoom=True)
    assert '{"zoom": [x0,y0,x1,y1]}' in p
    assert "3 times" in p
    assert "ALWAYS relative to the full original TARGET image" in p


def test_load_refs_reads_configured_images(tmp_path):
    (tmp_path / "data" / "refs").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(tmp_path / "data" / "refs" / "a.jpg")
    brands = [{"name": "adidas", "description": "x", "refs": ["data/refs/a.jpg"]},
              {"name": "delay", "description": "y", "refs": []}]
    refs = load_refs(brands, str(tmp_path))
    assert len(refs) == 1
    assert refs[0][1] == 'a reference crop of the "adidas" mark'
    assert refs[0][0][:2] == b"\xff\xd8"  # JPEG magic


def test_load_refs_prepends_reference_sheet(tmp_path):
    from PIL import Image
    (tmp_path / "data" / "refs").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(tmp_path / "data" / "refs" / "mood.jpg")
    brands = [{"name": "adidas", "description": "x", "refs": []}]
    refs = load_refs(brands, str(tmp_path), sheet_path="data/refs/mood.jpg")
    assert len(refs) == 1
    assert refs[0][1] == "a labeled reference sheet of all brand marks"


def test_retry_suffix_demands_bare_json():
    assert "JSON" in RETRY_SUFFIX
