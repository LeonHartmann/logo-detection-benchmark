import os
import bench.config as cfg


def test_load_models_parses_rows():
    models = cfg.load_models()
    assert len(models) == 12
    m = {x.name: x for x in models}
    assert m["gpt-5.6-terra"].provider == "openai"
    assert m["gemini-3.1-pro"].model == "google/gemini-3.1-pro-preview"
    assert m["gpt-5.6-sol"].price_out == 30.0
    assert all(isinstance(x.enabled, bool) for x in models)


def test_load_bench_defaults():
    b = cfg.load_bench()
    assert b["rungs"] == [1080, 720, 480, 240, 144]
    assert b["jpeg_quality"] == 85
    assert b["concurrency"]["anthropic"] == 8


def test_load_brands():
    brands = cfg.load_brands()
    names = [b["name"] for b in brands]
    assert names == ["adidas", "stripes", "dkh", "11teamsports", "delay"]
    assert all("description" in b for b in brands)


def test_load_env_reads_and_does_not_overwrite(tmp_path):
    p = tmp_path / ".env"
    p.write_text('FOO_TEST_KEY="abc" \nBAR_TEST_KEY=def\n# comment\n')
    os.environ["BAR_TEST_KEY"] = "already"
    cfg.load_env(str(p))
    assert os.environ["FOO_TEST_KEY"] == "abc"
    assert os.environ["BAR_TEST_KEY"] == "already"
