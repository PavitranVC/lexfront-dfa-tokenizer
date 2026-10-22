"""peak_rss_gb works with and without the Unix `resource` module (Windows has none)."""
from experiments import peak_memory


def test_peak_rss_is_a_non_negative_float():
    value = peak_memory.peak_rss_gb()
    assert isinstance(value, float) and value >= 0.0


def test_without_resource_module_it_does_not_raise(monkeypatch):
    monkeypatch.setattr(peak_memory, "resource", None)
    value = peak_memory.peak_rss_gb()          # on Linux the Windows API is missing: falls back to 0.0
    assert isinstance(value, float) and value >= 0.0
