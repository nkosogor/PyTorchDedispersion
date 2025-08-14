import torch
from pytorch_dedispersion.boxcar_filter import BoxcarFilter
from pytorch_dedispersion.candidate_finder import CandidateFinder

def test_boxcar_and_candidates_simple():
    # summed_data: (n_dm, n_time)
    n_dm, n_t = 3, 32
    summed = torch.zeros((n_dm, n_t), dtype=torch.float32)
    # put a spike at dm=1, t=10
    summed[1, 10] = 10.0

    widths = [1, 4]
    box = BoxcarFilter(summed)
    boxed = box.apply_boxcar(widths)  # expect (n_widths, n_dm, n_time) or similar

    # very loose shape checks that won't break if you change internals
    assert isinstance(boxed, torch.Tensor)
    assert boxed.shape[-1] == n_t
    assert n_dm in boxed.shape   # somewhere we still have the DM axis

    finder = CandidateFinder(boxed, window_size=None)
    cands = finder.find_candidates(snr_threshold=5.0, boxcar_widths=widths, remove_trend=False)

    # Expect at least one candidate near our inserted spike
    assert len(cands) >= 1
    # Most implementations report exact t and DM index for obvious spikes
    got = any((c["Boxcar Width"] in widths) and (abs(c["Time Sample"] - 10) <= 1) and (c["DM Index"] == 1)
              for c in cands)
    assert got

def test_find_candidates_with_trend_removal():
    n_dm, n_t = 2, 10
    data = torch.ones((1, n_dm, n_t), dtype=torch.float32)
    data[0, 1, 5] = 10.0  # spike

    finder = CandidateFinder(data, window_size=3)
    cands = finder.find_candidates(snr_threshold=2.0, boxcar_widths=[1], remove_trend=True)
    assert any(c["DM Index"] == 1 and c["Time Sample"] == 5 for c in cands)

def test_baseline_and_snr_direct():
    data = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    finder = CandidateFinder(data, window_size=5)

    baseline = finder.calculate_baseline(data)
    assert baseline.shape == data.shape

    snr = finder.calculate_snr(data)
    assert torch.isclose(snr.mean(), torch.tensor(0.0), atol=1e-6)