import csv
import os
import torch
from pytorch_dedispersion.dedisperse_candidates import generate_dm_range, save_candidates_to_csv

def test_generate_dm_range_basic():
    cfg = [{"start": 0.0, "stop": 2.0, "step": 0.5}, {"start": 3.0, "stop": 4.0, "step": 0.5}]
    out = generate_dm_range(cfg)
    # torch.arange stop is exclusive
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0, 1.5, 3.0, 3.5]))

def test_save_candidates_to_csv(tmp_path):
    cands = [
        {"SNR": 10.2, "Time Sample": 5, "Time (sec)": 0.05, "Boxcar Width": 4, "DM Value": 12.5},
        {"SNR": 7.1,  "Time Sample": 9, "Time (sec)": 0.09, "Boxcar Width": 8, "DM Value": 20.0},
    ]
    f = tmp_path / "cands.csv"
    save_candidates_to_csv(cands, str(f))
    assert f.exists() and f.stat().st_size > 0

    with f.open() as fh:
        r = list(csv.reader(fh))
    assert r[0] == ["SNR","Sample Number","Time (sec)","Boxcar Width","DM Value"]
    assert len(r) == 3

