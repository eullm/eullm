"""`tools/lumi/vram_slopes.py` reads the runner's job output and the node's
samples, and must say which server's VRAM climbed and which device emptied
because its server died: on 19-20-10-2026 every soak server on LUMI-G grew
1.3-2.5 GiB an hour until its GCD was full, and this is the check that a fix
holds.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "lumi" / "vram_slopes.py"

spec = importlib.util.spec_from_file_location("vram_slopes", SCRIPT)
vram_slopes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vram_slopes)

GIB = 2**30
T0 = datetime(2026, 10, 19, 19, 29, 5, tzinfo=timezone.utc).timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def write_job(queue: Path, job: str) -> None:
    (queue / "logs").mkdir(parents=True)
    (queue / "results").mkdir(parents=True)
    out = [
        f"[{iso(T0)}] job {job} on nid007524: devices ['0', '1']",
        f"[{iso(T0)}] start long-32b-aaaaaaaaaa on ['0'] for 57600s",
        f"[{iso(T0)}] start long-control-14b-bbbbbbbbbb on ['1'] for 57600s",
        f"[{iso(T0 + 6 * 3600)}] failed   long-32b-aaaaaaaaaa after 21600s",
        f"[{iso(T0 + 6 * 3600)}] done     long-control-14b-bbbbbbbbbb after 21600s",
    ]
    (queue / "logs" / f"eullm-campaign-{job}.out").write_text("\n".join(out) + "\n")
    with open(queue / "results" / f"{job}.node.jsonl", "w", encoding="utf-8") as f:
        for i in range(0, 6 * 3600 + 1, 60):
            t = T0 + i
            # Device 0 climbs 2 GiB an hour from 52 and its server dies at the
            # top, 5 h in; device 1 holds 27 GiB throughout.
            v0 = 0.0 if i > 5 * 3600 else 52 + 2 * i / 3600
            d = {"0": {"used": int(v0 * GIB), "total": 64 * GIB, "use": 99},
                 "1": {"used": int(27 * GIB), "total": 64 * GIB, "use": 99}}
            f.write(json.dumps({"t": t, "d": d}) + "\n")


def test_a_climbing_server_that_died_and_a_flat_one(tmp_path):
    write_job(tmp_path, "22973477")
    points = vram_slopes.points_of(str(tmp_path / "logs" / "eullm-campaign-22973477.out"))
    assert points["long-32b-aaaaaaaaaa"]["devices"] == ["0"]
    assert points["long-32b-aaaaaaaaaa"]["state"] == "failed"
    samples = vram_slopes.samples_of(str(tmp_path / "results" / "22973477.node.jsonl"))
    rows = {r[0]: r for r in vram_slopes.report(points, samples, settle_s=900, dead_gib=1.0)}

    _, dev, hours, first, peak, last, slope, state = rows["long-32b-aaaaaaaaaa"]
    assert dev == "0"
    assert abs(slope - 2.0) < 0.01
    assert first > 52.4 and peak == last and last > 61.9
    assert "emptied after 5.0 h" in state
    assert hours < 5

    _, _, _, first, peak, last, slope, state = rows["long-control-14b-bbbbbbbbbb"]
    assert first == peak == last == 27.0
    assert abs(slope) < 1e-9
    assert state == "done"


def test_a_server_stopped_at_a_normal_end_is_not_reported_as_emptied(tmp_path):
    """The runner stops the servers before the END line is printed, so a
    clean point's device is already empty at its end timestamp. The trailing
    zero is the normal stop, not a death."""
    write_job(tmp_path, "22973477")
    node = tmp_path / "results" / "22973477.node.jsonl"
    rows = [json.loads(line) for line in node.read_text().splitlines()]
    for r in rows:
        if r["t"] == T0 + 6 * 3600:
            r["d"]["1"]["used"] = 0
    node.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    points = vram_slopes.points_of(str(tmp_path / "logs" / "eullm-campaign-22973477.out"))
    samples = vram_slopes.samples_of(str(node))
    out = {r[0]: r for r in vram_slopes.report(points, samples, settle_s=900, dead_gib=1.0)}
    _, _, _, first, peak, last, slope, state = out["long-control-14b-bbbbbbbbbb"]
    assert state == "done"
    assert first == peak == last == 27.0
    assert abs(slope) < 1e-9


def test_the_command_line_prints_a_row_a_point_and_names_what_is_missing(tmp_path, capsys):
    write_job(tmp_path, "22973477")
    assert vram_slopes.main(["22973477", "99999999", "--queue", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "long-32b-aaaaaaaaaa" in out and "2.00" in out
    assert "long-control-14b-bbbbbbbbbb" in out
    assert "== job 99999999: missing" in out


def test_another_sites_job_output_is_found(tmp_path, capsys):
    write_job(tmp_path, "2326571")
    logs = tmp_path / "logs"
    (logs / "eullm-campaign-2326571.out").rename(logs / "eullm-gh200-2326571.out")
    (logs / "eullm-summary-2326571.out").write_text("nothing of the runner's here\n")
    assert vram_slopes.job_output(str(tmp_path), "2326571").endswith("eullm-gh200-2326571.out")
    assert vram_slopes.main(["2326571", "--queue", str(tmp_path)]) == 0
    assert "long-32b-aaaaaaaaaa" in capsys.readouterr().out
