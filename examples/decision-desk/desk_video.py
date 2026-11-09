#!/usr/bin/env python3
"""The Decision Desk as a video to post.

Runs the built-in inbox through your EuLLM server, the same way the page
does, keeps every step of every email with the moment it happened, and
renders it with the same page, frame by frame, in a headless browser. Three
files come out, ready to upload:

    desk-wide.mp4     1920x1080, for X, LinkedIn, YouTube
    desk-tall.mp4     1080x1920, for Instagram Reels (and X, vertical)
    desk-readme.mp4   1280x720 and small, for a GitHub README

Everything on screen is the run as it happened: the passages found, the
probabilities, the rule, every token of every draft at the moment it was
written, the decision times. Only the pauses between emails are chosen, for
reading (`--hold`).

Besides the Python standard library it needs Playwright, a browser for it,
and the ffmpeg program:

    pip install playwright
    python -m playwright install chromium
    sudo apt install ffmpeg        # or your system's own package

Then, with `eullm serve` running (the desk loads the three models itself):

    python examples/decision-desk/desk_video.py \\
        --decision-model jev-style-2b-decision-v3-gguf-q4_k_m \\
        --hardware "RTX 5070 Ti"

The run is saved too (`desk-run.json`): `--from desk-run.json` renders it
again, with another `--hold` or `--caption`, without the models.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import desk  # the desk and its page, next to this file

FPS = 30
LAYOUTS = {"wide": (1920, 1080), "tall": (1080, 1920)}
INTRO = 1.5  # seconds of the page before the first email
OUTRO = 4.0  # seconds the last email and the totals stay


def record(d, samples):
    """Every email through the desk: each step with the seconds since its
    email started."""
    emails = []
    for n, sample in enumerate(samples):
        events = []
        started = time.perf_counter()

        def keep(event, started=started, events=events):
            events.append({"dt": time.perf_counter() - started, "event": event})

        d.process(sample, keep)
        failed = [e["event"] for e in events if e["event"]["stage"] == "error"]
        if failed:
            raise SystemExit(f"{sample['id']}: {failed[0]['message']}")
        rule = next(e["event"] for e in events if e["event"]["stage"] == "rule")
        print(
            f"  {n + 1:>2}/{len(samples)} {sample['id']:<12} {rule['action']:<10} "
            f"{events[-1]['dt']:.1f} s",
            flush=True,
        )
        emails.append({"index": n, "events": events})
    return emails


def timeline(emails, hold):
    """One clock for the whole video: each email starts `hold` seconds after
    the one before it ended, the first after `INTRO`. Returns every step
    with its second, and the video's length."""
    steps, t = [], INTRO
    for email in emails:
        for e in email["events"]:
            steps.append(
                {
                    "t": round(t + e["dt"], 4),
                    "index": email["index"],
                    "event": e["event"],
                }
            )
        t += email["events"][-1]["dt"] + hold
    return steps, t - hold + OUTRO


def warm(d):
    """The models loaded before the first email, as the page does it, so
    that no load is in the video."""
    try:
        d.client.call("/api/version", timeout=5)
    except desk.EuLLMError as e:
        raise SystemExit(f"{e}\nStart it in another terminal: eullm serve") from None
    # Desk.warm_up waits for the server forever, as a page should; here it
    # runs beside a loop that says what is loading.
    worker = threading.Thread(target=d.warm_up, daemon=True)
    worker.start()
    said = None
    while worker.is_alive():
        now = {k: d.warmup[k] for k in ("index", "decision", "chat")}
        if now != said:
            loading = [k for k, v in now.items() if v in ("waiting", "loading")]
            print(f"loading the models: {', '.join(loading) or 'done'}", flush=True)
            said = now
        worker.join(1.0)
    if "failed" in (d.warmup["decision"], d.warmup["chat"]):
        raise SystemExit(f"a model did not load: {d.warmup['error']}")
    status = d.status()
    missing = [k for k, m in status["models"].items() if not m["loaded"]]
    if status["error"] or missing:
        raise SystemExit(
            status["error"]
            or f"not loaded: {', '.join(missing)} — is the decision model named "
            "(--decision-model) or loaded by the server?"
        )
    return status


def encoder(outputs):
    """ffmpeg, reading PNG frames on stdin, writing each (path, width, crf)."""
    command = [shutil.which("ffmpeg"), "-loglevel", "error", "-y"]
    command += ["-f", "image2pipe", "-framerate", str(FPS), "-c:v", "png", "-i", "-"]
    for path, width, crf in outputs:
        if width:
            command += ["-vf", f"scale={width}:-2:flags=lanczos"]
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf)]
        command += ["-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart"]
        command.append(str(path))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render(browser, run, layout, outputs):
    """Draw every video frame with the page and pipe it to ffmpeg. A frame
    is drawn again only when a step changed it."""
    width, height = LAYOUTS[layout]
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(desk.PAGE.as_uri() + "?video=" + layout)
    page.evaluate("run => deskReplay.load(run)", run)
    ffmpeg = encoder(outputs)
    frames = round(run["seconds"] * FPS)
    image = None
    started = time.perf_counter()
    try:
        for n in range(frames):
            changed = page.evaluate("t => deskReplay.at(t)", n / FPS)
            if changed or image is None:
                image = page.screenshot(type="png")
            ffmpeg.stdin.write(image)
            if n and n % (FPS * 10) == 0:
                print(
                    f"  {layout}: {n // FPS} of {frames // FPS} s "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
    finally:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        page.close()
    if ffmpeg.returncode:
        raise SystemExit(f"ffmpeg failed on {layout} (exit {ffmpeg.returncode})")


def main():
    parser = desk.add_desk_options(
        argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    )
    parser.add_argument(
        "--from",
        dest="source",
        default=None,
        help="render this saved run instead of running the inbox",
    )
    parser.add_argument(
        "--save", default="desk-run.json", help="where the run is saved"
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=2.5,
        help="seconds each email stays once handled (default 2.5)",
    )
    parser.add_argument(
        "--caption",
        default="",
        help="a line under the title instead of the page's own, "
        'e.g. "Jev-Style 2B decides, Gemma 4 E4B writes"',
    )
    parser.add_argument(
        "--hardware",
        default="",
        help='what the server runs on, for the "all local" badge, e.g. "RTX 5070 Ti"',
    )
    parser.add_argument(
        "--layouts",
        default="wide,tall",
        help="which videos: wide (X, README) and/or tall (Instagram)",
    )
    parser.add_argument("--out-dir", default=".", help="where the videos go")
    parser.add_argument(
        "--browser",
        default=None,
        help="a Chrome or Chromium to draw with, instead of Playwright's own",
    )
    args = desk.resolve(parser.parse_args())
    layouts = [name for name in args.layouts.split(",") if name]
    for name in layouts:
        if name not in LAYOUTS:
            parser.error(f"unknown layout {name!r}: wide or tall")

    # Everything the rendering needs is checked before the inbox is run.
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found: install it (sudo apt install ffmpeg)")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright not found: pip install playwright, then "
            "python -m playwright install chromium"
        ) from None

    if args.source:
        with open(args.source, encoding="utf-8") as f:
            saved = json.load(f)
    else:
        d = desk.Desk(args)
        status = warm(d)
        print(f"running the inbox on {args.url}", flush=True)
        try:
            emails = record(d, desk.SAMPLES)
        except desk.EuLLMError as e:
            raise SystemExit(str(e)) from None
        saved = {"status": status, "emails": emails}
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False)
        print(f"run saved: {args.save}", flush=True)

    steps, seconds = timeline(saved["emails"], args.hold)
    run = {
        "status": saved["status"],
        "timeline": steps,
        "seconds": seconds,
        "caption": args.caption,
        "hardware": args.hardware,
    }
    print(f"video: {seconds:.0f} s", flush=True)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser)
        try:
            for layout in layouts:
                outputs = [(out / f"desk-{layout}.mp4", None, 18)]
                if layout == "wide":
                    outputs.append((out / "desk-readme.mp4", 1280, 30))
                render(browser, run, layout, outputs)
                for path, _, _ in outputs:
                    size = path.stat().st_size / 1e6
                    print(f"{path}: {size:.1f} MB", flush=True)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
