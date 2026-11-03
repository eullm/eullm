"""The campaign's queue: one JSON file per point, one directory per state.

    <root>/todo/      waiting
    <root>/running/   claimed by a job (+ <id>.owner.json: which job, when)
    <root>/done/      measured
    <root>/failed/    failed after max_attempts
    <root>/blocked/   cannot run here as things stand (a model not pulled);
                      `unblock` puts them back once the cause is fixed

A claim is a rename from todo/ to running/. Rename is atomic on one
filesystem — Lustre included — so when two nodes go for the same point one of
them gets ENOENT and moves on: several jobs can drain one campaign at once,
which is how a project behind its calendar runs two nodes instead of one.
"""

from __future__ import annotations

import json
import os
import time

STATES = ("todo", "running", "done", "failed", "blocked")
MAX_ATTEMPTS = 2
# An item in running/ with no owner file yet may be mid-claim on another node;
# only after this long is it treated as abandoned.
ORPHAN_GRACE_S = 600


def _write(path: str, data: dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


class Queue:
    def __init__(self, root: str):
        self.root = root
        for state in STATES:
            os.makedirs(self._dir(state), exist_ok=True)

    def _dir(self, state: str) -> str:
        return os.path.join(self.root, state)

    def _path(self, state: str, pid: str) -> str:
        return os.path.join(self._dir(state), f"{pid}.json")

    def _owner_path(self, pid: str) -> str:
        return os.path.join(self._dir("running"), f"{pid}.owner.json")

    def ids(self, state: str) -> list:
        return sorted(
            n[: -len(".json")]
            for n in os.listdir(self._dir(state))
            if n.endswith(".json") and not n.endswith(".owner.json") and ".tmp." not in n
        )

    def where(self, pid: str):
        for state in STATES:
            if os.path.exists(self._path(state, pid)):
                return state
        return None

    def add(self, point: dict) -> bool:
        """Queue `point` unless it is already known in any state."""
        if self.where(point["id"]) is not None:
            return False
        _write(self._path("todo", point["id"]), point)
        return True

    def load(self, state: str, pid: str) -> dict:
        return _read(self._path(state, pid))

    def todo(self) -> list:
        points = []
        for pid in self.ids("todo"):
            try:
                points.append(self.load("todo", pid))
            except (FileNotFoundError, json.JSONDecodeError):
                continue  # claimed or being written by another node meanwhile
        return points

    def claim(self, pid: str, owner: dict):
        """The point, now ours; None if another job took it first."""
        try:
            os.rename(self._path("todo", pid), self._path("running", pid))
        except FileNotFoundError:
            return None
        _write(self._owner_path(pid), dict(owner, claimed=time.time()))
        return self.load("running", pid)

    def _leave_running(self, pid: str, state: str, point: dict) -> None:
        _write(self._path(state, pid), point)
        for path in (self._path("running", pid), self._owner_path(pid)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def release(self, pid: str) -> None:
        """Back to todo, as if never claimed: the job ended, not the point."""
        self._leave_running(pid, "todo", self.load("running", pid))

    def finish(self, pid: str, state: str, note: str = "", retry: bool = True) -> str:
        """Leave running/ for `state`; a failure is retried until
        MAX_ATTEMPTS, unless `retry` is false (a server death is a finding,
        not a flaky attempt). Returns the state the point ended up in."""
        if state not in ("done", "failed", "blocked"):
            raise ValueError(state)
        point = self.load("running", pid)
        if note:
            point.setdefault("notes", []).append(note)
        if state == "failed":
            point["attempts"] = point.get("attempts", 0) + 1
            if retry and point["attempts"] < MAX_ATTEMPTS:
                state = "todo"
        self._leave_running(pid, state, point)
        return state

    def running(self) -> list:
        out = []
        for pid in self.ids("running"):
            try:
                owner = _read(self._owner_path(pid))
            except (FileNotFoundError, json.JSONDecodeError):
                owner = None
            out.append((pid, owner))
        return out

    def requeue_stale(self, alive_jobs, now: float, max_age_s: float) -> list:
        """Release points whose job is gone. `alive_jobs` is the set of job
        ids still queued or running (None when it cannot be known, and then
        only age decides)."""
        released = []
        for pid, owner in self.running():
            if owner is None:
                try:
                    age = now - os.path.getmtime(self._path("running", pid))
                except FileNotFoundError:
                    continue
                stale = age > ORPHAN_GRACE_S
            elif alive_jobs is not None:
                stale = str(owner.get("job")) not in alive_jobs
            else:
                stale = now - owner.get("claimed", now) > max_age_s
            if stale:
                try:
                    self.release(pid)
                except FileNotFoundError:
                    continue
                released.append(pid)
        return released

    def unblock(self) -> list:
        moved = []
        for pid in self.ids("blocked"):
            point = self.load("blocked", pid)
            _write(self._path("todo", pid), point)
            os.remove(self._path("blocked", pid))
            moved.append(pid)
        return moved

    def retry(self, prefix: str = "", label=None, note=None) -> list:
        """Failed points whose id starts with `prefix` back to todo, with
        their attempts reset and their notes kept: after the cause is fixed.
        `label`: only the points of that engine label; `note`: only those
        whose last note holds that text (the cause that was fixed)."""
        moved = []
        for pid in self.ids("failed"):
            if not pid.startswith(prefix):
                continue
            point = self.load("failed", pid)
            if label is not None and (point.get("engine_label") or "") != label:
                continue
            if note is not None and note not in ((point.get("notes") or [""])[-1] or ""):
                continue
            point["attempts"] = 0
            _write(self._path("todo", pid), point)
            os.remove(self._path("failed", pid))
            moved.append(pid)
        return moved

    def set_aside(self, dest: str, label=None, prefixes=(), round_=None) -> list:
        """Waiting points moved out of the queue into `dest` (a directory of
        their own, beside the states), not deleted: those of engine label
        `label`, of round `round_`, whose id starts with one of `prefixes`,
        each filter only when given. A point a job claims meanwhile stays
        with the job. For a round an engine has made pointless, or points
        no job of the round's length can run (26-10-2026, 03-11-2026)."""
        moved = []
        for pid in self.ids("todo"):
            if prefixes and not any(pid.startswith(p) for p in prefixes):
                continue
            try:
                point = self.load("todo", pid)
            except (OSError, ValueError):
                continue
            if label is not None and (point.get("engine_label") or "") != label:
                continue
            if round_ is not None and point.get("round") != round_:
                continue
            os.makedirs(dest, exist_ok=True)
            try:
                os.replace(self._path("todo", pid), os.path.join(dest, f"{pid}.json"))
            except FileNotFoundError:
                continue  # claimed by a job in between
            moved.append(pid)
        return moved

    def counts(self) -> dict:
        return {state: len(self.ids(state)) for state in STATES}
