"""Outbox durability and drain semantics.

The bug this exists for: entries were added before a send but never removed
after a successful one, so every session was replayed on the next launch and
relied on the server's dedupe to no-op it. And a permanently-rejected entry
halted the drain forever, blocking every later session from ever syncing.
"""
from __future__ import annotations

import outbox


def _entry(task_id, date="2026-08-02"):
    return {"date": date, "task": {"id": task_id, "text": task_id, "done": True}}


def test_add_and_read_roundtrip(isolated_state):
    outbox.add(_entry("a"))
    outbox.add(_entry("b"))
    assert [e["task"]["id"] for e in outbox.read_all()] == ["a", "b"]


def test_read_all_on_missing_file(isolated_state):
    assert outbox.read_all() == []


def test_remove_is_by_task_id(isolated_state):
    for i in "abc":
        outbox.add(_entry(i))
    outbox.remove("b")
    assert [e["task"]["id"] for e in outbox.read_all()] == ["a", "c"]


def test_drain_removes_each_entry_after_a_successful_send(isolated_state):
    for i in "abc":
        outbox.add(_entry(i))
    sent = []
    outbox.drain(lambda date, task: sent.append(task["id"]))
    assert sent == ["a", "b", "c"]
    assert outbox.read_all() == []


def test_drain_stops_at_a_transient_failure_and_keeps_the_rest(isolated_state):
    """A network outage must not skip ahead and silently drop later entries."""
    for i in "abc":
        outbox.add(_entry(i))
    attempts = []

    def sender(date, task):
        attempts.append(task["id"])
        if task["id"] == "b":
            raise ConnectionError("network down")

    outbox.drain(sender)
    assert attempts == ["a", "b"]
    assert [e["task"]["id"] for e in outbox.read_all()] == ["b", "c"]


def test_drain_drops_permanent_failures_and_continues(isolated_state):
    """One poison entry used to block the queue forever."""
    for i in "abc":
        outbox.add(_entry(i))

    class Rejected(Exception):
        pass

    def sender(date, task):
        if task["id"] == "a":
            raise Rejected("400 bad request")

    outbox.drain(sender, is_permanent=lambda e: isinstance(e, Rejected))
    assert outbox.read_all() == []


def test_drain_resumes_where_it_stopped(isolated_state):
    for i in "ab":
        outbox.add(_entry(i))
    fail = {"on": True}

    def sender(date, task):
        if fail["on"]:
            raise ConnectionError("down")

    outbox.drain(sender)
    assert len(outbox.read_all()) == 2
    fail["on"] = False
    outbox.drain(sender)
    assert outbox.read_all() == []


def test_rewrite_is_atomic_and_leaves_no_temp_file(isolated_state):
    outbox.add(_entry("a"))
    outbox.remove("a")
    assert list(isolated_state.glob("*.tmp")) == []
