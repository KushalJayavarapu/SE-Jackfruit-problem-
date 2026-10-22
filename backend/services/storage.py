"""
Tiny JSON-file-backed index, standing in for a real database for now.
Each index file is a list of record dicts, keyed by "id".

This is intentionally simple — swapping this for a real DB later is a
later backlog item and shouldn't touch the routes or the service
functions that call these helpers.
"""
import json
import os
import threading

_lock = threading.Lock()


def _read(index_path):
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _write(index_path, records):
    with open(index_path, "w") as f:
        json.dump(records, f, indent=2)


def add_record(index_path, record):
    with _lock:
        records = _read(index_path)
        records.append(record)
        _write(index_path, records)
    return record


def update_record(index_path, record_id, updates):
    with _lock:
        records = _read(index_path)
        for r in records:
            if r["id"] == record_id:
                r.update(updates)
                break
        else:
            return None
        _write(index_path, records)
        return next(r for r in records if r["id"] == record_id)


def get_record(index_path, record_id):
    records = _read(index_path)
    for r in records:
        if r["id"] == record_id:
            return r
    return None


def list_records(index_path):
    return _read(index_path)
