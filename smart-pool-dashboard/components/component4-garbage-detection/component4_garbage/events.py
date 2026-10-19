"""Event log: one JSON line per event in events/events.jsonl, a snapshot
JPG per event, and (optional) a MongoDB insert when MONGO_URI is set."""

import json
import time
from datetime import datetime

import cv2

from component4_garbage import config


class EventLogger:
    def __init__(self, events_dir=None):
        self.dir = events_dir or config.EVENTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "events.jsonl"
        self.collection = None
        if config.MONGO_URI:
            try:
                from pymongo import MongoClient
                client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=2000)
                self.collection = client[config.MONGO_DB]["garbage_events"]
            except Exception:  # noqa: BLE001
                self.collection = None

    def log(self, event, frame=None):
        event = dict(event)
        event.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        if frame is not None and config.SAVE_SNAPSHOTS:
            name = f"{int(time.time() * 1000)}_{event.get('type', 'event')}.jpg"
            cv2.imwrite(str(self.dir / name), frame)
            event["snapshot"] = name
        with self.file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        if self.collection is not None:
            try:
                self.collection.insert_one(dict(event))
            except Exception:  # noqa: BLE001
                pass
        return event
