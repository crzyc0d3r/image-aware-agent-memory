"""Local image-memory demonstration with explicit scope and lazy byte reads."""

import hashlib
import io
import json
import math
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw


def describe_vector(text):
    """Deterministic token hashing for an offline demo; no learned semantics."""
    values = [0.0] * 128
    for token in re.findall(r"\w+", text.lower()):
        slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], "big") % 128
        values[slot] += 1
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def evidence_image():
    """Draw synthetic support evidence; no customer image or remote model."""
    picture = Image.new("RGB", (900, 430), "#f4f7fa")
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 0, 900, 72), fill="#162338")
    draw.text((30, 25), "CHECKOUT SUPPORT EVIDENCE / SYNTHETIC", fill="white", font_size=22)
    for y, line in enumerate([
        "Case: CASE-204 / Region: eu-frankfurt-1",
        "Signal: checkout timeout spikes / Severity: high",
        "Next step: inspect gateway retries and inventory cache",
    ]):
        draw.text((30, 105 + y * 45), line, fill="#162338", font_size=20)
    draw.line([(40, 360), (180, 338), (300, 350), (450, 285), (590, 320), (760, 245), (860, 270)], fill="#cf3e36", width=7)
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


DESCRIPTION = "Synthetic support screenshot: checkout timeout spikes, high severity, eu-frankfurt-1. Inspect gateway retries and inventory cache."


def validate_image(data, mime_type):
    if not isinstance(data, bytes) or not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Image must contain 1 byte to 10 MiB")
    formats = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
    if mime_type not in formats:
        raise ValueError("Use PNG, JPEG or WebP")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != formats[mime_type] or getattr(image, "is_animated", False):
            raise ValueError("Decoded format mismatch or animated image")
        image.verify()


def scope_values(user_id=None, agent_id=None, thread_id=None):
    values = {"user_id": user_id, "agent_id": agent_id, "thread_id": thread_id}
    if not any(values.values()):
        raise ValueError("At least one nonempty user, agent or thread scope is required")
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in values.values()):
        raise ValueError("Scope identifiers must be nonempty strings")
    return {key: value for key, value in values.items() if value is not None}


class LocalMemory:
    """SQLite model of the workflow, independent of the Oracle implementation."""

    def __init__(self, database):
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS image (
                id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT, thread_id TEXT,
                message_id TEXT, description TEXT NOT NULL, metadata TEXT NOT NULL,
                mime_type TEXT NOT NULL, vector TEXT NOT NULL, payload BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS message (
                id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT, thread_id TEXT,
                parts TEXT NOT NULL
            );
        """)

    def close(self):
        self.connection.close()

    def add_image(self, data, description, *, mime_type="image/png", metadata=None,
                  user_id=None, agent_id=None, thread_id=None):
        scope_values(user_id, agent_id, thread_id)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("A caller-provided description is required")
        validate_image(data, mime_type)
        metadata_json = json.dumps(metadata or {}, allow_nan=False)
        image_id = uuid4().hex
        with self.connection:
            self.connection.execute("INSERT INTO image VALUES (?,?,?,?,?,?,?,?,?,?)", (
                image_id, user_id, agent_id, thread_id, None, description,
                metadata_json, mime_type, json.dumps(describe_vector(description)), data,
            ))
        return image_id

    def _rows(self, *, image_id=None, include_bytes=False, user_id=None, agent_id=None,
              thread_id=None, standalone_only=False):
        scope = scope_values(user_id, agent_id, thread_id)
        if include_bytes and not image_id:
            raise ValueError("Hydration requires an exact image ID")
        clauses = [key + " = ?" for key in scope]
        parameters = list(scope.values())
        if image_id is not None:
            clauses.append("id = ?")
            parameters.append(image_id)
        if standalone_only:
            clauses.append("message_id IS NULL")
        fields = "id,user_id,agent_id,thread_id,message_id,description,metadata,mime_type,vector"
        if include_bytes:
            fields += ",payload"
        rows = self.connection.execute("SELECT " + fields + " FROM image WHERE " + " AND ".join(clauses), parameters)
        return list(rows)

    @staticmethod
    def _record(row):
        value = dict(row)
        value["metadata"] = json.loads(value["metadata"])
        value.pop("vector", None)
        value["bytes"] = value.pop("payload", None)
        return value

    def list_images(self, *, include_bytes=False, image_id=None, **scope):
        return [self._record(row) for row in self._rows(image_id=image_id, include_bytes=include_bytes,
                                                       standalone_only=True, **scope)]

    def search(self, query, *, metadata_filter=None, max_results=5, **scope):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search text cannot be empty")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        query_vector = describe_vector(query)
        results = []
        for row in self._rows(**scope):
            record = self._record(row)
            if any(record["metadata"].get(key) != value for key, value in (metadata_filter or {}).items()):
                continue
            score = sum(a * b for a, b in zip(query_vector, json.loads(row["vector"])))
            if score > 0:
                results.append({**record, "score": round(score, 6)})
        return sorted(results, key=lambda record: (-record["score"], record["id"]))[:max_results]

    def update_description(self, image_id, description, **scope):
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Description cannot be empty")
        rows = self._rows(image_id=image_id, standalone_only=True, **scope)
        if not rows:
            raise KeyError("Scoped standalone image not found")
        with self.connection:
            self.connection.execute("UPDATE image SET description=?,vector=? WHERE id=?", (
                description, json.dumps(describe_vector(description)), image_id))

    def add_message(self, parts, **scope):
        values = scope_values(**scope)
        if not values.get("thread_id"):
            raise ValueError("Messages require a thread ID")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Message requires ordered content parts")
        validated = []
        for part in parts:
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                validated.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image":
                validate_image(part["bytes"], part["mime_type"])
                if not isinstance(part.get("description"), str) or not part["description"].strip():
                    raise ValueError("Image descriptions are required")
                validated.append(dict(part))
            else:
                raise ValueError("Use text or image parts")
        message_id = uuid4().hex
        stored = []
        with self.connection:
            for part in validated:
                if part["type"] == "text":
                    stored.append(part)
                else:
                    image_id = uuid4().hex
                    self.connection.execute("INSERT INTO image VALUES (?,?,?,?,?,?,?,?,?,?)", (
                        image_id, values.get("user_id"), values.get("agent_id"), values["thread_id"],
                        message_id, part["description"], "{}", part["mime_type"],
                        json.dumps(describe_vector(part["description"])), part["bytes"],
                    ))
                    stored.append({"type": "image", "image_id": image_id})
            self.connection.execute("INSERT INTO message VALUES (?,?,?,?,?)", (
                message_id, values.get("user_id"), values.get("agent_id"), values["thread_id"], json.dumps(stored)))
        return message_id

    def get_message(self, message_id, *, included_image_ids=(), **scope):
        values = scope_values(**scope)
        row = self.connection.execute("SELECT parts FROM message WHERE id=? AND " + " AND ".join(
            key + "=?" for key in values), [message_id, *values.values()]).fetchone()
        if row is None:
            raise KeyError("Scoped message not found")
        output = []
        for part in json.loads(row["parts"]):
            if part["type"] == "text":
                output.append(part)
            else:
                image_id = part["image_id"]
                image_row = self._rows(image_id=image_id, include_bytes=image_id in included_image_ids, **scope)[0]
                output.append({"type": "image", **self._record(image_row)})
        return output
