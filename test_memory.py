"""Behavioral tests for scope, persisted payloads, retrieval and SDK calls."""

import asyncio
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import HTTPServer
from unittest.mock import create_autospec

from PIL import Image

from app import SCOPE, make_handler, seed
from memory import DESCRIPTION, LocalMemory, describe_vector, evidence_image, validate_image


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.store = LocalMemory(self.path)
        self.data = evidence_image()
        self.image_id = self.store.add_image(self.data, DESCRIPTION, metadata={"case": "204"}, user_id="alice", agent_id="support")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_standalone_payload_roundtrip(self):
        self.assertIsNone(self.store.list_images(user_id="alice")[0]["bytes"])
        self.assertEqual(self.store.list_images(image_id=self.image_id, include_bytes=True, user_id="alice")[0]["bytes"], self.data)

    def test_bytes_survive_database_reopen(self):
        other = LocalMemory(self.path)
        try:
            self.assertEqual(other.list_images(image_id=self.image_id, include_bytes=True, user_id="alice")[0]["bytes"], self.data)
        finally:
            other.close()

    def test_search_returns_no_bytes(self):
        results = self.store.search("checkout timeout", user_id="alice")
        self.assertEqual(results[0]["id"], self.image_id)
        self.assertIsNone(results[0]["bytes"])

    def test_search_reads_no_payload_column(self):
        statements = []
        self.store.connection.set_trace_callback(statements.append)
        self.store.search("timeout", user_id="alice")
        self.assertFalse(any("payload" in statement.lower() for statement in statements))

    def test_scope_is_required(self):
        for operation in [lambda: self.store.search("timeout"), lambda: self.store.list_images(),
                          lambda: self.store.add_image(self.data, "sample")]:
            with self.assertRaises(ValueError):
                operation()

    def test_scopes_combine_with_and(self):
        self.assertEqual(self.store.search("timeout", user_id="alice", agent_id="wrong"), [])
        self.assertEqual(self.store.list_images(user_id="other"), [])
        self.assertEqual(self.store.list_images(image_id=self.image_id, include_bytes=True, user_id="other"), [])

    def test_scope_sql_is_parameterized(self):
        self.assertEqual(self.store.search("timeout", user_id="alice' OR 1=1 --"), [])

    def test_hydration_requires_image_id(self):
        with self.assertRaises(ValueError):
            self.store.list_images(include_bytes=True, user_id="alice")

    def test_metadata_filters_before_limit(self):
        self.assertEqual(self.store.search("timeout", metadata_filter={"case": "missing"}, user_id="alice"), [])
        self.assertEqual(len(self.store.search("timeout", metadata_filter={"case": "204"}, max_results=1, user_id="alice")), 1)

    def test_update_reindexes_without_changing_bytes(self):
        self.store.update_description(self.image_id, "ocean sailing vessel", user_id="alice")
        self.assertEqual(self.store.search("ocean sailing vessel", user_id="alice")[0]["score"], 1.0)
        self.assertEqual(self.store.list_images(image_id=self.image_id, include_bytes=True, user_id="alice")[0]["bytes"], self.data)

    def test_update_requires_matching_scope(self):
        with self.assertRaises(KeyError):
            self.store.update_description(self.image_id, "changed", user_id="other")

    def test_invalid_formats_and_mime_mismatch(self):
        for data, mime in [(b"garbage", "image/png"), (self.data, "image/jpeg"), (self.data, "text/plain"), (b"", "image/png")]:
            with self.assertRaises((ValueError, OSError)):
                validate_image(data, mime)

    def test_animated_image_rejected(self):
        out = io.BytesIO()
        frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
        frames[0].save(out, format="PNG", save_all=True, append_images=frames[1:])
        with self.assertRaises(ValueError):
            validate_image(out.getvalue(), "image/png")

    def test_jpeg_and_webp_supported(self):
        for format_name, mime in [("JPEG", "image/jpeg"), ("WEBP", "image/webp")]:
            out = io.BytesIO()
            Image.new("RGB", (2, 2), "red").save(out, format=format_name)
            validate_image(out.getvalue(), mime)

    def test_bad_limits_and_empty_text(self):
        for limit in [0, 101, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                self.store.search("x", max_results=limit, user_id="alice")
        with self.assertRaises(ValueError):
            self.store.search(" ", user_id="alice")
        with self.assertRaises(ValueError):
            self.store.add_image(self.data, "", user_id="alice")

    def test_vectors_are_normalized_and_deterministic(self):
        first = describe_vector("checkout timeout checkout")
        self.assertEqual(first, describe_vector("CHECKOUT timeout checkout"))
        self.assertAlmostEqual(sum(value * value for value in first), 1)
        self.assertEqual(sum(describe_vector(" ")), 0)

    def test_ordered_message_and_selective_hydration(self):
        message_id = self.store.add_message([
            {"type": "text", "text": "before"},
            {"type": "image", "bytes": self.data, "mime_type": "image/png", "description": DESCRIPTION},
            {"type": "text", "text": "after"},
        ], user_id="alice", thread_id="support-thread")
        parts = self.store.get_message(message_id, user_id="alice", thread_id="support-thread")
        self.assertEqual([part["type"] for part in parts], ["text", "image", "text"])
        self.assertIsNone(parts[1]["bytes"])
        hydrated = self.store.get_message(message_id, included_image_ids=[parts[1]["id"]], user_id="alice")
        self.assertEqual(hydrated[1]["bytes"], self.data)
        self.assertEqual(len(self.store.list_images(user_id="alice")), 1)
        self.assertEqual(len(self.store.search("checkout", user_id="alice")), 2)
        with self.assertRaises(KeyError):
            self.store.get_message(message_id, user_id="other")

    def test_invalid_message_is_atomic(self):
        before = self.store.connection.execute("SELECT COUNT(*) FROM image").fetchone()[0]
        with self.assertRaises(ValueError):
            self.store.add_message([
                {"type": "image", "bytes": self.data, "mime_type": "image/png", "description": DESCRIPTION},
                {"type": "unknown"},
            ], user_id="alice", thread_id="support-thread")
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM image").fetchone()[0], before)
        self.assertEqual(self.store.connection.execute("SELECT COUNT(*) FROM message").fetchone()[0], 0)

    def test_loopback_http_routes(self):
        seed(self.store)
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/?q=checkout") as response:
                content = response.read().decode()
                self.assertIn("zero image bytes", content)
                self.assertNotIn("<img ", content)
            image_id = self.store.list_images(**SCOPE)[0]["id"]
            with urlopen(base + "/image?id=" + image_id) as response:
                self.assertEqual(response.read(), self.data)
                self.assertEqual(response.headers["Content-Type"], "image/png")
            with urlopen(base + "/message") as response:
                message = json.load(response)
                self.assertEqual([part["type"] for part in message], ["text", "image", "text"])
                self.assertIsNone(message[1]["bytes"])
            for path in ["/image?id=missing", "/missing"]:
                with self.assertRaises(HTTPError) as result:
                    urlopen(base + path)
                self.assertEqual(result.exception.code, 404)
            with self.assertRaises(HTTPError) as result:
                urlopen(base + "/image")
            self.assertEqual(result.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_message_endpoint_ignores_foreign_scope_with_same_thread(self):
        self.store.add_message([{"type": "text", "text": "Foreign private message"}],
                               user_id="foreign-user", agent_id=SCOPE["agent_id"], thread_id="support-thread")
        seed(self.store)
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/message") as response:
                self.assertEqual(response.status, 200)
                parts = json.load(response)
            self.assertEqual([part["type"] for part in parts], ["text", "image", "text"])
            self.assertEqual(parts[0]["text"], "Please retain this checkout support evidence.")
            self.assertNotIn("Foreign private message", json.dumps(parts))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_seed_recovers_missing_message_without_duplicates(self):
        image_id = self.store.add_image(self.data, DESCRIPTION, **SCOPE)
        self.assertEqual(seed(self.store), image_id)
        self.assertEqual(seed(self.store), image_id)
        count = self.store.connection.execute(
            "SELECT COUNT(*) FROM message WHERE thread_id=? AND user_id=? AND agent_id=?",
            ("support-thread", SCOPE["user_id"], SCOPE["agent_id"]),
        ).fetchone()[0]
        self.assertEqual(count, 1)


class OracleContractTests(unittest.TestCase):
    def test_real_sdk_autospec_workflow(self):
        try:
            from oracleagentmemory.core import OracleAgentMemory, OracleThread
            from oracleagentmemory.apis.message import ImageContent, ImageMimeType
            from oracleagentmemory.apis.records import ImageRecord
        except ImportError:
            self.skipTest("Install requirements-oracle.txt to run native SDK contract test")
        from native import exercise
        client = create_autospec(OracleAgentMemory, instance=True)
        thread = create_autospec(OracleThread, instance=True)
        client.create_thread.return_value = thread
        data = evidence_image()
        client.add_image_async.return_value = "evidence-contract"
        client.list_images_async.side_effect = [
            [ImageRecord("evidence-contract", mime_type=ImageMimeType.PNG)],
            [ImageRecord("evidence-contract", bytes=data, mime_type=ImageMimeType.PNG)],
        ]
        client.search_async.return_value = [SimpleNamespace(record=ImageRecord("evidence-contract", mime_type=ImageMimeType.PNG))]
        thread.add_messages_async.return_value = ["message-contract"]
        part = ImageContent(None, ImageMimeType.PNG, DESCRIPTION, id="attached-image")
        hydrated_part = SimpleNamespace(type="image", id="attached-image", bytes=data)
        thread.get_message_async.side_effect = [SimpleNamespace(content=[part]), SimpleNamespace(content=[hydrated_part])]
        report = asyncio.run(exercise(client, data, user_id="synthetic-user", agent_id="support-demo", run_id="contract"))
        self.assertTrue(report["default_bytes_omitted"])
        self.assertEqual(report["hydrated_byte_count"], len(data))
        thread.get_message_async.assert_awaited_with("message-contract", included_image_ids=["attached-image"])
        call = client.add_image_async.call_args.kwargs
        self.assertEqual(call["user_id"], "synthetic-user")
        self.assertEqual(call["agent_id"], "support-demo")
        self.assertEqual(call["description"], DESCRIPTION)
        self.assertEqual(client.search_async.call_args.kwargs["record_types"], ["image"])


if __name__ == "__main__":
    unittest.main()
