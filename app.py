"""Loopback-only local demo: search descriptions, then request the PNG."""

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

from memory import DESCRIPTION, LocalMemory, evidence_image


SCOPE = {"user_id": "synthetic-user", "agent_id": "support-demo"}


def seed(store):
    records = store.list_images(**SCOPE)
    if records:
        image_id = records[0]["id"]
    else:
        image_id = store.add_image(evidence_image(), DESCRIPTION, metadata={"region": "eu-frankfurt-1", "case_id": "CASE-204"}, **SCOPE)
    existing_message = store.connection.execute(
        "SELECT id FROM message WHERE thread_id=? AND user_id=? AND agent_id=?",
        ("support-thread", SCOPE["user_id"], SCOPE["agent_id"]),
    ).fetchone()
    if existing_message is None:
        store.add_message([
            {"type": "text", "text": "Please retain this checkout support evidence."},
            {"type": "image", "bytes": evidence_image(), "mime_type": "image/png", "description": DESCRIPTION},
            {"type": "text", "text": "Inspect the gateway retries next."},
        ], thread_id="support-thread", **SCOPE)
    return image_id


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, mime_type, body):
            self.send_response(code)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            if url.path == "/favicon.ico":
                return self.send(204, "image/x-icon", b"")
            if url.path == "/image":
                image_id = params.get("id", [""])[0]
                if not image_id:
                    return self.send(400, "text/plain", b"An exact image ID is required")
                records = store.list_images(image_id=image_id, include_bytes=True, **SCOPE)
                if not records:
                    return self.send(404, "text/plain", b"Scoped image not found")
                return self.send(200, records[0]["mime_type"], records[0]["bytes"])
            if url.path == "/message":
                row = store.connection.execute(
                    "SELECT id FROM message WHERE thread_id=? AND user_id=? AND agent_id=?",
                    ("support-thread", SCOPE["user_id"], SCOPE["agent_id"]),
                ).fetchone()
                parts = store.get_message(row[0], thread_id="support-thread", **SCOPE) if row else []
                return self.send(200, "application/json", json.dumps(parts).encode())
            if url.path != "/":
                return self.send(404, "text/plain", b"Route not found")
            query = params.get("q", ["checkout timeout"])[0]
            query = query[:500]
            try:
                results = store.search(query, **SCOPE)
            except ValueError:
                results = []
            rows = "".join(
                "<tr><td>" + html.escape(record["description"]) + "</td><td>" + str(record["score"]) + "</td><td>"
                + ("Message attachment" if record["message_id"] else "Standalone image") + "</td><td>"
                + ("<a href='/?" + urlencode({"q": query, "show": record["id"]}) + "'>Load original image</a>" if not record["message_id"] else "Available in ordered message")
                + "</td></tr>" for record in results)
            selected = params.get("show", [""])[0]
            image = ""
            if selected and store.list_images(image_id=selected, **SCOPE):
                image = "<h2>Explicitly loaded original</h2><img width='900' style='max-width:100%' alt='Synthetic support evidence showing checkout timeout spikes' src='/image?" + urlencode({"id": selected}) + "'>"
            page = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
            <title>Image-aware agent memory</title><style>body{font:17px system-ui;color:#172338;background:#f4f7fa;margin:40px auto;max-width:1100px;padding:0 24px}h1{font-size:40px}p{line-height:1.6}input{padding:12px;width:60%;font:inherit}button{padding:12px;font:inherit;background:#174e77;color:white;border:0;border-radius:4px}table{border-collapse:collapse;background:white;width:100%}td,th{text-align:left;padding:15px;border-bottom:1px solid #ccd7e3}a{color:#125686}.note{padding:18px;background:#e2edf7;border-left:4px solid #174e77}code{background:white;padding:3px}</style>
            <h1>Image-aware agent memory</h1><p>Store visual evidence once. Search its description. Load the pixels only when a turn needs them.</p>
            <p class='note'>Offline demonstration: SQLite persistence and deterministic token-hash vectors. The separate Oracle workflow uses Oracle AI Database and configured model embeddings.</p>
            <form action='/'><label for='q'>Find evidence by description</label><p><input id='q' name='q' value='""" + html.escape(query, quote=True) + """'><button>Search memory</button></p></form>
            <p>Search returned """ + str(len(results)) + """ record(s), with <strong>zero image bytes</strong> in result payloads. Scope: synthetic-user / support-demo.</p>
            <table><thead><tr><th>Description</th><th>Similarity</th><th>Source</th><th>Original</th></tr></thead><tbody>""" + rows + """</tbody></table>
            <p><a href='/message'>Inspect ordered text / image / text message</a> (image bytes omitted).</p>""" + image + "<p>All evidence is synthetic. Search scopes narrow retrieval; this local demo has no account authentication and binds only to loopback.</p></html>"
            self.send(200, "text/html; charset=utf-8", page.encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=".local/memory.sqlite3")
    parser.add_argument("--port", type=int, default=8882)
    args = parser.parse_args()
    store = LocalMemory(args.database)
    seed(store)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Image-aware memory preview: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
