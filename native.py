"""Native Oracle workflow. Only the explicit CLI connects to external services."""

import argparse
import asyncio
import json
import os
import re
import sys
from uuid import uuid4

from memory import DESCRIPTION, evidence_image


async def exercise(memory, image_bytes, *, user_id, agent_id, run_id):
    """Use public 26.8 SDK APIs; callable with an autospecced client in tests."""
    from oracleagentmemory.apis.message import ImageMimeType

    scope = {"user_id": user_id, "agent_id": agent_id}
    image_id = await memory.add_image_async(image=image_bytes, description=DESCRIPTION,
        mime_type=ImageMimeType.PNG, image_id="evidence-" + run_id,
        metadata={"run_id": run_id, "region": "eu-frankfurt-1"}, **scope)
    default = await memory.list_images_async(image_id=image_id, include_bytes=False, **scope)
    results = await memory.search_async(query="checkout timeout spikes eu-frankfurt-1",
        record_types=["image"], metadata_filter={"run_id": run_id}, max_results=5, **scope)
    hydrated = await memory.list_images_async(image_id=image_id, include_bytes=True, **scope)
    await memory.update_image_async(image_id=image_id, description=DESCRIPTION + " Retry correlation confirmed.")
    thread = memory.create_thread(thread_id="support-" + run_id, **scope)
    message_ids = await thread.add_messages_async([{"role": "user", "content": [
        {"type": "text", "text": "Please retain this support evidence."},
        {"type": "image", "bytes": image_bytes, "mime_type": "image/png", "description": DESCRIPTION},
    ]}])
    message = await thread.get_message_async(message_ids[0])
    attached = [part for part in message.content if getattr(part, "type", None) == "image"]
    image_ids = [part.id for part in attached]
    loaded_message = await thread.get_message_async(message_ids[0], included_image_ids=image_ids)
    if not default or any(record.bytes is not None for record in default):
        raise RuntimeError("Default image readback did not omit bytes")
    if not hydrated or hydrated[0].bytes != image_bytes:
        raise RuntimeError("Standalone image bytes failed round trip")
    if not any(result.record.id == image_id for result in results):
        raise RuntimeError("Search did not return this run's image")
    loaded_parts = [part for part in loaded_message.content if getattr(part, "type", None) == "image"]
    if not attached or not loaded_parts or loaded_parts[0].bytes != image_bytes:
        raise RuntimeError("Message image failed round trip")
    return {"mode": "oracle", "image_id": image_id, "search_matches": len(results),
            "default_bytes_omitted": True, "hydrated_byte_count": len(hydrated[0].bytes),
            "message_id": message_ids[0], "message_image_count": len(attached)}


def main():
    parser = argparse.ArgumentParser(description="Persist synthetic evidence to a configured Oracle AI Database")
    parser.add_argument("--create-schema", action="store_true", help="Allow creation of the managed memory schema")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Oracle image validation requires Linux/macOS/WSL; use app.py for the Windows offline demo")
    required = ["DB_USER", "DB_PASSWORD", "DB_DSN", "MODEL_PROVIDER_API_KEY", "MEMORY_EMBEDDING_MODEL", "MEMORY_LLM_MODEL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        parser.error("Missing environment settings: " + ", ".join(missing))
    store_id = os.environ.get("MEMORY_STORE_ID", "IMGMEMORY")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", store_id):
        parser.error("MEMORY_STORE_ID must start with a letter and contain at most 16 alphanumeric/underscore characters")

    import oracledb
    from oracleagentmemory.core import OracleAgentMemory, OracleDBMemoryStore, SchemaPolicy, SearchStrategy, MemoryExtractionConfig
    from oracleagentmemory.core.embedders import Embedder
    from oracleagentmemory.core.llms import Llm

    embedder = Embedder(model=os.environ["MEMORY_EMBEDDING_MODEL"], api_key=os.environ["MODEL_PROVIDER_API_KEY"],
                        embedding_dimension=int(os.environ.get("MEMORY_EMBEDDING_DIMENSION", "1536")))
    llm = Llm(model=os.environ["MEMORY_LLM_MODEL"], api_key=os.environ["MODEL_PROVIDER_API_KEY"], supports_vision=False)
    pool_args = {"user": os.environ["DB_USER"], "password": os.environ["DB_PASSWORD"],
                 "dsn": os.environ["DB_DSN"], "min": 1, "max": 2, "increment": 1}
    if os.environ.get("DB_WALLET_LOCATION"):
        pool_args.update(config_dir=os.environ["DB_WALLET_LOCATION"], wallet_location=os.environ["DB_WALLET_LOCATION"])
    if os.environ.get("DB_WALLET_PASSWORD"):
        pool_args["wallet_password"] = os.environ["DB_WALLET_PASSWORD"]
    pool = oracledb.create_pool(**pool_args)
    try:
        store = OracleDBMemoryStore(pool=pool, embedder=embedder, memory_store_id=store_id,
            schema_policy=SchemaPolicy.CREATE_IF_NECESSARY if args.create_schema else SchemaPolicy.REQUIRE_EXISTING,
            search_strategy=SearchStrategy.VECTOR, vector_dim=int(os.environ.get("MEMORY_EMBEDDING_DIMENSION", "1536")))
        memory = OracleAgentMemory(store=store, llm=llm,
            memory_extraction_config=MemoryExtractionConfig(extract_memories=False, enable_context_summary=False))
        report = asyncio.run(exercise(memory, evidence_image(), user_id="synthetic-user",
                                      agent_id="support-demo", run_id=uuid4().hex[:12]))
        with pool.acquire() as connection:
            with connection.cursor() as cursor:
                row = cursor.execute("SELECT document_type, DBMS_LOB.GETLENGTH(blob) FROM "
                                     + store_id.upper() + "_DOCUMENT WHERE record_id=:record_id",
                                     record_id=report["image_id"]).fetchone()
        if row is None or row[0] != "image" or row[1] <= 0:
            raise RuntimeError("Managed DOCUMENT row did not contain image bytes")
        report["document_type"], report["document_byte_length"] = row
        print(json.dumps(report, indent=2))
    finally:
        pool.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Native workflow failed (" + type(error).__name__
              + "). Check database/schema permissions, model configuration and service availability locally.", file=sys.stderr)
        raise SystemExit(1) from None
