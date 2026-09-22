"""Consumer → SEIF → mock LLM → SEIF → consumer. Fictional data only."""
import os
import uuid

import httpx

headers = {}
if os.getenv("SEIF_CLIENT_SYSTEM"):
    headers = {"X-System-ID": os.environ["SEIF_CLIENT_SYSTEM"], "X-API-Key": os.environ["SEIF_CLIENT_KEY"]}
with httpx.Client(base_url=os.getenv("SEIF_URL", "http://127.0.0.1:8765"), headers=headers) as client:
    payload_id = uuid.uuid4().hex
    protected = client.post("/v1/mask", json={"payload": "Напиши письмо: клиент Иванов Иван Иванович, email demo@example.invalid.",
                                            "payload_id": payload_id, "mode": "token"})
    protected.raise_for_status()
    # Replace this local function with your private LLM client; only masked text goes in.
    llm_reply = "Черновик ответа: " + protected.json()["result"]
    restored = client.post("/v1/unmask", json={"payload": llm_reply, "payload_id": payload_id})
    restored.raise_for_status()
    print(restored.json()["result"])
