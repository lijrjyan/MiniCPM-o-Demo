"""Contract test without a GPU: fake /v1/realtime upstream + a client shaped like the worker.

    python -m sglang_omni_backend.test_backend

The fake upstream enforces what sglang-omni enforces (event_id, contiguous seq and
t_start_ms, up to four frames per unit, JPEG/PNG) and answers like the MiniCPM-o native
duplex session: one ``sglang.unit.done`` per 1 s unit, and a response spanning
units 2-4 (``response.created``, transcript + audio deltas, ``response.done``).
"""

from __future__ import annotations

import asyncio
import array
import base64
import json
import math
import io
import wave
import socket

import aiohttp
from aiohttp import web

from .server import create_app
from .session import BackendConfig

JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 64 + b"\xff\xd9"
SPEAK_UNITS = (2, 3, 4)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeUpstream:
    def __init__(self, grant_image: bool = True, max_per_unit: int = 4) -> None:
        self.grant_image = grant_image
        self.max_per_unit = max_per_unit
        self.wire_events = []
        self.appends: list[tuple[int, float, int]] = []
        self.images: list[float] = []
        self.update: dict | None = None
        self.closed_by_client = False

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        n = 0

        async def send(event: dict) -> None:
            nonlocal n
            n += 1
            await ws.send_str(json.dumps({**event, "event_id": f"evt_{n}"}))

        await send({"type": "session.created", "session": {"id": "sess_fake"}})
        samples = 0
        next_unit = 0
        async for msg in ws:
            event = json.loads(msg.data)
            assert event.get("event_id"), event
            self.wire_events.append(event)
            kind = event["type"]
            if kind == "session.update":
                self.update = event["session"]
                granted = {
                    "native_full_duplex": True,
                    "native_unit_ms": 1000,
                    "input_modalities": ["audio", "image"] if self.grant_image else ["audio"],
                    "input_audio_format": {"type": "audio/pcm", "rate": 16000},
                    "output_audio_format": {"type": "audio/pcm", "rate": 24000},
                    "input_image_format": {"types": ["image/jpeg"], "max_bytes": 524288, "max_per_unit": self.max_per_unit, "max_slice_nums": 9},
                }
                await send({"type": "session.updated", "session": {"sglang": {"granted": granted}}})
            elif kind == "input_audio_buffer.append":
                seq, t_start = event["sglang"]["seq"], event["sglang"]["t_start_ms"]
                pcm = base64.b64decode(event["audio"])
                assert seq == len(self.appends), (seq, len(self.appends))
                assert math.isclose(t_start, samples / 16, abs_tol=0.01), (t_start, samples / 16)
                self.appends.append((seq, t_start, len(pcm) // 2))
                samples += len(pcm) // 2
                while samples >= (next_unit + 1) * 16000:
                    await self.finish_unit(send, next_unit)
                    next_unit += 1
            elif kind == "sglang.input_image.append":
                assert base64.b64decode(event["image"])[:2] == b"\xff\xd8"
                unit = int(event["sglang"]["t_ms"] // 1000)
                assert sum(int(t // 1000) == unit for t in self.images) < self.max_per_unit, "too many frames in one unit"
                self.images.append(event["sglang"]["t_ms"])
            elif kind == "session.close":
                self.closed_by_client = True
                await send({"type": "session.closed", "reason": "client_closed"})
                await ws.close()
        return ws

    async def finish_unit(self, send, unit: int) -> None:
        ext = {"unit_id": f"unit_{unit}", "media_time": {"t_start_ms": unit * 1000.0, "duration_ms": 1000.0}}
        rid = "resp-0"
        if unit == SPEAK_UNITS[0]:
            await send({"type": "response.created", "response": {"id": rid}, "sglang": ext})
        if unit in SPEAK_UNITS:
            tone = array.array("h", [int(8000 * math.sin(i / 10)) for i in range(24000)]).tobytes()
            await send({"type": "response.output_audio_transcript.delta", "response_id": rid, "delta": f"word{unit} ", "sglang": ext})
            await send({"type": "response.output_audio.delta", "response_id": rid, "delta": base64.b64encode(tone).decode(), "sglang": ext})
        if unit == SPEAK_UNITS[-1]:
            await send({"type": "response.done", "response": {"id": rid}, "sglang": ext})
        await send({"type": "sglang.unit.done", "unit_id": f"unit_{unit}", "sglang": ext})


def float_chunk(seconds: float = 1.0) -> str:
    samples = int(16000 * seconds)
    return base64.b64encode(array.array("f", [0.1 * math.sin(i / 5) for i in range(samples)]).tobytes()).decode()


async def run_case(*, grant_image: bool, force_listen_at: int | None, pause_s: float = 0.0, frame_count: int = 1) -> dict:
    upstream = FakeUpstream(grant_image=grant_image)
    up_app = web.Application()
    up_app.router.add_get("/v1/realtime", upstream.handler)
    up_app.router.add_get("/health", lambda request: web.json_response({"status": "healthy"}))
    up_port, port = free_port(), free_port()
    runners = []
    for app, p in ((up_app, up_port), (create_app(BackendConfig(upstream_url=f"ws://127.0.0.1:{up_port}/v1/realtime", silence_grace_ms=300)), port)):
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", p).start()
        runners.append(runner)
    events: list[dict] = []
    try:
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://127.0.0.1:{port}/health") as response:
                assert response.status == 200, await response.text()
            ws = await http.ws_connect(f"ws://127.0.0.1:{port}/backend")
            await ws.send_str(json.dumps({"type": "session.init", "payload": {
                "mode": "full_duplex", "system_prompt": "be brief", "config": {"length_penalty": 1.1, "decode_mode": "sampling", "temperature": 0.6, "text_repetition_penalty": 1.2},
                "ref_audio_base64": float_chunk(0.1)}}))
            created = json.loads((await ws.receive()).data)
            assert created["type"] == "session.created" and created["mode"] == "full_duplex", created
            assert upstream.update["output_modalities"] == ["audio"]
            assert upstream.update["instructions"] == "be brief"
            assert upstream.update["sglang"]["sampling"] == {"greedy": False, "temperature": 0.6, "repetition_penalty": 1.2}
            with wave.open(io.BytesIO(base64.b64decode(upstream.update["sglang"]["reference_audio"]["data"]))) as wav:
                assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, 1600)
            assert created["sglang"]["ignored_init_fields"] == ["config.length_penalty"]
            session_id = created["session_id"]

            async def reader() -> None:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        events.append(json.loads(msg.data))

            reader_task = asyncio.create_task(reader())
            for index in range(7):
                if index == 3 and pause_s:
                    await asyncio.sleep(pause_s)
                payload = {"audio": float_chunk(), "video_frames": [base64.b64encode(JPEG).decode()] * frame_count}
                if force_listen_at is not None and index >= force_listen_at:
                    payload["force_listen"] = True
                await ws.send_str(json.dumps({"type": "input.append", "input": payload}))
                await asyncio.sleep(0.15)
            await asyncio.sleep(0.3)
            async with http.post(f"http://127.0.0.1:{port}/sessions/{session_id}/close", json={"reason": "client_closed"}) as response:
                body = await response.json()
                assert response.status == 200 and body == {"ok": True, "session_id": session_id, "closed": True}, body
            async with http.post(f"http://127.0.0.1:{port}/sessions/{session_id}/close", json={}) as response:
                assert response.status == 404
            await asyncio.wait_for(reader_task, 5)
    finally:
        for runner in runners:
            await runner.cleanup()
    return {"events": events, "upstream": upstream}


async def main() -> None:
    result = await run_case(grant_image=True, force_listen_at=None)
    events, upstream = result["events"], result["upstream"]
    kinds = [e.get("kind") or e["type"] for e in events]
    audio = [e for e in events if e.get("kind") == "audio"]
    text = "".join(e.get("text", "") for e in events if e.get("kind") == "text")
    assert all("server_send_ts" in e for e in events)
    assert len(audio) == 3 and len(base64.b64decode(audio[0]["audio"])) == 24000 * 4, len(audio)
    assert text == "word2 word3 word4 ", text
    assert kinds[:2] == ["listen", "listen"] and kinds[-1] == "session.closed", kinds
    assert "first_audio_ms" in audio[0]["metrics"] and "wall_clock_ms" in audio[0]["metrics"]
    turn_end = [e for e in events if e.get("kind") == "listen" and e.get("response_id")]
    assert len(turn_end) == 1, turn_end
    assert upstream.images == [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0], upstream.images
    assert sum(n for _, _, n in upstream.appends) == 7 * 16000 and upstream.closed_by_client
    print("case 1 (speak + frames + close):", " ".join(k[:6] for k in kinds))

    result = await run_case(grant_image=False, force_listen_at=3)
    events, upstream = result["events"], result["upstream"]
    audio = [e for e in events if e.get("kind") == "audio"]
    assert len(audio) == 1 and not upstream.images, (len(audio), upstream.images)  # units 3, 4 muted
    assert events[-1]["sglang"]["breaks"] == 1 and events[-1]["sglang"]["audio_muted_s"] == 2.0, events[-1]
    print("case 2 (force_listen mutes, image not granted):", events[-1]["sglang"]["audio_muted_s"], "s muted")

    result = await run_case(grant_image=True, force_listen_at=None, pause_s=2.0)
    upstream = result["upstream"]
    filled = result["events"][-1]["sglang"]["silence_filled_s"]
    starts = [t for _, t, _ in upstream.appends]
    assert filled >= 1.0 and float(filled).is_integer(), filled  # whole units only
    assert all(t % 1000 == 0 for t in upstream.images), upstream.images  # still unit-aligned
    print("case 3 (pause -> silence fill):", filled, "s filled, timeline", starts[-1] + 40, "ms")
    result = await run_case(grant_image=True, force_listen_at=None, frame_count=4)
    assert result["upstream"].images == [float(unit * 1000) for unit in range(7) for _ in range(4)]
    assert result["events"][-1]["sglang"]["frames_forwarded"] == 28
    wire = result["upstream"].wire_events
    first_audio = next(i for i, event in enumerate(wire) if event["type"] == "input_audio_buffer.append")
    assert [event["type"] for event in wire[first_audio-4:first_audio]] == ["sglang.input_image.append"] * 4
    print("case 4 (four frames per unit): 28 frames forwarded before their audio")
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
