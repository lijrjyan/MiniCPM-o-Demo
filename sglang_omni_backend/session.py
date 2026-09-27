"""One backend-protocol session served by one sglang-omni ``/v1/realtime`` session.

Downstream (the demo worker) speaks the backend protocol of docs/backend-protocol/:
``session.init`` -> ``session.created``, ``input.append`` (one float32 16 kHz time
slice + optional JPEG frames) -> ``response.output.delta`` (kind listen/text/audio),
close over HTTP -> ``session.closed``.

Upstream (sglang-omni) speaks ``/v1/realtime``: ``session.update`` ->
``session.updated`` (granted capabilities), ``input_audio_buffer.append`` (80 ms
PCM16 packets with contiguous ``sglang.seq`` / ``sglang.t_start_ms``),
``sglang.input_image.append``, and per-unit output events
(``response.created``, ``response.output_audio.delta``,
``response.output_audio_transcript.delta``, ``response.done``,
``sglang.unit.done``). See BACKEND.md for the full mapping.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
from aiohttp import web

from .audio import INPUT_RATE, OUTPUT_RATE, Packetizer, float32_to_pcm16, float32_to_wav, pcm16_to_float32

log = logging.getLogger("sglang_omni_backend.session")

BACKEND_NAME = "sglang-omni"
TEXT_DELTAS = ("response.output_audio_transcript.delta", "response.output_text.delta")
UPSTREAM_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
SENT_EVENT_MEMORY = 4096


class ProtocolViolation(RuntimeError):
    """Illegal input from the worker; the protocol requires fail-fast (schema §7.4)."""


class UpstreamFailure(RuntimeError):
    """The sglang-omni session failed or closed on its own."""


@dataclass
class BackendConfig:
    upstream_url: str = "ws://127.0.0.1:18260/v1/realtime"
    forward_images: bool = True
    silence_fill: bool = True
    silence_grace_ms: float = 1000.0
    open_timeout_s: float = 20.0
    close_timeout_s: float = 5.0


def _extract_audio_base64(payload: dict[str, Any]) -> Optional[str]:
    # Same aliases as py_backend/server.py::_extract_audio_base64.
    for key in ("audio_base64", "audio_data"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    audio = payload.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, dict):
        value = audio.get("data") or audio.get("base64") or audio.get("audio_base64")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_frames(payload: dict[str, Any]) -> list[str]:
    # Same aliases as py_backend/server.py::_extract_frame_base64_list.
    direct = payload.get("frame_base64_list") or payload.get("video_frames")
    if direct:
        return [frame for frame in direct if isinstance(frame, str) and frame]
    out = []
    for frame in payload.get("frames") or []:
        if isinstance(frame, str):
            out.append(frame)
        elif isinstance(frame, dict) and (frame.get("data") or frame.get("base64")):
            out.append(frame.get("data") or frame.get("base64"))
    return out


def _unit_index(unit_id: Any) -> Optional[int]:
    if isinstance(unit_id, str) and unit_id.startswith("unit_"):
        try:
            return int(unit_id[5:])
        except ValueError:
            return None
    return None


class BackendSession:
    def __init__(self, *, session_id: str, ws: web.WebSocketResponse, config: BackendConfig, registry: Any) -> None:
        self.session_id = session_id
        self.ws = ws
        self.config = config
        self.registry = registry
        self.closed = False
        self._closing_upstream = False
        self._down_lock = asyncio.Lock()
        self._input_lock = asyncio.Lock()
        self._http: Optional[aiohttp.ClientSession] = None
        self._up: Optional[aiohttp.ClientWebSocketResponse] = None
        self._tasks: list[asyncio.Task] = []
        self._fatal_task: Optional[asyncio.Task] = None
        self.upstream_session_id: Optional[str] = None
        self.granted: dict[str, Any] = {}
        self.unit_ms = 1000
        self.image_enabled = False
        self.image_max_bytes = 0
        self.packetizer = Packetizer()
        self._anchor_wall: Optional[float] = None  # wall time of the last real chunk; None before the first
        self._anchor_ms = 0.0
        self._unit_input_done: dict[int, float] = {}
        self._unit_frame_counts: dict[int, int] = {}
        self.image_max_per_unit = 1
        self.max_slice_nums = 1
        self._units_emitted: set[int] = set()
        self._sent_kinds: "OrderedDict[str, str]" = OrderedDict()
        self.current_response: Optional[str] = None
        self.response_active = False
        self.force_listen = False
        self.muted: set[str] = set()
        self._first_audio_seen: set[str] = set()
        self.ignored_init_fields: list[str] = []
        self.stats: dict[str, Any] = {
            "pushes": 0,
            "input_audio_s": 0.0,
            "silence_filled_s": 0.0,
            "frames_forwarded": 0,
            "frames_skipped": 0,
            "units": 0,
            "responses": 0,
            "audio_out_s": 0.0,
            "audio_muted_s": 0.0,
            "breaks": 0,
            "upstream_errors": [],
            "first_audio": [],
        }

    # ------------------------------------------------------------------ downstream
    async def send(self, event_type: str, **fields: Any) -> None:
        data = {"type": event_type, **{k: v for k, v in fields.items() if v is not None}}
        data["server_send_ts"] = time.time()
        async with self._down_lock:
            await self.ws.send_str(json.dumps(data, ensure_ascii=False))

    async def send_delta(self, kind: str, **fields: Any) -> None:
        await self.send("response.output.delta", kind=kind, session_id=self.session_id, **fields)

    # ------------------------------------------------------------------ lifecycle
    async def init(self, params: dict[str, Any]) -> None:
        mode = str(params.get("mode") or "full_duplex")
        if mode != "full_duplex":
            raise ProtocolViolation(f"mode {mode!r} is not served: this backend only implements full_duplex")
        instructions = params.get("system_prompt") or params.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise ProtocolViolation("system_prompt must be a string")
        use_tts = params.get("use_tts", True)
        modalities = ["text"] if use_tts is False else ["audio"]
        config = params.get("config") or {}
        if not isinstance(config, dict):
            raise ProtocolViolation("config must be an object")
        sampling = {}
        sampling_fields = {
            "temperature": "temperature", "top_k": "top_k", "top_p": "top_p",
            "text_repetition_penalty": "repetition_penalty",
            "repetition_penalty": "repetition_penalty",
            "listen_prob_scale": "listen_prob_scale",
            "force_listen_count": "force_listen_count", "greedy": "greedy",
        }
        for key, value in config.items():
            if key in sampling_fields:
                sampling[sampling_fields[key]] = value
            elif key == "decode_mode":
                if value not in ("sampling", "greedy"):
                    raise ProtocolViolation("decode_mode must be sampling or greedy")
                sampling["greedy"] = value == "greedy"
            else:
                self.ignored_init_fields.append(f"config.{key}")
        extension = {}
        if sampling:
            extension["sampling"] = sampling
        self.max_slice_nums = params.get("max_slice_nums", 1)
        if "max_slice_nums" in params:
            extension["max_slice_nums"] = self.max_slice_nums
        voice = params.get("voice") or {}
        if not isinstance(voice, dict):
            raise ProtocolViolation("voice must be an object")
        if params.get("ref_audio_path") or voice.get("ref_audio_path"):
            raise ProtocolViolation("reference audio must be uploaded, not a server file path")
        for source, target in (("ref_audio", "reference_audio"), ("tts_ref_audio", "tts_reference_audio")):
            reference = (params.get(f"{source}_base64") or params.get(source)
                         or voice.get(f"{source}_base64") or voice.get(source))
            if reference:
                try:
                    wav = float32_to_wav(base64.b64decode(reference, validate=True))
                except (binascii.Error, ValueError, TypeError) as exc:
                    raise ProtocolViolation(f"invalid {source}: {exc}") from exc
                extension[target] = {"media_type": "audio/wav", "data": base64.b64encode(wav).decode("ascii")}

        self._http = aiohttp.ClientSession()
        self._up = await self._http.ws_connect(
            self.config.upstream_url,
            max_msg_size=UPSTREAM_MAX_MESSAGE_BYTES,
            compress=0,
            autoping=True,
            heartbeat=None,
        )
        created = await self._recv_until("session.created")
        self.upstream_session_id = (created.get("session") or {}).get("id")
        session: dict[str, Any] = {"output_modalities": modalities}
        if extension:
            session["sglang"] = extension
        if instructions:
            session["instructions"] = instructions
        await self._send_up("session.update", "control", session=session)
        updated = await self._recv_until("session.updated")
        self.granted = ((updated.get("session") or {}).get("sglang") or {}).get("granted") or {}
        in_rate = (self.granted.get("input_audio_format") or {}).get("rate", INPUT_RATE)
        out_rate = (self.granted.get("output_audio_format") or {}).get("rate", OUTPUT_RATE)
        if in_rate != INPUT_RATE or out_rate != OUTPUT_RATE:
            raise UpstreamFailure(f"upstream audio rates {in_rate}/{out_rate} Hz, expected {INPUT_RATE}/{OUTPUT_RATE}")
        if not self.granted.get("native_full_duplex", False):
            raise UpstreamFailure("upstream session is not native full duplex")
        self.unit_ms = int(self.granted.get("native_unit_ms") or 1000)
        image_format = self.granted.get("input_image_format") or {}
        self.image_enabled = self.config.forward_images and "image" in (self.granted.get("input_modalities") or [])
        self.image_max_bytes = int(image_format.get("max_bytes") or 0)
        self.image_max_per_unit = int(image_format.get("max_per_unit", 1))
        self._tasks = [asyncio.create_task(self._receiver(), name=f"{self.session_id}-recv")]
        if self.config.silence_fill:
            self._tasks.append(asyncio.create_task(self._pacer(), name=f"{self.session_id}-pacer"))
        log.info(
            "session %s open: upstream %s, unit %d ms, images %s, modalities %s, ignored init fields %s",
            self.session_id, self.upstream_session_id, self.unit_ms,
            "on" if self.image_enabled else "off", modalities, self.ignored_init_fields,
        )
        await self.send(
            "session.created",
            session_id=self.session_id,
            mode="full_duplex",
            metrics={"backend": BACKEND_NAME},
            sglang={
                "upstream_session_id": self.upstream_session_id,
                "native_unit_ms": self.unit_ms,
                "image_input": self.image_enabled,
                "max_images_per_unit": self.image_max_per_unit,
                "max_slice_nums": self.max_slice_nums,
                "fixed_settings": ["max_slice_nums"],
                "sampling": sampling,
                "output_modalities": modalities,
                "ignored_init_fields": self.ignored_init_fields,
            },
        )

    async def close(self, *, reason: str = "client_closed", emit_event: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        await self._shutdown_upstream()
        if emit_event:
            try:
                await self.send("session.closed", session_id=self.session_id, reason=reason, sglang=self.summary())
            except Exception:
                pass
        try:
            await self.ws.close(code=1000, message=reason.encode()[:120])
        except Exception:
            pass
        self.registry.forget(self.session_id)
        log.info("session %s closed (%s): %s", self.session_id, reason, json.dumps(self.summary()))

    async def fatal(self, reason: str, *, message: Optional[str] = None) -> None:
        if self.closed:
            return
        log.error("session %s fatal: %s %s", self.session_id, reason, message)
        self.closed = True
        try:
            await self.send(
                "session.closed",
                session_id=self.session_id,
                reason=reason,
                diagnostic={"message": message} if message else None,
                sglang=self.summary(),
            )
        except Exception:
            pass
        try:
            await self.ws.close(code=1011, message=reason.encode()[:120])
        except Exception:
            pass
        await self._shutdown_upstream()
        self.registry.forget(self.session_id)
        log.info("session %s summary: %s", self.session_id, json.dumps(self.summary()))

    async def _shutdown_upstream(self) -> None:
        self._closing_upstream = True
        current = asyncio.current_task()
        pacers = [task for task in self._tasks if task is not current and not task.get_name().endswith("-recv")]
        for task in pacers:
            task.cancel()
        receiver = next((t for t in self._tasks if t.get_name().endswith("-recv") and t is not current), None)
        if self._up is not None and not self._up.closed:
            try:
                await self._send_up("session.close", "control")
                if receiver is not None:
                    await asyncio.wait_for(asyncio.shield(receiver), self.config.close_timeout_s)
            except Exception:
                pass
        for task in self._tasks:
            if task is not current:
                task.cancel()
        await asyncio.gather(*(t for t in self._tasks if t is not current), return_exceptions=True)
        if self._up is not None:
            try:
                await self._up.close()
            except Exception:
                pass
        if self._http is not None:
            await self._http.close()

    def summary(self) -> dict[str, Any]:
        out = dict(self.stats)
        out["upstream_session_id"] = self.upstream_session_id
        out["timeline_s"] = round(self.packetizer.sent_ms / 1000, 3)
        for key in ("input_audio_s", "silence_filled_s", "audio_out_s", "audio_muted_s"):
            out[key] = round(out[key], 3)
        out["upstream_errors"] = out["upstream_errors"][:5]
        return out

    # ------------------------------------------------------------------ input
    async def push(self, message: dict[str, Any]) -> None:
        if self.closed:
            raise ProtocolViolation("session is closed")
        payload = message.get("input")
        if not isinstance(payload, dict):
            raise ProtocolViolation("input.append must carry an object `input`")
        audio_b64 = _extract_audio_base64(payload)
        if not audio_b64:
            raise ProtocolViolation("full_duplex input requires audio")
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProtocolViolation(f"audio is not base64: {exc}") from exc
        if len(raw) % 4:
            raise ProtocolViolation("audio must be raw float32 PCM")
        pcm = float32_to_pcm16(raw)
        frames = _extract_frames(payload)
        hints = payload.get("hints") if isinstance(payload.get("hints"), dict) else {}
        slice_nums = payload.get("max_slice_nums", hints.get("max_slice_nums", self.max_slice_nums))
        slice_values = slice_nums if isinstance(slice_nums, list) else [slice_nums]
        if isinstance(slice_nums, list) and len(slice_nums) != len(frames):
            raise ProtocolViolation("max_slice_nums must have one entry per frame")
        if any(type(value) is not int or value != self.max_slice_nums for value in slice_values):
            raise ProtocolViolation("max_slice_nums is fixed at session.init; reconnect to change HD slices")
        self._set_force_listen(bool(payload.get("force_listen", hints.get("force_listen", False))))
        async with self._input_lock:
            self.stats["pushes"] += 1
            self.stats["input_audio_s"] += len(pcm) / 2 / INPUT_RATE
            unit = math.floor(self.packetizer.sent_ms / self.unit_ms)
            if self.image_enabled and self._unit_frame_counts.get(unit, 0) + len(frames) > self.image_max_per_unit:
                raise ProtocolViolation(f"unit frame count exceeds upstream limit {self.image_max_per_unit}")
            for frame in frames:
                await self._send_frame(frame)
            if pcm:
                await self._send_audio(pcm)
            self._anchor_wall = time.monotonic()
            self._anchor_ms = self.packetizer.sent_ms

    def _set_force_listen(self, active: bool) -> None:
        """force_listen is the demo's only barge-in control; /v1/realtime has no cancel.

        While it is asserted, the response in flight and any response that starts are muted
        here (their deltas are dropped); the server keeps generating them.
        """
        if active and not self.force_listen:
            self.stats["breaks"] += 1
            if self.current_response and self.response_active:
                self.muted.add(self.current_response)
            log.info("session %s force_listen on: muting %s", self.session_id, self.current_response if self.response_active else None)
        self.force_listen = active

    async def _send_frame(self, frame_b64: str) -> None:
        if not self.image_enabled:
            self.stats["frames_skipped"] += 1
            return
        t_ms = self.packetizer.sent_ms
        unit = math.floor(t_ms / self.unit_ms)
        try:
            size = len(base64.b64decode(frame_b64, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise ProtocolViolation(f"video frame is not base64: {exc}") from exc
        if self.image_max_bytes and size > self.image_max_bytes:
            self.stats["frames_skipped"] += 1
            return
        self._unit_frame_counts[unit] = self._unit_frame_counts.get(unit, 0) + 1
        await self._send_up("sglang.input_image.append", "image", image=frame_b64, sglang={"t_ms": t_ms})
        self.stats["frames_forwarded"] += 1

    async def _send_audio(self, pcm: bytes) -> None:
        for seq, t_start_ms, body in self.packetizer.push(pcm):
            await self._send_up(
                "input_audio_buffer.append",
                "audio",
                audio=base64.b64encode(body).decode("ascii"),
                sglang={"seq": seq, "t_start_ms": t_start_ms},
            )
            end_ms = self.packetizer.sent_ms
            now = time.monotonic()
            unit = math.floor(t_start_ms / self.unit_ms)
            while (unit + 1) * self.unit_ms <= end_ms + 1e-6:
                self._unit_input_done.setdefault(unit, now)
                unit += 1

    async def _pacer(self) -> None:
        """Advance the timeline with silence only when the worker's stream has stopped (after its first chunk).

        Fills whole model units (first up to the next unit boundary), so a stream of
        1 s chunks stays aligned with the model's 1 s units after a pause.
        """
        try:
            while not self.closed:
                await asyncio.sleep(0.05)
                if self._anchor_wall is None:
                    continue  # the page is still starting its media; nothing to keep alive yet
                async with self._input_lock:
                    elapsed_ms = (time.monotonic() - self._anchor_wall) * 1000
                    filled_ms = self.packetizer.sent_ms - self._anchor_ms
                    behind_ms = elapsed_ms - self.config.silence_grace_ms - filled_ms
                    to_boundary = (-self.packetizer.sent_ms) % self.unit_ms or self.unit_ms
                    if behind_ms >= to_boundary:
                        samples = int(round(to_boundary * INPUT_RATE / 1000))
                        await self._send_audio(b"\0\0" * samples)
                        self.stats["silence_filled_s"] += samples / INPUT_RATE
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not self.closed:
                await self.fatal("backend_error", message=f"silence pacer failed: {exc!r}")

    # ------------------------------------------------------------------ output
    def _metrics(self, unit: Optional[int]) -> dict[str, Any]:
        metrics: dict[str, Any] = {"backend": BACKEND_NAME}
        done = self._unit_input_done.get(unit) if unit is not None else None
        if done is not None:
            # Time from the moment this unit's last input packet was forwarded upstream
            # to the moment this event arrived from sglang-omni.
            metrics["wall_clock_ms"] = round((time.monotonic() - done) * 1000, 1)
        if unit is not None:
            metrics["unit_index"] = unit
        return metrics

    async def _receiver(self) -> None:
        assert self._up is not None
        try:
            async for msg in self._up:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                await self._on_upstream(json.loads(msg.data))
            if not self._closing_upstream:
                raise UpstreamFailure("upstream websocket closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._closing_upstream or self.closed:
                return
            reason = "upstream_closed" if isinstance(exc, UpstreamFailure) else "backend_error"
            self._fatal_task = asyncio.create_task(self.fatal(reason, message=str(exc)))

    async def _on_upstream(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        ext = event.get("sglang") or {}
        unit = _unit_index(ext.get("unit_id") or event.get("unit_id"))
        if kind == "response.created":
            rid = (event.get("response") or {}).get("id")
            self.current_response = rid
            self.response_active = True
            self.stats["responses"] += 1
            if self.force_listen and rid:
                self.muted.add(rid)
        elif kind == "response.output_audio.delta":
            rid = event.get("response_id")
            pcm = base64.b64decode(event.get("delta") or "")
            seconds = len(pcm) / 2 / OUTPUT_RATE
            if rid in self.muted:
                self.stats["audio_muted_s"] += seconds
                return
            metrics = self._metrics(unit)
            if rid not in self._first_audio_seen:
                self._first_audio_seen.add(rid)
                if "wall_clock_ms" in metrics:
                    metrics["first_audio_ms"] = metrics["wall_clock_ms"]
                entry = {"response_id": rid, "unit_index": unit, "unit_to_audio_ms": metrics.get("wall_clock_ms"), "received_at": round(time.time(), 3)}
                self.stats["first_audio"] = (self.stats["first_audio"] + [entry])[-20:]
                log.info("session %s first audio of %s in unit %s, %s ms after the unit's input was forwarded", self.session_id, rid, unit, metrics.get("wall_clock_ms"))
            self.stats["audio_out_s"] += seconds
            if unit is not None:
                self._units_emitted.add(unit)
            await self.send_delta(
                "audio",
                response_id=rid,
                audio=base64.b64encode(pcm16_to_float32(pcm)).decode("ascii"),
                metrics=metrics,
            )
        elif kind in TEXT_DELTAS:
            rid = event.get("response_id")
            text = event.get("delta") or ""
            if rid in self.muted or not text:
                return
            if unit is not None:
                self._units_emitted.add(unit)
            await self.send_delta("text", response_id=rid, text=text, metrics=self._metrics(unit))
        elif kind == "response.done":
            rid = (event.get("response") or {}).get("id")
            self.response_active = False
            self.muted.discard(rid)
            if unit is not None:
                self._units_emitted.add(unit)
            # py_backend ends a full-duplex turn with a kind=listen carrying the response id.
            await self.send_delta("listen", response_id=rid, metrics=self._metrics(unit))
        elif kind == "sglang.unit.done":
            self.stats["units"] += 1
            if unit is not None and unit not in self._units_emitted:
                if not self.response_active or self.current_response in self.muted:
                    await self.send_delta("listen", metrics=self._metrics(unit))
            if unit is not None:
                self._units_emitted.discard(unit)
                self._unit_input_done.pop(unit, None)
                self._unit_frame_counts.pop(unit, None)
        elif kind == "error":
            error = event.get("error") or {}
            source = self._sent_kinds.get(error.get("event_id") or event.get("client_event_id") or "")
            entry = {"code": error.get("code"), "message": error.get("message"), "param": error.get("param"), "source": source}
            self.stats["upstream_errors"].append(entry)
            if ext.get("fatal") or source in ("audio", "control"):
                # A rejected audio packet breaks the contiguous seq/t_start_ms timeline for good.
                raise UpstreamFailure(f"upstream error: {json.dumps(entry, ensure_ascii=False)}")
            log.warning("session %s upstream error (non-fatal): %s", self.session_id, entry)
        elif kind == "session.closed":
            raise UpstreamFailure(f"upstream session closed: {event.get('reason')}")

    # ------------------------------------------------------------------ upstream wire
    async def _send_up(self, event_type: str, source: str, **payload: Any) -> None:
        if self._up is None or self._up.closed:
            raise UpstreamFailure("upstream websocket is closed")
        event_id = "evt_" + uuid.uuid4().hex[:16]
        self._sent_kinds[event_id] = source
        if len(self._sent_kinds) > SENT_EVENT_MEMORY:
            self._sent_kinds.popitem(last=False)
        await self._up.send_str(json.dumps({"type": event_type, "event_id": event_id, **payload}))

    async def _recv_until(self, event_type: str) -> dict[str, Any]:
        assert self._up is not None
        deadline = time.monotonic() + self.config.open_timeout_s
        while True:
            msg = await self._up.receive(timeout=max(0.01, deadline - time.monotonic()))
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise UpstreamFailure(f"upstream closed before {event_type} ({msg.type.name})")
            event = json.loads(msg.data)
            if event.get("type") == event_type:
                return event
            if event.get("type") in ("error", "session.closed"):
                raise UpstreamFailure(f"upstream refused the session: {json.dumps(event)[:400]}")
