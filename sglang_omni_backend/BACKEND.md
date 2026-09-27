# sglang-omni backend for the MiniCPM-o demo

This directory is a drop-in replacement for `py_backend/server.py`: a backend-protocol server whose inference runs on a separate [sglang-omni](https://github.com/sgl-project/sglang-omni) server through its native full-duplex `/v1/realtime` WebSocket. Gateway and worker transport are unchanged; the audio/video pages expose shared sampling controls; the worker is pointed at this process with `--backend-server-url`, the same way `docker-compose.cpp.yml` swaps in the C++ backend.

```text
Browser -> gateway.py (/v1/realtime) -> worker.py (/v1/worker/duplex) -> sglang_omni_backend (/backend) -> sglang-omni (/v1/realtime)
```

Only `aiohttp` is needed (`pip install aiohttp`); no torch, no model weights.

```bash
python -m sglang_omni_backend --host 127.0.0.1 --port 22500 --upstream-url ws://127.0.0.1:18260/v1/realtime
python worker.py --host 127.0.0.1 --port 22400 --backend-server-url http://127.0.0.1:22500
python gateway.py --host 127.0.0.1 --port 8006 --internal-port 8007 --http
curl -X PUT http://127.0.0.1:8007/internal/workers/sglang-omni-0 -H 'content-type: application/json' \
     --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}'
python -m sglang_omni_backend.test_backend   # contract test against a fake upstream, no GPU
```

Options (`python -m sglang_omni_backend --help`): `--upstream-url` (env `UPSTREAM_URL`), `--max-sessions` (env `MAX_SESSIONS`, default 1), `--[no-]forward-images` (default on; frames are sent only if the server grants image input), `--[no-]silence-fill` and `--silence-grace-ms` (see [Input](#input-inputappend)).

The rest of this file is the contract as the worker exercises it, with references into this repository (paths relative to the repo root), and how each part is mapped onto `/v1/realtime`.

## 1. Who calls what

| Step | Caller | Reference |
|---|---|---|
| Page connects `ws://<gateway>/v1/realtime?mode=video\|audio`, waits for `session.queue_done`, sends `session.init`, then one `input.append` per second, `session.close` on Stop | browser | `static/duplex/lib/realtime-session.js:107-236` (init), `:243-272` (chunks), `:305-312` (stop) |
| Gateway queues, picks an idle worker, opens `ws://<worker>/v1/worker/duplex`, passes text frames through unchanged both ways, 300 s (video) / 600 s (audio) watchdog | gateway | `gateway.py:1384-1418`, `gateway.py:360-594` |
| Worker accepts one session at a time (`is_idle`), opens the backend session, forwards `input.append` as push, `session.close` as HTTP close, and every backend event back verbatim | worker | `worker.py:174-297`, idle check `:190-192`, endpoints `:300-325`, flag `--backend-server-url` `:367` |
| Backend-protocol client used by the worker | runtime | `runtime/backend_client.py:54-133`, `runtime/session.py:11-32` |

So everything below is what `runtime/backend_client.py::RemoteBackendSession` sends and what the page finally reads; the worker and gateway inspect nothing but `type` (`worker.py:253-260`, `gateway.py:517-546`).

## 2. Transport

| Item | Contract | Reference | Here |
|---|---|---|---|
| Data channel | `WS /backend`, JSON text frames, `type` on every message, unknown fields ignored | `docs/backend-protocol/schema.md` §1.1, §2.1; `py_backend/server.py:594-638`; client `runtime/backend_client.py:64-71` (http→ws URL, `max_size` 128 MiB) | same path; 128 MiB frame limit |
| Control channel | `POST /sessions/{id}/close`, body `{"reason"}`, reply `{"ok":true,"session_id","closed":true}`, 404 once forgotten | schema §2.2, §7.3; `py_backend/server.py:641-657`; client `runtime/backend_client.py:110-133` (`raise_for_status`) | same |
| Health | `GET /health` (not called by the worker; used by deploy scripts) | `py_backend/server.py:584-591` | `status` is `ready` only when the upstream `/health` answers 200 (else 503 `upstream_unavailable`) |
| Timestamps | every downstream event SHOULD carry `server_send_ts` (Unix s, float) | schema §1.5; `py_backend/server.py:173-176` | on every event; the page uses it for its drift readout (`realtime-session.js:432-443`) |
| Concurrency | one active session per backend; a second init is fatal | schema §7.4; `py_backend/server.py:127-145`; worker also refuses a second socket while busy (`worker.py:190-192`) | `--max-sessions N` (default 1). Each session still needs its own worker process, because a worker serves one socket at a time |

## 3. Session open

`runtime/backend_client.py:54-78` connects, sends

```json
{"type": "session.init", "payload": {"mode": "full_duplex", "system_prompt": "...", "config": {"length_penalty": 1.0}, "max_slice_nums": 1, "use_tts": true, "ref_audio_base64": "<f32 16k>", "tts_ref_audio_base64": "<f32 16k>"}}
```

(`mode` is added by the worker, `worker.py:220-222`; the rest is the page's `preparePayload`, `static/omni/omni-app.js:1597-1605`, `static/audio-duplex/audio-duplex-app.js:968-975`, merged at `realtime-session.js:135-141`; schema §3.1 also allows `voice.{ref_audio,tts_ref_audio}`), and requires the **first** downstream event to be `session.created` with a backend-assigned `session_id` (`backend_client.py:73-78`, schema §3.2). A worker may also open with `input.append` directly; it then synthesises `{"mode": ...}` as init (`worker.py:214-216`).

| Init field | py_backend | Here |
|---|---|---|
| `mode` | `full_duplex` or `turn_based` (`server.py:608`) | `full_duplex` only; `turn_based` (the gateway's `mode=chat`, `/turnbased` page) is refused fail-fast with a diagnostic |
| `system_prompt` / `instructions` | system prompt, default `You are a helpful assistant.` (`server.py:275-279`) | `session.update.session.instructions`; omitted when empty, so the server's default applies |
| `use_tts` | not read by py_backend | `false` → `output_modalities: ["text"]`, else `["audio"]` |
| `voice.*`, `ref_audio_base64`, `tts_ref_audio_base64`, `ref_audio_path` | LLM and TTS reference audio (`server.py:258-271`, `py_backend/voice.py`) | Uploaded 16 kHz float32 references are converted to PCM16 WAV and sent as `session.sglang.reference_audio` / `tts_reference_audio`. Separate TTS reference takes precedence; otherwise upstream uses the main reference, then its deployment default. File paths are rejected. |
| `config` (sampling, `length_penalty`, ...) | `set_duplex_config` (`server.py:254-256`; field list schema §6) | `temperature`, `top_k`, `top_p`, `listen_prob_scale`, `force_listen_count` and `greedy` map to `session.sglang.sampling`. `decode_mode` maps to `greedy`; `text_repetition_penalty` maps to `repetition_penalty`. Other keys, including `length_penalty`, are listed individually as unsupported and shown in the page log. |
| `max_slice_nums` | vision HD slices (schema §4.3) | Forwarded as `session.sglang.max_slice_nums`; fixed for the session. Stop and restart to change HD. |

Open sequence here: connect `UPSTREAM_URL` → wait `session.created` → `session.update {output_modalities, instructions, sglang}` → wait `session.updated` and read `session.sglang.granted` (checks 16 kHz in / 24 kHz out and `native_full_duplex`; takes `native_unit_ms`, `input_modalities`, `input_image_format.max_bytes`) → reply

```json
{"type": "session.created", "session_id": "sess_<12 hex>", "mode": "full_duplex", "metrics": {"backend": "sglang-omni"},
 "sglang": {"upstream_session_id": "...", "native_unit_ms": 1000, "image_input": true, "output_modalities": ["audio"], "max_images_per_unit": 4, "max_slice_nums": 1, "fixed_settings": ["max_slice_nums"], "ignored_init_fields": ["config.length_penalty"]}, "server_send_ts": 1.7e9}
```

Any upstream failure during open is fatal (see §6).

## 4. Input (`input.append`)

`runtime/backend_client.py:80-86` sends `{"type":"input.append","input":{...}}`; full-duplex `input` is schema §4.1:

| Field | Shape | What the pages send | Here |
|---|---|---|---|
| `audio` (aliases `audio_base64`, `audio_data`, `{data}`; `server.py:94-107`) | base64 raw float32, native byte order, 16 kHz mono (schema §1.3, §8) | exactly 1 s (16000 samples) per message from the AudioWorklet (`static/duplex/lib/capture-processor.js:19,62-65`, `chunkSize: SAMPLE_RATE_IN` at `omni-app.js:355-357`); file mode also 1 s (`omni-app.js:38,601`); `examples/realtime/audio_probe.py` default `--chunk-ms 1000` | converted to PCM16 LE and cut into 80 ms `input_audio_buffer.append` packets (`sglang.seq` contiguous from 0, `sglang.t_start_ms` = audio time already sent), all sent at once. A 1 s chunk is 12 packets of 80 ms + one of 40 ms, sent immediately so the unit it completes is not held back |
| `video_frames` (alias `frame_base64_list`, `frames[]`; `server.py:75-91`) | base64 JPEG strings (schema §1.4) | one frame per 1 s chunk in the video page, full camera resolution, JPEG q=0.7 (`omni-app.js:369-374,421-437`, `realtime-session.js:257-259`) | all frames are sent in list order as `sglang.input_image.append` before the chunk's audio, with the chunk start as `sglang.t_ms`. The protocol carries no per-frame capture timestamps; equal timestamps preserve list order. The advertised `input_image_format.max_per_unit` limits frames in each unit. Exceeding it fails explicitly instead of dropping all but the last frame. Image input disabled/not granted and oversized frames retain the existing counted-skip behavior. |
| `max_slice_nums` | int or list (schema §4.3) | `1`, or `2` with HD on (overview + up to two crops) | Must match the session setting. Uniform per-frame lists are accepted; changes or mixed values require a new session and are rejected with a diagnostic. The SGLang-backed page locks HD during a session. |
| `force_listen` (or `hints.force_listen`) | bool (`server.py:418-419`) | set on every chunk while the page's Force Listen toggle is on (`realtime-session.js:254-256,274-286`) | see [Break](#7-break--interrupt) |

Cadence: py_backend handles one push at a time, prefill+generate synchronously, and the model cuts 1 s units (first unit 1035 ms in the vendored model). The sglang-omni server cuts 1 s units from media time 0 (`granted.first_unit_ms = native_unit_ms = 1000`), so a stream of 1 s chunks completes exactly one unit per chunk and no unit waits for the next chunk.

Pauses: the page's Pause button simply stops sending (`realtime-session.js:288-303`; the protocol has no pause, network §8). The upstream model only advances on input, so an in-flight reply would freeze. With `--silence-fill` (default) a pacer advances the timeline with silence when, after the first chunk, the next chunk is more than `--silence-grace-ms` (default 1000 ms) late: it fills up to the next unit boundary, then whole units, so a later 1 s chunk is again unit-aligned. With `--no-silence-fill` nothing is inserted (py_backend's behaviour: the model waits).

## 5. Output events

What py_backend emits per push (`server.py:408-518`) and what the page does with it (`realtime-session.js:461-553`), against what is emitted here from the upstream events:

| Downstream event | py_backend | Page | Here, from `/v1/realtime` |
|---|---|---|---|
| `response.output.delta kind=listen` (`session_id`, `response_id` at turn end, `metrics`) | once per push when the model listens (`server.py:471-482`), and once after the last speak unit of a turn (`end_of_turn`, `:507-516`) | ends the audio turn, closes the subtitle bubble, records the time for TTFS (`:486-512`) | on `sglang.unit.done` for a unit that produced nothing while no response is active (or the active one is muted); and on `response.done` (with `response_id`), which ends the turn |
| `kind=text` (`text`) | ≤1 per speak unit (`:487-496`) | appended to the subtitle (`:542-549`) | each `response.output_audio_transcript.delta` / `response.output_text.delta` |
| `kind=audio` (`audio` = base64 float32 **24 kHz** mono, per `docs/audio-duplex-protocol.md:9`, `docs/realtime-protocol-overview.md:80`) | ≤1 per speak unit, whole unit at once (`:497-506`) | queued to the player (`:526-529`, `outputSampleRate: 24000`) | each `response.output_audio.delta`, PCM16 → float32. With the current server this is also one text + one audio delta per speak unit, sent as the unit completes |
| `response_id` | `resp_<12 hex>` per turn | not interpreted | upstream response id |
| `response.done` | turn_based only (schema §5.2; network §6.6) | — | never (full_duplex only) |
| `session.closed` (`reason`, `diagnostic`) | on close and fail-fast (`server.py:209-251`) | stops the session (`:389-394`) | same; also carries `sglang` session counters |

`metrics` (schema §5.4) as the page reads them (`realtime-session.js:445-459,571-590`): latency readout = `wall_clock_ms || generate_ms`, "cost" = `generate_ms`, `kv_cache_length` (the page auto-stops at 8192 by default, `:610-627`), `vision_slices`, `vision_tokens`; `usage`/`chunk_index` are extra py_backend fields the page ignores.

Here `metrics` carries `backend: "sglang-omni"`, `unit_index`, and `wall_clock_ms` = time from forwarding the unit's last input packet upstream to receiving this event (the unit's server round trip; py_backend's `wall_clock_ms` is its prefill+generate time for the push, `server.py:422-439`). The first audio delta of each response also carries `first_audio_ms` (same quantity) and is logged. Not reported because `/v1/realtime` does not expose them: `generate_ms`, `prefill_ms`, `cost_*_ms`, `n_tokens`, `n_tts_tokens`, `vision_*`, and `kv_cache_length` (so the page's 8192-token auto-stop never fires; the gateway's 300/600 s session limit still does).

## 6. Close, disconnect, fail-fast

| Case | Contract | Here |
|---|---|---|
| `POST /sessions/{id}/close` | completion semantics: session closed and resources released before the reply; best-effort `session.closed` on the WS, then WS closed; later close → 404 (network §3.2, §4.3; schema §7.3; `server.py:209-231,641-657`) | sends upstream `session.close`, waits (≤ 5 s) for the upstream `session.closed`, closes the upstream socket, sends `session.closed {reason}`, closes the WS (1000), forgets the id, then replies |
| Worker WS drops | no resume; release resources (network §9; `server.py:630-632`) | same cleanup, no event |
| Illegal input: first message not `session.init`, second init, unknown `type`, missing/invalid audio, init over capacity, `mode` other than `full_duplex` | fail-fast: best-effort `session.closed {reason, diagnostic.message}`, WS close (schema §7.4-7.5; `server.py:633-638`) | same (close code 1011, `reason: backend_error`) |
| Upstream closes, upstream `error` with `sglang.fatal`, or a non-fatal error on an audio packet or control event (a rejected packet breaks the contiguous timeline) | "inference engine exception" is fail-fast (schema §7.4) | fail-fast with `reason: upstream_closed` or `backend_error` and the upstream error in `diagnostic.message` |
| Non-fatal upstream error on an image | — | logged and counted in `sglang.upstream_errors`; the session continues |

## 7. Break / interrupt

The backend protocol has no break or cancel message (network §2 lists only init/push/pull/close; §8 no pause). The pages' only barge-in control is `force_listen` (the Force Listen button, `realtime-session.js:274-286`, which also stops local playback). In py_backend it makes the model listen for that push; mid-speech it ends the turn in the model (`server.py:419,432`).

`/v1/realtime` on the MiniCPM-o native duplex server has no cancel (`granted.supports_server_interrupt: false`) and no forced-listen input. So here `force_listen` is mapped to **muting**: when it switches on, the response in flight is muted, and every response that starts while it stays on is muted too; muted responses' text and audio deltas are dropped here (counted in `audio_muted_s`) and each of their units is reported as `kind=listen`. The server keeps generating the muted response and its tokens stay in the model's context; the model is not forced to listen. Barge-in by speaking over the model works as the model itself decides (it may stop at the next unit).

## 8. Sessions and deployment

The reference model is one session per backend process and one backend per worker (`docker-compose.yml`, `docker/entrypoint-worker-backend.sh`). This process accepts up to `--max-sessions` concurrent sessions (default 1, matching py_backend), each with its own upstream `/v1/realtime` session; the sglang-omni server has its own session limit. For N concurrent users run N workers, all with `--backend-server-url` pointing here, start this process with `--max-sessions N`, and register each worker with the gateway (`PUT /internal/workers/<id>`); the gateway queues the rest.

`docker-compose.sglang-omni.yml` + `docker/Dockerfile.sglang-omni-worker-backend` + `docker/entrypoint-sglang-omni-worker-backend.sh` follow the C++ pattern (`docker-compose.cpp.yml`): one container runs this backend on 127.0.0.1:22500 and the worker on :22400, registers the worker with the gateway, and needs no GPU. `scripts/sglang_omni_local.sh` starts the same three processes on bare metal.

## 9. Known differences from py_backend

- Requires the SGLang-Omni sampling/reference/multiframe capabilities from private PRs #358–#360 (or an equivalent integrated revision). Unsupported explicit settings are refused by upstream; no fallback pretends they worked.
- `length_penalty`, TTS sampling and other unmapped config fields remain unsupported. Length penalty is tracked separately in [Draft #361](https://github.com/lijrjyan/sglang-omni-private/pull/361); it is not repetition penalty.
- HD slices are fixed per session; per-frame mixed slice limits and turn-based chat are not supported.
- The existing camera capture still supplies one frame per second; clients supplying a frame list can now forward all frames up to the negotiated limit. No new camera frame-rate control is added.
- `force_listen` mutes instead of forcing the model to listen (§7).
- Metrics are round-trip times measured here, not model timings; no KV length (§5).
- With the default silence fill, a paused page keeps the model's clock running (§4).

## Validation

```bash
python -m sglang_omni_backend.test_backend
python -m unittest sglang_omni_backend.test_controls
npm ci
npm test
```

The backend tests exercise actual HTTP/WebSocket messages with a simulated
upstream, including reference WAV conversion, sampling aliases, four-frame
ordering, overflow rejection and fixed slice settings. They are not GPU or
browser end-to-end evidence.
