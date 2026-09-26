"""PCM conversion and the 80 ms packetizer (standard library only).

The demo's backend protocol carries base64 raw float32 PCM (native byte order,
16 kHz mono in, 24 kHz mono out; docs/backend-protocol/schema.md §1.3). The
sglang-omni ``/v1/realtime`` protocol carries base64 PCM16 little-endian.
"""

from __future__ import annotations

import array
import sys

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
PACKET_MS = 80
SAMPLES_PER_MS = INPUT_RATE // 1000
PACKET_BYTES = PACKET_MS * SAMPLES_PER_MS * 2


def float32_to_pcm16(data: bytes) -> bytes:
    """Native-endian float32 samples in [-1, 1] -> little-endian PCM16 (clipped)."""
    floats = array.array("f")
    floats.frombytes(data[: len(data) // 4 * 4])
    out = array.array("h", [min(32767, max(-32768, int(round(v * 32767.0)))) for v in floats])
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()


def pcm16_to_float32(data: bytes) -> bytes:
    """Little-endian PCM16 -> native-endian float32 in [-1, 1)."""
    samples = array.array("h")
    samples.frombytes(data[: len(data) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    return array.array("f", [v / 32768.0 for v in samples]).tobytes()


class Packetizer:
    """Cuts one contiguous 16 kHz PCM16 timeline into 80 ms packets.

    ``seq`` counts packets from 0 and ``t_start_ms`` is the audio time already
    sent, as ``/v1/realtime`` requires (contiguous seq, sample-contiguous media
    time). A chunk that is not a multiple of 80 ms ends with one shorter packet
    that goes out immediately: holding the remainder back would keep the model
    unit that ends inside it open until the next chunk arrives.
    """

    def __init__(self) -> None:
        self.seq = 0
        self.sent_samples = 0

    @property
    def sent_ms(self) -> float:
        return self.sent_samples / SAMPLES_PER_MS

    def push(self, pcm: bytes) -> list[tuple[int, float, bytes]]:
        packets = []
        for offset in range(0, len(pcm), PACKET_BYTES):
            body = pcm[offset : offset + PACKET_BYTES]
            packets.append((self.seq, self.sent_ms, body))
            self.seq += 1
            self.sent_samples += len(body) // 2
        return packets
