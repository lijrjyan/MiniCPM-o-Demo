"""Session controls exercised through the public backend HTTP/WebSocket surface."""
import array
import base64
import io
import unittest
import wave

import aiohttp
from aiohttp import web

from .audio import float32_to_wav
from .server import create_app
from .session import BackendConfig
from .test_backend import FakeUpstream, JPEG, float_chunk, free_port


class ControlsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream = FakeUpstream()
        upstream_app = web.Application()
        upstream_app.router.add_get('/v1/realtime', self.upstream.handler)
        upstream_port, self.port = free_port(), free_port()
        self.runners = []
        for app, port in ((upstream_app, upstream_port), (create_app(BackendConfig(
            upstream_url=f'ws://127.0.0.1:{upstream_port}/v1/realtime', silence_fill=False)), self.port)):
            runner = web.AppRunner(app)
            await runner.setup()
            await web.TCPSite(runner, '127.0.0.1', port).start()
            self.runners.append(runner)
        self.http = aiohttp.ClientSession()
        self.ws = await self.http.ws_connect(f'ws://127.0.0.1:{self.port}/backend')

    async def asyncTearDown(self):
        await self.ws.close()
        await self.http.close()
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def init(self, **params):
        await self.ws.send_json({'type': 'session.init', 'payload': params})
        return await self.ws.receive_json(timeout=5)

    async def test_voice_aliases_and_sampler_mapping(self):
        a, b = float_chunk(.1), float_chunk(.2)
        event = await self.init(voice={'ref_audio': a, 'tts_ref_audio': b}, max_slice_nums=4,
                               config={'decode_mode': 'greedy', 'temperature': 0., 'top_k': 0,
                                       'top_p': 1., 'text_repetition_penalty': 1.2,
                                       'listen_prob_scale': 0., 'force_listen_count': 0,
                                       'length_penalty': 1.5})
        self.assertEqual(event['type'], 'session.created')
        ext = self.upstream.update['sglang']
        self.assertEqual(ext['sampling'], {'greedy': True, 'temperature': 0., 'top_k': 0,
                                          'top_p': 1., 'repetition_penalty': 1.2,
                                          'listen_prob_scale': 0., 'force_listen_count': 0})
        self.assertEqual(ext['max_slice_nums'], 4)
        for name, frames in (('reference_audio', 1600), ('tts_reference_audio', 3200)):
            with wave.open(io.BytesIO(base64.b64decode(ext[name]['data']))) as wav:
                self.assertEqual((wav.getframerate(), wav.getnframes()), (16000, frames))
        self.assertEqual(event['sglang']['ignored_init_fields'], ['config.length_penalty'])

    async def test_reference_only_uses_upstream_tts_fallback(self):
        await self.init(ref_audio_base64=float_chunk(.1))
        self.assertIn('reference_audio', self.upstream.update['sglang'])
        self.assertNotIn('tts_reference_audio', self.upstream.update['sglang'])

    async def test_empty_config_preserves_server_defaults(self):
        event = await self.init()
        self.assertEqual(event['type'], 'session.created')
        self.assertEqual(self.upstream.update, {'output_modalities': ['audio']})

    async def test_invalid_reference_is_rejected_before_upstream_open(self):
        event = await self.init(ref_audio_base64='not-base64!')
        self.assertEqual(event['type'], 'session.closed')
        self.assertIn('invalid ref_audio', event['diagnostic']['message'])
        self.assertIsNone(self.upstream.update)

    async def test_reference_path_is_not_accepted(self):
        event = await self.init(ref_audio_path='/tmp/server.wav')
        self.assertEqual(event['type'], 'session.closed')
        self.assertIn('uploaded', event['diagnostic']['message'])

    async def test_invalid_decode_mode_is_rejected(self):
        event = await self.init(config={'decode_mode': 'beam'})
        self.assertEqual(event['type'], 'session.closed')
        self.assertIn('decode_mode', event['diagnostic']['message'])

    async def test_slice_change_requires_reconnect(self):
        await self.init(max_slice_nums=2)
        await self.ws.send_json({'type': 'input.append', 'input': {
            'audio': float_chunk(), 'max_slice_nums': 4}})
        event = await self.ws.receive_json(timeout=5)
        self.assertEqual(event['type'], 'session.closed')
        self.assertIn('reconnect', event['diagnostic']['message'])
        self.assertEqual(self.upstream.appends, [])

    async def test_frame_overflow_is_not_silently_truncated(self):
        await self.init()
        await self.ws.send_json({'type': 'input.append', 'input': {
            'audio': float_chunk(), 'video_frames': [base64.b64encode(JPEG).decode()] * 5}})
        event = await self.ws.receive_json(timeout=5)
        self.assertEqual(event['type'], 'session.closed')
        self.assertIn('frame count', event['diagnostic']['message'])
        self.assertEqual(self.upstream.images, [])
        self.assertEqual(self.upstream.appends, [])

    async def test_uniform_per_frame_slices(self):
        await self.init(max_slice_nums=4)
        await self.ws.send_json({'type': 'input.append', 'input': {
            'audio': float_chunk(), 'video_frames': [base64.b64encode(JPEG).decode()] * 4,
            'max_slice_nums': [4, 4, 4, 4]}})
        event = await self.ws.receive_json(timeout=5)
        self.assertEqual(event['kind'], 'listen')
        self.assertEqual(self.upstream.images, [0.] * 4)


class ReferenceFormatTest(unittest.TestCase):
    def test_pcm_roundtrip(self):
        data = array.array('f', [-1., -.5, 0., .5, 1.]).tobytes()
        with wave.open(io.BytesIO(float32_to_wav(data))) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
            self.assertEqual(wav.getnframes(), 5)

    def test_invalid_length_and_budget(self):
        for data in (b'', b'abc', bytes(2 * 1024 * 1024)):
            with self.subTest(length=len(data)), self.assertRaises(ValueError):
                float32_to_wav(data)


if __name__ == '__main__':
    unittest.main()
