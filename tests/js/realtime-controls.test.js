import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { RealtimeSession } from '../../static/duplex/lib/realtime-session.js';

vi.mock('../../static/duplex/lib/audio-player.js', () => ({
    AudioPlayer: class { init() {} stop() {} },
}));

let reply;
class Socket {
    static OPEN = 1;
    readyState = 1;
    sent = [];
    constructor() { setTimeout(() => this.onopen(), 0); }
    send(raw) {
        const event = JSON.parse(raw);
        this.sent.push(event);
        if (event.type === 'session.init') queueMicrotask(() => this.onmessage({ data: JSON.stringify(reply) }));
    }
    close() { this.readyState = 3; }
}

beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('WebSocket', Socket);
});
afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

describe('duplex session controls', () => {
    it('forwards page settings and exposes unsupported/fixed settings before media starts', async () => {
        reply = { type: 'session.created', session_id: 'demo', sglang: {
            ignored_init_fields: ['config.length_penalty'], fixed_settings: ['max_slice_nums'],
            sampling: { greedy: false, temperature: 0.6 },
        } };
        const session = new RealtimeSession('test', { getWsUrl: () => 'ws://test' });
        const logs = [];
        session.onSystemLog = text => logs.push(text);
        const payload = { config: { decode_mode: 'sampling', temperature: 0.6 }, max_slice_nums: 4,
            ref_audio_base64: 'reference', tts_ref_audio_base64: 'tts' };
        const start = session.start('prompt', payload, async () => {
            expect(session.backendInfo).toEqual(reply.sglang);
        });
        await vi.advanceTimersByTimeAsync(101);
        await start;
        expect(session.ws.sent[0].payload).toEqual({ system_prompt: 'prompt', ...payload });
        expect(logs.some(text => text.includes('config.length_penalty'))).toBe(true);
        expect(logs.some(text => text.includes('Stop and restart'))).toBe(true);
        session.stop();
        expect(session.running).toBe(false);
    });

    it('rejects initialization when backend closes with a configuration diagnostic', async () => {
        reply = { type: 'session.closed', reason: 'backend_error', diagnostic: { message: 'invalid reference' } };
        const session = new RealtimeSession('test', { getWsUrl: () => 'ws://test' });
        const check = expect(session.start('', {})).rejects.toThrow('invalid reference');
        await vi.advanceTimersByTimeAsync(101);
        await check;
        expect(session.running).toBe(false);
        expect(session.ws).toBeNull();
    });
});
