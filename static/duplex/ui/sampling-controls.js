/** Shared session sampling controls for the audio and video duplex pages. */
export function mountSamplingControls(lengthPenaltyId) {
    const lengthPenalty = document.getElementById(lengthPenaltyId);
    const section = document.createElement('details');
    section.className = 'config-group';
    section.innerHTML = `
        <summary>Sampling</summary><div class="cg-body">
        <label class="cg-row inline"><span class="cg-label">Decode mode</span>
            <select class="cg-input" name="decode_mode"><option value="sampling">Sampling</option><option value="greedy">Greedy</option></select>
        </label>
        <label class="cg-row inline"><span class="cg-label">Temperature</span> <input class="cg-input-sm" name="temperature" type="number" min="0" max="2" step="0.05" value="0.7"></label>
        <label class="cg-row inline"><span class="cg-label">Top K</span> <input class="cg-input-sm" name="top_k" type="number" min="0" step="1" value="20"></label>
        <label class="cg-row inline"><span class="cg-label">Top P</span> <input class="cg-input-sm" name="top_p" type="number" min="0.01" max="1" step="0.01" value="0.8"></label>
        <label class="cg-row inline"><span class="cg-label">Repetition penalty</span> <input class="cg-input-sm" name="text_repetition_penalty" type="number" min="1" step="0.05" value="1.05"></label>
        <label class="cg-row inline"><span class="cg-label">Listen scale</span> <input class="cg-input-sm" name="listen_prob_scale" type="number" min="0" step="0.1" value="1"></label>
        <label class="cg-row inline"><span class="cg-label">Initial listen units</span> <input class="cg-input-sm" name="force_listen_count" type="number" min="0" step="1" value="3"></label>
        <p class="cg-label">Applied at session start. Temperature, Top K and Top P apply in Sampling mode.</p></div>`;
    lengthPenalty.closest('.config-group').after(section);
    const notice = document.createElement('p');
    notice.className = 'cg-label';
    notice.hidden = true;
    notice.textContent = 'Length penalty is not supported by this backend.';
    lengthPenalty.closest('.cg-row').after(notice);
    return {
        read() {
            return Object.fromEntries([...section.querySelectorAll('input, select')].map(input => [
                input.name, input.type === 'number' ? Number(input.value) : input.value,
            ]));
        },
        setRunning(running, backendInfo) {
            for (const input of section.querySelectorAll('input, select')) input.disabled = running;
            const unsupported = backendInfo?.ignored_init_fields?.includes('config.length_penalty') ?? false;
            lengthPenalty.disabled = running || unsupported;
            notice.hidden = !unsupported;
        },
    };
}
