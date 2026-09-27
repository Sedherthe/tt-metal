"""Does upstream's incremental decode (forward_one_step + DynamicCache) match a full no-cache forward under
transformers 5.12.1? Case 1's real LLM input (corpus case zero_shot_260-123286-0014); greedy for N steps; logits
compared per step against one no-cache forward over the same sequence.

Arm "upstream": a verbatim copy of upstream's Qwen2Encoder.forward_one_step body (cosyvoice/llm/llm.py @ 074ca6dc9e),
with the length-1 decode mask its inference_wrapper passes. Arm "full mask": the same call with the mask spanning
cache + input, which is what scripts/reference_env.py's shim does (commit 0d687d840e). The upstream arm is called
directly, so the shim, now installed by load_upstream(), cannot hide it.

Run in the reference venv: COSYVOICE2_REPO=... HF_HOME=... /bin/python ref_mask_check.py"""
import sys, json, numpy as np, torch
sys.path.insert(0, "/home/user/tt-metal/models/demos/audio/cosyvoice2/scripts")
import reference_env
m = reference_env.load_upstream()
lm = m.model.llm
d = np.load("/home/user/data/cosyvoice2_inputs/zero_shot_260-123286-0014.npz")
text = torch.tensor(json.loads(str(d["segment_text_ids_json"]))[0]).reshape(1, -1)
ptext = torch.from_numpy(d["prompt_text_ids"]).long()
pspeech = torch.from_numpy(d["llm_prompt_speech_tokens"]).long()
with torch.inference_mode():
    t = torch.cat([ptext, text], 1)
    lm_input = torch.cat([lm.llm_embedding.weight[0].reshape(1, 1, -1), lm.llm.model.model.embed_tokens(t),
                          lm.llm_embedding.weight[1].reshape(1, 1, -1), lm.speech_embedding(pspeech)], 1)
    N = 40
    def upstream_forward_one_step(xs, masks, cache=None):  # verbatim body, cosyvoice/llm/llm.py
        input_masks = masks[:, -1, :]
        outs = lm.llm.model(inputs_embeds=xs, attention_mask=input_masks, output_hidden_states=True,
                            return_dict=True, use_cache=True, past_key_values=cache)
        return outs.hidden_states[-1], outs.past_key_values

    def incremental(fix):
        x, cache, toks, logits = lm_input, None, [], []
        for i in range(N):
            masks = torch.tril(torch.ones((1, x.shape[1], x.shape[1]))).to(torch.bool)
            if fix and cache is not None:
                L = cache.get_seq_length() + x.shape[1]
                masks = torch.ones((1, 1, L), dtype=torch.bool)
            y, cache = upstream_forward_one_step(x, masks=masks, cache=cache)
            lg = lm.llm_decoder(y[:, -1]).log_softmax(-1)[0]
            logits.append(lg); tok = int(lg[:6561].argmax()); toks.append(tok)
            x = lm.speech_embedding.weight[tok].reshape(1, 1, -1)
        return toks, torch.stack(logits)
    def full(toks):
        x = torch.cat([lm_input, lm.speech_embedding.weight[torch.tensor(toks[:-1])].unsqueeze(0)], 1)
        out = lm.llm.model(inputs_embeds=x, output_hidden_states=True, return_dict=True, use_cache=False)
        h = out.hidden_states[-1][0, lm_input.shape[1] - 1:]
        return lm.llm_decoder(h).log_softmax(-1)
    for fix in (False, True):
        toks, inc = incremental(fix)
        ful = full(toks)
        diff = (inc - ful).abs().max(dim=-1).values
        eos = inc[:, 6561].exp()
        print(f"{'full mask' if fix else 'upstream'}: step0 max|dlogp| {diff[0]:.2e}, steps1.. max {diff[1:].max():.3e}; "
              f"P(eos) max over {N} steps {eos.max():.3e}; tokens {toks[:12]}...")
