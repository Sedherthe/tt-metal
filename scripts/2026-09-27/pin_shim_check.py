"""Under transformers 4.51.3: are reference_env's shim 3 (fp32 Qwen2 load) and shim 4 (decode mask) still needed?
Loads upstream with ONLY shims 1-2 (load_wav, pyworld), then checks (a) the Qwen2 backbone's dtype and weights
against llm.pt, and (b) upstream's own forward_one_step decode (its length-1 mask) against a no-cache forward."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/user/tt-metal/models/experimental/cosyvoice2/scripts")
import reference_env as re_
import transformers
print("transformers", transformers.__version__)
repo = re_.upstream_repo()
for p in (repo, os.path.join(repo, "third_party", "Matcha-TTS")):
    sys.path.insert(0, p)
sys.modules.setdefault("pyworld", re_._stub_module("pyworld", "training only"))
import cosyvoice.cli.frontend as frontend, cosyvoice.utils.file_utils as file_utils
file_utils.load_wav = re_._load_wav; frontend.load_wav = re_._load_wav
import cosyvoice.llm.llm as llm_mod
assert not getattr(llm_mod.Qwen2ForCausalLM.from_pretrained, "_cosyvoice2_fp32", False)
assert not getattr(llm_mod.Qwen2Encoder.forward_one_step, "_cosyvoice2_mask", False)
from cosyvoice.cli.cosyvoice import CosyVoice2
m = CosyVoice2(re_.model_dir(), load_jit=False, load_trt=False, fp16=False)
lm = m.model.llm
sd = torch.load(re_.model_dir() + "/llm.pt", map_location="cpu")
params = dict(lm.named_parameters())
print("(a) dtypes:", {p.dtype for p in lm.parameters()}, "| every param equals llm.pt:",
      all(torch.equal(params[k], sd[k]) for k in params if k in sd))
d = np.load("/home/user/data/cosyvoice2_inputs/zero_shot_260-123286-0014.npz")
text = torch.tensor(json.loads(str(d["segment_text_ids_json"]))[0]).reshape(1, -1)
ptext = torch.from_numpy(d["prompt_text_ids"]).long(); pspeech = torch.from_numpy(d["llm_prompt_speech_tokens"]).long()
with torch.inference_mode():
    lm_input = torch.cat([lm.llm_embedding.weight[0].reshape(1, 1, -1), lm.llm.model.model.embed_tokens(torch.cat([ptext, text], 1)),
                          lm.llm_embedding.weight[1].reshape(1, 1, -1), lm.speech_embedding(pspeech)], 1)
    x, cache, toks, inc = lm_input, None, [], []
    for i in range(40):  # upstream's own method and mask, exactly as inference_wrapper calls it
        y, cache = lm.llm.forward_one_step(x, masks=torch.tril(torch.ones((1, x.shape[1], x.shape[1]))).to(torch.bool), cache=cache)
        lg = lm.llm_decoder(y[:, -1]).log_softmax(-1)[0]; inc.append(lg); t = int(lg[:6561].argmax()); toks.append(t)
        x = lm.speech_embedding.weight[t].reshape(1, 1, -1)
    full_in = torch.cat([lm_input, lm.speech_embedding.weight[torch.tensor(toks[:-1])].unsqueeze(0)], 1)
    h = lm.llm.model(inputs_embeds=full_in, output_hidden_states=True, return_dict=True, use_cache=False).hidden_states[-1][0, lm_input.shape[1] - 1:]
    diff = (torch.stack(inc) - lm.llm_decoder(h).log_softmax(-1)).abs().max(-1).values
print(f"(b) upstream decode vs no-cache forward: step0 {diff[0]:.2e}, steps 1-39 max {diff[1:].max():.3e}; tokens {toks[:8]}")
