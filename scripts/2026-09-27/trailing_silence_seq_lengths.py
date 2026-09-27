"""R7/R8: total LLM sequence length of the 2026-09-23 trailing-silence clips vs the max_seq_len=512 the scripts
hardcoded. CPU only. Prompt = LibriSpeech dummy validation item 0 (text capitalized + "."), speech tokens at 25 Hz;
targets = the four-sentence set; TT generated counts from the 09-23 audio lengths (x 25 tokens/s)."""
import shutil, tempfile
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

d = tempfile.mkdtemp()
for fn in ("tokenizer_config.json", "vocab.json", "merges.txt"):
    shutil.copy(hf_hub_download("FunAudioLLM/CosyVoice2-0.5B", f"CosyVoice-BlankEN/{fn}"), f"{d}/{fn}")
tok = AutoTokenizer.from_pretrained(d)
ref = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)[0]
wav, sr = ref["audio"]["array"], ref["audio"]["sampling_rate"]
P = len(tok.encode(ref["text"].capitalize() + "."))
S = min((1 + int(len(wav) / sr * 24000) // 480) // 2, (len(wav) // 160) // 4)
TEXTS = ["Please close the door when you leave.",
         "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch.",
         "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays.",
         "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day."]
for i, (t, g) in enumerate(zip(TEXTS, [109, 158, 184, 284]), 1):
    T = len(tok.encode(t)); L = 1 + P + T + 1 + S
    print(f"clip {i}: prefix {L}, worst case {L + 20 * T}, 09-23 total {L + g} (512 - total = {512 - (L + g)})")
