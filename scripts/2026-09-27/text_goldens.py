"""Upstream's own CosyVoiceFrontEnd.text_normalize (the real method, bound to a stand-in holding only what the
English path reads) plus CosyVoice2Tokenizer.encode, over hand-picked inputs. Prints JSON for the host test."""
import json, sys, types
sys.path.insert(0, "/home/user/tt-metal/models/demos/audio/cosyvoice2/scripts")
import reference_env
reference_env.setup_upstream()
import inflect
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
tok = get_qwen_tokenizer(reference_env.model_dir() + "/CosyVoice-BlankEN", skip_special_tokens=True)
fe = types.SimpleNamespace(tokenizer=tok, allowed_special="all", text_frontend="", inflect_parser=inflect.engine())
norm = lambda t, split: CosyVoiceFrontEnd.text_normalize(fe, t, split=split)
long = ("It was the white rabbit returning splendidly dressed, with a pair of white kid gloves in one hand and a large "
        "fan in the other. He came trotting along in a great hurry, muttering to himself as he came. Oh the duchess, "
        "the duchess! Oh won't she be savage if I've kept her waiting! Alice felt so desperate that she was ready to "
        "ask help of any one; so, when the rabbit came near her, she began, in a low, timid voice. If you please, sir. "
        "The rabbit started violently. Then he dropped the white kid gloves and the fan, and skurried away into the "
        "darkness as hard as he could go. Short end.")
cases = [
    ("plain", "Truly this sea is of infinite width.", True),
    ("numbers", "In 1987, 3 of the 25 boats sailed 100 miles by 7:45 and 2.5 hours later.", True),
    ("long_split_merge", long, True),
    ("punctuation_only_dropped", "Wait... what?! ... Fine.", True),
    ("quote_after_stop", 'He said "stop." Then he left!', True),
    ("marker_bypass", "<|en|>Hello there 42.", True),
    ("whitespace_no_final_stop", "   no final punctuation here 7   ", True),
    ("prompt_unsplit", "I have 2 cats; they are 11 years old.", False),
]
out = []
for name, text, split in cases:
    got = norm(text, split)
    segs = got if split else [got]
    out.append({"name": name, "text": text, "split": split, "expected": got,
                "ids": [tok.encode(s, allowed_special="all") for s in segs]})
print(json.dumps(out, ensure_ascii=False))
