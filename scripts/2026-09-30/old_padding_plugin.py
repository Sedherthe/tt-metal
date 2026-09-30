"""pytest plugin: run the streaming tests on the final HiFT call as it was before `ed1c3ad1c5`, padded at its end with
silence mel, with no masks and no end gain. It is for showing that the stage A end gate fails on the old padding.

`ed1c3ad1c5` changed only `HiFTStream.step` in `tt/streaming.py`. The generator masks only when that step passes
`valid_frames`, and the old step never does. So this takes the old method's source verbatim from `6a2ab97dde` (the
commit before the fix), compiles it in the current module's namespace, and installs it on `HiFTStream`. Nothing in
the PR changes.

    PYTHONPATH=<this dir>:$PYTHONPATH pytest -p old_padding_plugin <streaming test>
"""
import ast
import subprocess
import textwrap

BEFORE_FIX = "6a2ab97dde"


def pytest_configure(config):
    from models.experimental.cosyvoice2.tt import streaming as st

    src = subprocess.check_output(
        ["git", "-C", "/home/user/tt-metal", "show", f"{BEFORE_FIX}:models/experimental/cosyvoice2/tt/streaming.py"],
        text=True,
    )
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "HiFTStream")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    body = textwrap.dedent(ast.get_source_segment(src, fn, padded=True))
    assert "valid_frames" not in body and "silence_mel(back)" in body, "not the pre-fix step"
    namespace = vars(st)
    exec(compile(body, f"{BEFORE_FIX}:tt/streaming.py", "exec"), namespace)  # noqa: S102
    st.HiFTStream.step = namespace.pop("step")
    print(f"old_padding_plugin: HiFTStream.step is {BEFORE_FIX}'s (silence end padding, no masks, no end gain)")
