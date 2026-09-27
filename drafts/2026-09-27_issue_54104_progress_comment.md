Progress update on CosyVoice2. Work is in draft PR #56651 (branch `bringup/cosyvoice2-istft`).

**Built and verified on N150 (Wormhole), with the real CosyVoice2-0.5B weights where noted**
- **iSTFT** as a matmul inverse DFT plus windowed overlap-add. Device output (bf16 inputs) matches `torch.istft` at PCC ≥ 0.999 from 256 to 65,537 frames.
- **HiFT vocoder** (3-stage upsample and resblocks, F0 predictor, NSF source, iSTFT head). Real weights: decode PCC ≥ 0.999 (fp32), F0 predictor 0.996–0.999. Along the way we found and fixed a silent STFT corruption at ≥ 65,536 samples.
- **Qwen2-0.5B LLM** on tt_transformers, with CosyVoice2 sequence assembly and RAS sampling. The traced decode runs at 9.8 ms/token (41 ms eager) and matches eager bit for bit.
- **Flow decoder:** Conformer encoder and CFM estimator (10 Euler steps, classifier-free guidance), traced solver, and chunk-causal streaming masks with bucketing. With the real `flow.pt`, the bucketed and exact-length streaming solves are bit-identical at six (valid length, bucket) pairs, with buckets up to 1,536 mel frames, and within PCC 0.9962–0.9988 of the fp32 torch reference.
- **End-to-end zero-shot synthesis** (non-streaming) runs on N150.
  - Quality: WER 4.17% and speaker similarity 0.889 on one LibriSpeech prompt/target pair. On four more sentences, WER 0–2.8% and similarity 0.56–0.87 (Whisper `base.en`, CAM++ cosine).
  - Speed: RTF 0.43–0.52 for 4.4–11.4 s of audio, measured by repeating the same request in-process with the LLM decode and CFM traced. RTF on distinct utterances isn't measured yet.
- 176 tests: 175 pass on N150, and 1 is opt-in.

**In progress:** the streaming pipeline. That means incremental LLM generation, streaming flow calls (the chunk-causal encoder and CFM modes are built, including the final, non-aligned chunk), and HiFT streaming (mel and source caches plus crossfade, which is where the chunked iSTFT overlap-add lives). After that, measured TTFP and RTF.

**Three questions**
1. **How will Stage 3 be judged?** From our measured component costs, at 10 Euler steps the CFM and LLM decode alone come to a streaming RTF of about 0.46–0.58. The first chunk takes about 0.67–0.9 s before the encoder and vocoder even run. So TTFP < 500 ms and RTF < 0.4 look out of reach without cutting steps, which changes the output. CosyVoice1 (#52540) was accepted with measured, partly met Stage 3 targets. Would a similar measured report be acceptable here, along with whatever step-count trade-off we can validate on WER? And is there a preferred definition of warm vs cold, and of which utterances to use?
2. **Where should the model live:** `models/experimental/cosyvoice2`, like CosyVoice1, or `models/demos/audio/cosyvoice2`, next to Whisper?
3. **What would you consider a representative WER / speaker-similarity evaluation set?** For example, how many LibriSpeech test-clean utterances, and which ASR and speaker-verification models?
