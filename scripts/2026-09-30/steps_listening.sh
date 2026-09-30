#!/bin/bash
# The step sweep's wavs for listening: noise seed 1, every utterance, TT and the reference, Stage 1 and streaming,
# at 10 (D43's draws), 8, 6 and 5 Euler steps: ~/listening/steps/<mode>/<utterance>/<side>_k<steps>.wav
D=/home/user/data/cosyvoice2_steps
D43=/home/user/data/cosyvoice2_draws
L=/home/user/listening/steps
for mode in stage1 stream; do
  for side in tt ref; do
    for k in 10 8 6 5; do
      src=$D/steps$k/${side}_${mode}_seed1
      [ "$k" = 10 ] && src=$D43/${side}_${mode}_seed1
      for w in $src/*.wav; do
        u=$(basename "$w" .wav); u=${u#zero_shot_}
        mkdir -p "$L/$mode/$u"
        cp "$w" "$L/$mode/$u/${side}_k$k.wav"
      done
    done
  done
done
find "$L" -name "*.wav" | wc -l
