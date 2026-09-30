| utterance | audio s | tokens | chunks | first chunk tokens | until it starts s (text + LLM) | its flow s | its CFM s | its HiFT s | first audio s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | 4 | 32 | 0.441 | 0.866 | 0.682 | 0.123 | **1.430** | 6.736 | 0.791 |
| zero_shot_121-127105-0015 | 3.80 | 95 | 3 | 32 | 0.441 | 0.869 | 0.683 | 0.122 | **1.432** | 4.192 | 1.103 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 5 | 32 | 0.434 | 0.864 | 0.676 | 0.121 | **1.420** | 11.379 | 0.820 |
| zero_shot_260-123286-0014 | 3.00 | 75 | 2 | 25 | 0.370 | 0.823 | 0.676 | 0.120 | **1.314** | 2.716 | 0.905 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 5 | 25 | 0.380 | 0.872 | 0.684 | 0.121 | **1.374** | 10.018 | 0.790 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 4 | 25 | 0.382 | 0.880 | 0.689 | 0.122 | **1.384** | 6.750 | 0.835 |

Distinct utterances: 6, audio 49.96 s, wall 41.79 s, aggregate RTF 0.836, worst 1.103. Config: reported, HiFT F0/source float32, bucketing True, warm-up buckets (184.3 s), seed 1986. Streaming: first audio 1.313-1.432 s; streaming warm-up 149.4 s.
