| utterance | audio s | tokens | chunks | first chunk tokens | until it starts s (text + LLM) | its flow s | its CFM s | its HiFT s | first audio s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | 4 | 32 | 0.445 | 0.848 | 0.696 | 0.125 | **1.418** | 6.916 | 0.812 |
| zero_shot_121-127105-0015 | 3.80 | 95 | 3 | 32 | 0.454 | 0.898 | 0.702 | 0.127 | **1.479** | 4.262 | 1.122 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 5 | 32 | 0.459 | 0.891 | 0.701 | 0.125 | **1.475** | 11.240 | 0.810 |
| zero_shot_260-123286-0014 | 3.00 | 75 | 2 | 25 | 0.393 | 0.880 | 0.692 | 0.126 | **1.399** | 2.835 | 0.945 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 5 | 25 | 0.371 | 0.841 | 0.692 | 0.124 | **1.336** | 9.984 | 0.787 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 4 | 25 | 0.393 | 0.920 | 0.731 | 0.127 | **1.441** | 6.869 | 0.850 |

Distinct utterances: 6, audio 49.96 s, wall 42.11 s, aggregate RTF 0.843, worst 1.122. Config: reported, HiFT F0/source float32, bucketing True, warm-up buckets (186.1 s), seed 1986. Streaming: first audio 1.336-1.479 s; streaming warm-up 149.8 s.
