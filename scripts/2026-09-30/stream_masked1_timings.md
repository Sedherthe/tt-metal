| utterance | audio s | tokens | chunks | first chunk tokens | until it starts s (text + LLM) | its flow s | its CFM s | its HiFT s | first audio s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | 4 | 32 | 0.473 | 0.891 | 0.714 | 0.127 | **1.491** | 6.939 | 0.814 |
| zero_shot_121-127105-0015 | 3.80 | 95 | 3 | 32 | 0.464 | 0.912 | 0.739 | 0.126 | **1.502** | 4.261 | 1.121 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 5 | 32 | 0.459 | 0.843 | 0.691 | 0.121 | **1.423** | 11.378 | 0.820 |
| zero_shot_260-123286-0014 | 3.00 | 75 | 2 | 25 | 0.383 | 0.865 | 0.687 | 0.124 | **1.373** | 2.872 | 0.957 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 5 | 25 | 0.389 | 0.875 | 0.692 | 0.123 | **1.387** | 10.310 | 0.813 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 4 | 25 | 0.384 | 0.848 | 0.677 | 0.120 | **1.353** | 6.780 | 0.839 |

Distinct utterances: 6, audio 49.96 s, wall 42.54 s, aggregate RTF 0.851, worst 1.121. Config: reported, HiFT F0/source float32, bucketing True, warm-up buckets (184.6 s), seed 1986. Streaming: first audio 1.353-1.502 s; streaming warm-up 149.0 s.
