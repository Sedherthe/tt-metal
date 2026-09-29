| utterance | audio s | tokens | chunks | first chunk tokens | until it starts s (text + LLM) | its flow s | its CFM s | its HiFT s | first audio s | wall s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot_121-127105-0003 | 8.52 | 213 | 4 | 32 | 0.466 | 0.865 | 0.687 | 0.124 | **1.455** | 6.869 | 0.806 |
| zero_shot_121-127105-0015 | 3.80 | 95 | 3 | 32 | 0.432 | 0.812 | 0.674 | 0.121 | **1.365** | 4.017 | 1.057 |
| zero_shot_121-127105-0024 | 13.88 | 347 | 5 | 32 | 0.455 | 0.861 | 0.686 | 0.123 | **1.439** | 11.585 | 0.835 |
| zero_shot_260-123286-0014 | 3.00 | 75 | 2 | 25 | 0.395 | 0.880 | 0.696 | 0.124 | **1.398** | 2.918 | 0.973 |
| zero_shot_260-123440-0002 | 12.68 | 317 | 5 | 25 | 0.393 | 0.894 | 0.710 | 0.126 | **1.413** | 10.390 | 0.819 |
| zero_shot_260-123440-0010 | 8.08 | 202 | 4 | 25 | 0.386 | 0.894 | 0.707 | 0.124 | **1.403** | 6.847 | 0.847 |

Distinct utterances: 6, audio 49.96 s, wall 42.63 s, aggregate RTF 0.853, worst 1.057. Config: reported, HiFT F0/source float32, bucketing True, warm-up buckets (186.1 s), seed 1986. Streaming: first audio 1.365-1.455 s; streaming warm-up 149.2 s.
