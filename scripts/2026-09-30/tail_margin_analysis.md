## 1. The last 0.4 s: difference below the signal, dB

- masked: 19.5–27.5, median 26.4, five lowest [19.5, 20.6, 21.2, 21.8, 22.7]
- exact: 19.9–29.5, median 27.0, five lowest [19.9, 21.6, 22.1, 22.2, 23.2]
- silence: -1.9–24.8, median 10.4, five lowest [-1.9, -1.3, -1.0, -0.9, -0.9]

| utterance | masked: range, mean ± sd | exact: range, mean ± sd | masked − exact | silence |
|---|---|---|---|---|
| 121-127105-0003 | 26.4–27.2, 27.0 ± 0.3 | 27.4–29.1, 28.2 ± 0.6 | -1.9 to -0.9 | -1.9–-0.6 |
| 121-127105-0015 | 19.5–23.6, 21.6 ± 1.5 | 19.9–24.0, 22.2 ± 1.4 | -1.0 to -0.4 | 10.4–10.9 |
| 121-127105-0024 | 25.1–25.9, 25.4 ± 0.3 | 25.1–25.9, 25.4 ± 0.3 | -0.2 to +0.3 | 8.4–8.5 |
| 260-123286-0014 | 26.9–27.5, 27.1 ± 0.2 | 26.9–27.3, 27.1 ± 0.1 | +0.0 to +0.2 | 17.7–17.8 |
| 260-123440-0002 | 24.3–26.2, 25.4 ± 0.7 | 24.7–27.0, 26.0 ± 0.8 | -0.8 to -0.4 | 23.4–24.8 |
| 260-123440-0010 | 27.1–27.4, 27.2 ± 0.1 | 29.1–29.5, 29.3 ± 0.1 | -2.2 to -1.7 | 2.4–2.5 |

## 2. 121-127105-0015: the loudest 20 ms frame of the window

| reference | variant | window margin | frame (s before the end) | frame signal dBFS | share of signal energy | share of difference energy | frame margin |
|---|---|---|---|---|---|---|---|
| cosyvoice2_streaming_ref | masked | 21.2 | 0.38–0.36 | -58 | 86 % | 81 % | 22 |
| cosyvoice2_streaming_ref | exact | 22.1 | 0.38–0.36 | -58 | 86 % | 82 % | 22 |
| ref_stream_seed1 | masked | 23.6 | 0.38–0.36 | -58 | 87 % | 81 % | 24 |
| ref_stream_seed1 | exact | 24.0 | 0.38–0.36 | -58 | 87 % | 87 % | 24 |
| ref_stream_seed2 | masked | 19.5 | 0.38–0.36 | -59 | 85 % | 90 % | 19 |
| ref_stream_seed2 | exact | 19.9 | 0.38–0.36 | -59 | 85 % | 92 % | 19 |
| ref_stream_seed3 | masked | 20.6 | 0.38–0.36 | -59 | 86 % | 94 % | 20 |
| ref_stream_seed3 | exact | 21.6 | 0.38–0.36 | -59 | 86 % | 97 % | 21 |
| ref_stream_seed4 | masked | 21.8 | 0.38–0.36 | -58 | 87 % | 93 % | 22 |
| ref_stream_seed4 | exact | 22.2 | 0.38–0.36 | -58 | 87 % | 97 % | 22 |
| ref_stream_seed5 | masked | 22.7 | 0.38–0.36 | -58 | 86 % | 89 % | 22 |
| ref_stream_seed5 | exact | 23.2 | 0.38–0.36 | -58 | 86 % | 94 % | 23 |

## 3. How far back the silence padding reaches

- 121-127105-0003: its difference first exceeds the masked one's by 3 dB 160, 180 ms before the end
- 121-127105-0015: its difference first exceeds the masked one's by 3 dB 160, 180 ms before the end
- 121-127105-0024: its difference first exceeds the masked one's by 3 dB 120 ms before the end
- 260-123286-0014: its difference first exceeds the masked one's by 3 dB 200 ms before the end
- 260-123440-0002: its difference first exceeds the masked one's by 3 dB 180 ms before the end
- 260-123440-0010: its difference first exceeds the masked one's by 3 dB 180 ms before the end

## 4. Shorter windows (from the 20 ms frames), dB

| variant | last 0.4 s | last 0.2 s | last 0.1 s |
|---|---|---|---|
| masked | 19.4 to 27.4 | 19.8 to 29.0 | 22.0 to 29.0 |
| silence | -1.9 to 24.8 | -2.1 to 15.1 | -3.1 to 0.4 |
