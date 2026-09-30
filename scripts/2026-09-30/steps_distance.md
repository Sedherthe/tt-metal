Whole-utterance log-mel L1, mean (max) over six utterances x five draws.

| Euler steps | TT Stage 1 vs its 10 steps | reference Stage 1 vs its 10 steps | TT streaming vs its 10 steps | reference streaming vs its 10 steps | TT streaming vs upstream's, same steps |
|---|---|---|---|---|---|
| 10 | — | — | — | — | 0.102 (max 0.113) |
| 8 | 0.110 (max 0.143) | 0.101 (max 0.124) | 0.123 (max 0.169) | 0.114 (max 0.160) | 0.104 (max 0.124) |
| 6 | 0.162 (max 0.220) | 0.140 (max 0.167) | 0.176 (max 0.250) | 0.169 (max 0.245) | 0.100 (max 0.114) |
| 5 | 0.187 (max 0.240) | 0.167 (max 0.216) | 0.189 (max 0.222) | 0.181 (max 0.223) | 0.110 (max 0.123) |

For scale, two noise draws of the same side at 10 steps (seed 1 against seeds 2-5), mean (max):

| TT Stage 1 | reference Stage 1 | TT streaming | reference streaming |
|---|---|---|---|
| 0.029 (max 0.037) | 0.011 (max 0.016) | 0.027 (max 0.031) | 0.011 (max 0.018) |
