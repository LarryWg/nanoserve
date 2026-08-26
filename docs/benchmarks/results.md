| engine | rate (req/s) | tok/s | TTFT p50 | TTFT p99 | ITL p99 | attained |
| --- | --- | --- | --- | --- | --- | --- |
| nanoserve | 1 | 210 | 27 ms | 29 ms | 51 ms | 0.94 |
| nanoserve | 2 | 394 | 27 ms | 29 ms | 52 ms | 1.76 |
| nanoserve | 4 | 697 | 27 ms | 28 ms | 61 ms | 3.12 |
| nanoserve | 8 | 1128 | 27 ms | 353 ms | 79 ms | 5.05 |
| nanoserve | 16 | 1331 | 28 ms | 6512 ms | 79 ms | 5.96 |
| nanoserve | 32 | 1423 | 1210 ms | 10719 ms | 64 ms | 6.37 |
| vLLM | 1 | 223 | 15 ms | 31 ms | 3 ms | 1.00 |
| vLLM | 2 | 444 | 13 ms | 25 ms | 3 ms | 1.99 |
| vLLM | 4 | 877 | 12 ms | 21 ms | 6 ms | 3.92 |
| vLLM | 8 | 1706 | 12 ms | 21 ms | 8 ms | 7.63 |
| vLLM | 16 | 3196 | 13 ms | 24 ms | 9 ms | 14.30 |
| vLLM | 32 | 5369 | 16 ms | 28 ms | 11 ms | 24.02 |

| engine | offline tok/s |
| --- | --- |
| nanoserve | 1040 |
| vLLM | 6018 |
| HF (static) | 375 |
