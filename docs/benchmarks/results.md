| engine | rate (req/s) | tok/s | TTFT p50 | TTFT p99 | ITL p99 | attained |
| --- | --- | --- | --- | --- | --- | --- |
| nanoserve | 1 | 213 | 20 ms | 23 ms | 38 ms | 0.95 |
| nanoserve | 2 | 406 | 21 ms | 23 ms | 41 ms | 1.82 |
| nanoserve | 4 | 731 | 21 ms | 22 ms | 42 ms | 3.27 |
| nanoserve | 8 | 1233 | 21 ms | 24 ms | 61 ms | 5.52 |
| nanoserve | 16 | 1683 | 21 ms | 2943 ms | 59 ms | 7.53 |
| nanoserve | 32 | 1808 | 449 ms | 7242 ms | 59 ms | 8.09 |
| vLLM | 1 | 224 | 27675 ms | 42494 ms | 1 ms | 1.00 |
| vLLM | 2 | 444 | 679 ms | 8492 ms | 3 ms | 1.99 |
| vLLM | 4 | 877 | 10 ms | 1400 ms | 6 ms | 3.92 |
| vLLM | 8 | 1707 | 10 ms | 101 ms | 7 ms | 7.64 |
| vLLM | 16 | 3208 | 11 ms | 24 ms | 7 ms | 14.35 |
| vLLM | 32 | 5438 | 12 ms | 22 ms | 8 ms | 24.33 |

| engine | offline tok/s |
| --- | --- |
| nanoserve | 1310 |
| vLLM | 6455 |
| HF (static) | 421 |
