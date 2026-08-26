| engine | rate (req/s) | tok/s | TTFT p50 | TTFT p99 | ITL p99 | attained |
| --- | --- | --- | --- | --- | --- | --- |
| nanoserve | 1 | 211 | 25 ms | 27 ms | 48 ms | 0.94 |
| nanoserve | 2 | 396 | 25 ms | 27 ms | 49 ms | 1.77 |
| nanoserve | 4 | 708 | 25 ms | 27 ms | 50 ms | 3.17 |
| nanoserve | 8 | 1164 | 25 ms | 88 ms | 74 ms | 5.21 |
| nanoserve | 16 | 1418 | 26 ms | 5360 ms | 74 ms | 6.34 |
| nanoserve | 32 | 1513 | 974 ms | 9717 ms | 65 ms | 6.77 |
| vLLM | 1 | 224 | 27731 ms | 42927 ms | 1 ms | 1.00 |
| vLLM | 2 | 444 | 562 ms | 7985 ms | 3 ms | 1.99 |
| vLLM | 4 | 877 | 13 ms | 936 ms | 3 ms | 3.92 |
| vLLM | 8 | 1706 | 13 ms | 118 ms | 9 ms | 7.63 |
| vLLM | 16 | 3198 | 15 ms | 27 ms | 10 ms | 14.31 |
| vLLM | 32 | 5385 | 15 ms | 29 ms | 11 ms | 24.09 |

| engine | offline tok/s |
| --- | --- |
| nanoserve | 1132 |
| vLLM | 6146 |
| HF (static) | 393 |
