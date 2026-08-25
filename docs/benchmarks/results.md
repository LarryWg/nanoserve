| engine | rate (req/s) | tok/s | TTFT p50 | TTFT p99 | ITL p99 | attained |
| --- | --- | --- | --- | --- | --- | --- |
| nanoserve | 1 | 182 | 51 ms | 74 ms | 57 ms | 0.95 |
| nanoserve | 2 | 332 | 54 ms | 79 ms | 66 ms | 1.73 |
| nanoserve | 4 | 548 | 57 ms | 83 ms | 89 ms | 2.86 |
| nanoserve | 8 | 682 | 63 ms | 2194 ms | 99 ms | 3.57 |
| nanoserve | 16 | 720 | 124 ms | 9640 ms | 99 ms | 3.78 |
| nanoserve | 32 | 736 | 1251 ms | 14317 ms | 91 ms | 3.86 |
| vLLM | 1 | 193 | 24 ms | 42 ms | 5 ms | 1.01 |
| vLLM | 2 | 385 | 23 ms | 43 ms | 6 ms | 2.01 |
| vLLM | 4 | 765 | 23 ms | 42 ms | 7 ms | 3.99 |
| vLLM | 8 | 1483 | 25 ms | 50 ms | 8 ms | 7.73 |
| vLLM | 16 | 2754 | 30 ms | 63 ms | 13 ms | 14.37 |
| vLLM | 32 | 4533 | 36 ms | 72 ms | 15 ms | 23.69 |

| engine | offline tok/s |
| --- | --- |
| nanoserve | 1068 |
| vLLM | 5837 |
| HF (static) | 378 |
