# TinyGrad `aes256` [![Tests](https://github.com/tinycrypto/aes256/actions/workflows/ci.yml/badge.svg)](https://github.com/tinycrypto/aes256/actions/workflows/ci.yml)

A TinyGrad-based implementation of the Advanced Encryption Standard (AES) algorithm. This implementation is based on [bozhu's Python AES implementation](https://github.com/bozhu/AES-Python) but rewritten to use TinyGrad tensors for computation.

## Usage

Install dependencies:
```bash
uv sync
```

Run tests: 
```bash
uv run pytest
```  

Run benchmarks:
 ```bash
uv run pytest tests/bench.py
```

## Benchmark Results

| Implementation | Operations | Time (μs) | Operations/sec |
|---------------|------------|-----------|----------------|
| Reference     | 2          | 178.37    | 5,606.17      |
| Reference     | 4          | 304.33    | 3,285.87      |
| Reference     | 8          | 602.62    | 1,659.41      |
| TinyGrad      | 2          | 616,860.92| 1.62          |
| TinyGrad      | 4          | 901,575.75| 1.11          |
| TinyGrad      | 8          | 1,828,307.83| 0.55        |

Each operation is one encryption + decryption. The TinyGrad implementation is slower due to tensor operation overhead.
