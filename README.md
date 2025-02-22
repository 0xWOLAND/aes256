# TinyGrad AES 

A TinyGrad-based implementation of the Advanced Encryption Standard (AES) algorithm. This implementation is based on [bozhu's Python AES implementation](https://github.com/bozhu/AES-Python) but rewritten to use TinyGrad tensors for computation.

## Benchmarks

The implementation includes benchmarks comparing the TinyGrad-based implementation with a pure Python reference implementation. Here are the results:

| Implementation | Operations | Time (μs) | Operations/sec |
|---------------|------------|-----------|----------------|
| Reference     | 2          | 178.37    | 5,606.17      |
| Reference     | 4          | 304.33    | 3,285.87      |
| Reference     | 8          | 602.62    | 1,659.41      |
| TinyGrad      | 2          | 616,860.92| 1.62          |
| TinyGrad      | 4          | 901,575.75| 1.11          |
| TinyGrad      | 8          | 1,828,307.83| 0.55        |

Each operation consists of one encryption followed by one decryption. The benchmark measures the time taken for different numbers of operations (2, 4, and 8).
