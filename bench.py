import random
import pytest
from aes256.aes import AES as TinyGradAES
from tests.reference.aes import AES as ReferenceAES


@pytest.mark.benchmark
@pytest.mark.parametrize("num_ops", [1, 100, 1000, 10000])
@pytest.mark.parametrize(
    "aes_class", [TinyGradAES, ReferenceAES], ids=["TinyGradAES", "ReferenceAES"]
)
def test_aes_performance(benchmark, aes_class, num_ops):
    key = random.randint(0, 2**128)
    data = random.randint(0, 2**128)

    aes = aes_class(key)

    def aes_ops():
        for _ in range(num_ops):
            c = aes.encrypt(data)
            __ = aes.decrypt(c)

    benchmark.pedantic(aes_ops, rounds=10, iterations=10)


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
