import pytest
from aes256.aes import AES as TinyGradAES
from tests.reference.aes import AES as ReferenceAES


@pytest.mark.benchmark
@pytest.mark.parametrize("num_ops", [2, 4, 8])
@pytest.mark.parametrize(
    "aes_class", [TinyGradAES, ReferenceAES], ids=["TinyGradAES", "ReferenceAES"]
)
def test_aes_performance(benchmark, aes_class, num_ops):
    key = 0x2B7E151628AED2A6ABF7158809CF4F3C
    data = 0x3243F6A8885A308D313198A2E0370734

    aes = aes_class(key)

    def aes_ops():
        for _ in range(num_ops):
            c = aes.encrypt(data)
            p = aes.decrypt(c)

    benchmark.pedantic(aes_ops, rounds=1, iterations=1)


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
