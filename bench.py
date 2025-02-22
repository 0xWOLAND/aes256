import pytest
from aes import AES as TinyGradAES
from reference_aes import AES as ReferenceAES

@pytest.mark.parametrize("num_ops", [2, 4, 8])
@pytest.mark.parametrize("aes_class", [TinyGradAES, ReferenceAES], ids=["TinyGradAES", "ReferenceAES"])
def test_aes_performance(benchmark, aes_class, num_ops):
    key = 0x2b7e151628aed2a6abf7158809cf4f3c
    data = 0x3243f6a8885a308d313198a2e0370734
    
    aes = aes_class(key)

    def aes_ops():
        for _ in range(num_ops):
            c = aes.encrypt(data)
            p = aes.decrypt(c)
    
    benchmark.pedantic(aes_ops, rounds=1, iterations=1)

if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
