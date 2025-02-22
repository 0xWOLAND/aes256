

import pytest
from aes import AES

class TestAESEncryptDecrypt:
    def setup_method(self):
        # Standard test vector from NIST FIPS-197
        self.key = 0x2b7e151628aed2a6abf7158809cf4f3c
        self.plaintext = 0x3243f6a8885a308d313198a2e0370734
        self.ciphertext = 0x3925841d02dc09fbdc118597196a0b32
        self.aes = AES(self.key)

    def test_encrypt(self):
        """Test basic encryption with NIST test vector"""
        encrypted = self.aes.encrypt(self.plaintext)
        assert encrypted == self.ciphertext

    def test_decrypt(self):
        """Test basic decryption with NIST test vector"""
        decrypted = self.aes.decrypt(self.ciphertext)
        assert decrypted == self.plaintext

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption followed by decryption returns the original plaintext"""
        encrypted = self.aes.encrypt(self.plaintext)
        decrypted = self.aes.decrypt(encrypted)
        assert decrypted == self.plaintext

    @pytest.mark.parametrize("plaintext,key,ciphertext", [
        # NIST SP 800-38A test vectors
        (0x6bc1bee22e409f96e93d7e117393172a,
         0x2b7e151628aed2a6abf7158809cf4f3c,
         0x3ad77bb40d7a3660a89ecaf32466ef97),
        (0xae2d8a571e03ac9c9eb76fac45af8e51,
         0x2b7e151628aed2a6abf7158809cf4f3c,
         0xf5d3d58503b9699de785895a96fdbaaf),
        # Additional test with all zeros
        (0x00000000000000000000000000000000,
         0x00000000000000000000000000000000,
         0x66e94bd4ef8a2c3b884cfa59ca342b2e)
    ])
    def test_known_vectors(self, plaintext, key, ciphertext):
        """Test encryption and decryption with additional known test vectors"""
        aes = AES(key)
        assert aes.encrypt(plaintext) == ciphertext
        assert aes.decrypt(ciphertext) == plaintext

if __name__ == "__main__":
    pytest.main([__file__])