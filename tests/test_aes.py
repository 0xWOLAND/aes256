import pytest
from aes256.aes import AES


class TestAESEncryptDecrypt:
    def setup_method(self):
        # Standard test vector from NIST FIPS-197
        self.key = 0x2B7E151628AED2A6ABF7158809CF4F3C
        self.plaintext = 0x3243F6A8885A308D313198A2E0370734
        self.ciphertext = 0x3925841D02DC09FBDC118597196A0B32
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

    @pytest.mark.parametrize(
        "plaintext,key,ciphertext",
        [
            # NIST SP 800-38A test vectors
            (
                0x6BC1BEE22E409F96E93D7E117393172A,
                0x2B7E151628AED2A6ABF7158809CF4F3C,
                0x3AD77BB40D7A3660A89ECAF32466EF97,
            ),
            (
                0xAE2D8A571E03AC9C9EB76FAC45AF8E51,
                0x2B7E151628AED2A6ABF7158809CF4F3C,
                0xF5D3D58503B9699DE785895A96FDBAAF,
            ),
            # Additional test with all zeros
            (
                0x00000000000000000000000000000000,
                0x00000000000000000000000000000000,
                0x66E94BD4EF8A2C3B884CFA59CA342B2E,
            ),
        ],
    )
    def test_known_vectors(self, plaintext, key, ciphertext):
        """Test encryption and decryption with additional known test vectors"""
        aes = AES(key)
        assert aes.encrypt(plaintext) == ciphertext
        assert aes.decrypt(ciphertext) == plaintext


if __name__ == "__main__":
    pytest.main([__file__])
