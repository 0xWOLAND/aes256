import pytest
import random
from reference_aes import AES as ReferenceAES
from aes import AES as TinyGradAES

def generate_random_128bit():
    return random.getrandbits(128)

class TestAESImplementations:
    def setup_method(self):
        self.key = generate_random_128bit()
        self.plaintext = generate_random_128bit()
        self.ref_aes = ReferenceAES(self.key)
        self.tiny_aes = TinyGradAES(self.key)

    def test_encryption_comparison(self):
        ref_ciphertext = self.ref_aes.encrypt(self.plaintext)
        tiny_ciphertext = self.tiny_aes.encrypt(self.plaintext)
        assert ref_ciphertext == tiny_ciphertext

    def test_decryption_comparison(self):
        ref_ciphertext = self.ref_aes.encrypt(self.plaintext)
        tiny_decrypted = self.tiny_aes.decrypt(ref_ciphertext)
        assert tiny_decrypted == self.plaintext
        
        tiny_ciphertext = self.tiny_aes.encrypt(self.plaintext)
        ref_decrypted = self.ref_aes.decrypt(tiny_ciphertext)
        assert ref_decrypted == self.plaintext

    def test_multiple_random_values(self):
        for _ in range(5):
            plaintext = generate_random_128bit()
            ref_ciphertext = self.ref_aes.encrypt(plaintext)
            tiny_ciphertext = self.tiny_aes.encrypt(plaintext)
            assert ref_ciphertext == tiny_ciphertext
            
            ref_decrypted = self.ref_aes.decrypt(tiny_ciphertext)
            tiny_decrypted = self.tiny_aes.decrypt(ref_ciphertext)
            assert ref_decrypted == plaintext and tiny_decrypted == plaintext

if __name__ == "__main__":
    pytest.main([__file__, "-v"])