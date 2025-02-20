from Crypto.Cipher import AES

KEY = bytes(range(16))
PLAINTEXT = bytes(range(16))

aes = AES.new(KEY, AES.MODE_ECB)
ct = aes.encrypt(PLAINTEXT)

print(ct)



pt = aes.decrypt(ct)
print(pt)

assert pt == PLAINTEXT