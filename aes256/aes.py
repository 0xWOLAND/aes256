from tinygrad.tensor import Tensor
from tinygrad import dtypes
from aes256.constants import (
    Sbox as Sbox_const,
    InvSbox as InvSbox_const,
    Rcon as Rcon_const,
)

Sbox = Tensor(Sbox_const, dtype=dtypes.uint8)
InvSbox = Tensor(InvSbox_const, dtype=dtypes.uint8)
Rcon = Tensor(Rcon_const, dtype=dtypes.uint8)

def xtime(a: Tensor) -> Tensor:
    shifted = a.lshift(1)
    return (a.bitwise_and(0x80) != 0).where(shifted.xor(0x1B), shifted).cast(dtypes.uint8)


def text2matrix(text: int) -> Tensor:
    return Tensor([text >> (8 * (15 - i)) for i in range(16)], dtype=dtypes.uint8).reshape((4, 4))


def matrix2text(matrix: Tensor) -> int:
    flat = matrix.flatten()
    result = 0
    for i in range(16):
        byte = int(flat[i].item())
        result = (result << 8) | byte
    return result


class AES:
    def __init__(self, master_key):
        self.change_key(master_key)

    def change_key(self, master_key):
        self.round_keys = Tensor.zeros((44, 4), dtype=dtypes.uint8).contiguous()
        self.round_keys[:4] = text2matrix(master_key)
        for i in range(4, 4 * 11):
            if i % 4 == 0:
                self.round_keys[i, 0] = (
                    self.round_keys[i - 4, 0]
                    ^ Sbox[self.round_keys[i - 1, 1].item()]
                    ^ Rcon[i // 4].item()
                )

                shifted_indices = Tensor([2, 3, 0], dtype=dtypes.uint8)
                sboxed = Sbox[self.round_keys[i - 1][shifted_indices]]
                self.round_keys[i, 1:] = self.round_keys[i - 4, 1:].xor(sboxed)
            else:
                self.round_keys[i] = self.round_keys[i - 4].xor(self.round_keys[i - 1])

    def encrypt(self, plaintext: int) -> int:
        self.plain_state = text2matrix(plaintext)
        self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.__round_encrypt(
                self.plain_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext: int) -> int:
        self.cipher_state = text2matrix(ciphertext)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)
        for i in range(9, 0, -1):
            self.__round_decrypt(
                self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

        self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)

    def __round_encrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)


    def __round_decrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)

    def __add_round_key(self, s: Tensor, k: Tensor) -> Tensor:
        s.assign(s.xor(k))

    def __sub_bytes(self, s: Tensor) -> Tensor:
        s.assign(Sbox[s])

    def __inv_sub_bytes(self, s: Tensor) -> Tensor:
        s.assign(InvSbox[s])

    def __shift_rows(self, s: Tensor) -> Tensor:
        _s = s
        for i in range(1, 4):
            _s[:, i] = _s[:, i].roll(-i, dims=0)

        s.assign(_s)


    def __inv_shift_rows(self, s: Tensor) -> Tensor:
        _s = s
        for i in range(1, 4):
            _s[:, i] = _s[:, i].roll(i, dims=0)

        s.assign(_s)

    def __mix_columns(self, s: Tensor) -> Tensor:
        t = s[:, 0].xor(s[:, 1]).xor(s[:, 2]).xor(s[:, 3])
        xtimes = xtime(s.roll(-1, dims=1).xor(s)).contiguous()
        s.assign(s.xor(t.unsqueeze(1)).xor(xtimes))


    def __inv_mix_columns(self, s: Tensor) -> Tensor:
        even_cols = s[:, [0,2]]  
        odd_cols = s[:, [1,3]]   
        
        u = xtime(xtime(even_cols[:,0].xor(even_cols[:,1])))
        v = xtime(xtime(odd_cols[:,0].xor(odd_cols[:,1])))
        
        s[:, [0,2]] = s[:, [0,2]].xor(u.unsqueeze(1))
        s[:, [1,3]] = s[:, [1,3]].xor(v.unsqueeze(1))
        
        self.__mix_columns(s)


if __name__ == "__main__":
    aes = AES(0x2B7E151628AED2A6ABF7158809CF4F3C)
    pt = 0x3243F6A8885A308D313198A2E0370734
    ct = aes.encrypt(pt)
    rec_pt = aes.decrypt(ct)
    print(f'pt: {hex(pt)}')
    print(f'ct: {hex(ct)}')
    print(f'rec_pt: {hex(rec_pt)}')

    assert pt == rec_pt
