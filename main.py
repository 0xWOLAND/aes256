from tinygrad.tensor import Tensor
from tinygrad import dtypes

Sbox = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

Rcon = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
]

def xtime(a: int) -> int:
    return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

def xtime_tensor(a: Tensor) -> Tensor:
    high_bits = (a.bitwise_and(0x80)).cast(dtypes.uint64)
    shifted = (a.lshift(1).bitwise_and(0xFF))
    return (shifted.xor(high_bits * 0x1B))


def text2matrix(text: int) -> Tensor:
    t_120 = Tensor.full((16,), 120, dtype=dtypes.uint64)
    t_8 = Tensor.full((16,), 8, dtype=dtypes.uint64)
    t_1 = Tensor.arange(16, dtype=dtypes.uint64)
    t_shift = t_120 - t_8 * t_1
    t_divisor = 2 ** t_shift
    t_x = Tensor.full((16,), text, dtype=dtypes.uint64)
    out = t_x.div(t_divisor).cast(dtypes.uint64).bitwise_and(0xFF)
    return out.reshape(4, 4)

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
        key_matrix = text2matrix(master_key)
        self.round_keys = Tensor.zeros((4, 44), dtype=dtypes.uint64).contiguous()
        self.round_keys[:, :4] = key_matrix
        for i in range(4, 44):
            if i % 4 == 0:
                temp = self.round_keys[:, i-1]
                temp = Tensor([Sbox[x] for x in temp.roll(-1, dims=0).numpy().astype(int)], dtype=dtypes.uint64)
                rcon = Tensor([Rcon[i // 4], 0, 0, 0], dtype=dtypes.uint64)
                self.round_keys[:, i] = self.round_keys[:, i-4].xor(temp).xor(rcon)
            else:
                self.round_keys[:, i] = self.round_keys[:, i-1].xor(self.round_keys[:, i-4])

    def encrypt(self, plaintext: int) -> int:
        self.plain_state = text2matrix(plaintext)

        self.plain_state = self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.plain_state = self.__round_encrypt(self.plain_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.plain_state = self.__sub_bytes(self.plain_state)
        self.plain_state = self.__shift_rows(self.plain_state)
        self.plain_state = self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext: int) -> int:
        self.cipher_state = text2matrix(ciphertext)

        self.cipher_state = self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.cipher_state = self.__inv_shift_rows(self.cipher_state)
        self.cipher_state = self.__inv_sub_bytes(self.cipher_state)

        for i in range(9, 0, -1):
            self.cipher_state = self.__round_decrypt(self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)])

        self.cipher_state = self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)
            

    def __round_encrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        state_matrix = self.__sub_bytes(state_matrix)
        state_matrix = self.__shift_rows(state_matrix)
        state_matrix = self.__mix_columns(state_matrix)
        state_matrix = self.__add_round_key(state_matrix, key_matrix)
        return state_matrix

    def __round_decrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        state_matrix = self.__add_round_key(state_matrix, key_matrix)
        state_matrix = self.__inv_mix_columns(state_matrix)
        state_matrix = self.__inv_shift_rows(state_matrix)
        state_matrix = self.__inv_sub_bytes(state_matrix)
        return state_matrix

    def __add_round_key(self, s: Tensor, k: Tensor) -> Tensor:
        return s.xor(k)
    
    def __sub_bytes(self, s: Tensor) -> Tensor:
        return Tensor([Sbox[int(x)] for x in s.flatten()], dtype=dtypes.uint64).reshape(4, 4)
    
    def __inv_sub_bytes(self, s: Tensor) -> Tensor:
        return Tensor([Sbox.index(int(x)) for x in s.flatten()], dtype=dtypes.uint64).reshape(4, 4)
    
    def __shift_rows(self, s: Tensor) -> Tensor:
        state = s.clone()

        for i in range(1, 4):
            state[i] = state[i].roll(-i, dims=0)
        
        return state

    def __inv_shift_rows(self, s: Tensor) -> Tensor:
        state = s.clone()

        for i in range(1, 4):
            state[i] = state[i].roll(i, dims=0)
        
        return state

    def __mix_columns(self, state: Tensor) -> Tensor:
        shifted = state.roll(1, axis=0)
        t = state.reduce_xor(axis=0)
        xt_pairs = xtime_tensor(state ^ shifted)
        result = state ^ t.reshape(1, 4) ^ xt_pairs

        return result
    
    def __inv_mix_columns(self, state: Tensor) -> Tensor:
        u_v = xtime_tensor(xtime_tensor(
            state[::2] ^ state[1::2]
        )).repeat_interleave(2, dim=0)
        
        state = state ^ u_v
        return self.__mix_columns(state.numpy().tolist())
    
if __name__ == "__main__":
    aes = AES(0x2b7e151628aed2a6abf7158809cf4f3c)
    print(aes.encrypt(0x3243f6a8885a308d313198a2e0370734))
    print(aes.decrypt(0x3925841d02dc09fbdc118597196a0b32))