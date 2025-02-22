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


def xtime(a: int) -> int:
    return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


def xtime_tensor(a: Tensor) -> Tensor:
    high_bit_mask = a.bitwise_and(0x80)
    shifted = a.lshift(1)

    condition = high_bit_mask != 0
    result = condition.where(shifted.xor(0x1B), shifted)

    return result.bitwise_and(0xFF)


def text2matrix(text: int) -> Tensor:
    return (
        Tensor([text >> (8 * (15 - i)) for i in range(16)], dtype=dtypes.uint64)
        .bitwise_and(0xFF)
        .reshape((4, 4))
    )


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
        self.plain_state = self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.plain_state = self.__round_encrypt(
                self.plain_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

        self.plain_state = self.__sub_bytes(self.plain_state)
        self.plain_state = self.__shift_rows(self.plain_state)
        self.plain_state = self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    def decrypt(self, ciphertext: int) -> int:
        self.cipher_state = text2matrix(ciphertext)
        self.cipher_state = self.__add_round_key(
            self.cipher_state, self.round_keys[40:]
        )
        self.cipher_state = self.__inv_shift_rows(self.cipher_state)
        self.cipher_state = self.__inv_sub_bytes(self.cipher_state)
        for i in range(9, 0, -1):
            self.cipher_state = self.__round_decrypt(
                self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

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
        return Sbox[s]

    def __inv_sub_bytes(self, s: Tensor) -> Tensor:
        return InvSbox[s]

    def __shift_rows(self, s: Tensor) -> Tensor:
        state = s.clone()

        for i in range(1, 4):
            state[:, i] = state[:, i].roll(-i, dims=0)

        return state

    def __inv_shift_rows(self, s: Tensor) -> Tensor:
        state = s.clone()

        for i in range(1, 4):
            state[:, i] = state[:, i].roll(i, dims=0)

        return state

    def __mix_columns(self, state: Tensor) -> Tensor:
        t = state[:, 0].xor(state[:, 1]).xor(state[:, 2]).xor(state[:, 3])
        xtimes = xtime_tensor(state.roll(-1, dims=1).xor(state))
        state = state.xor(t.unsqueeze(1)).xor(xtimes)

        return state

    def __inv_mix_columns(self, state: Tensor) -> Tensor:
        u = xtime_tensor(xtime_tensor(state[:, 0].xor(state[:, 2])))
        v = xtime_tensor(xtime_tensor(state[:, 1].xor(state[:, 3])))

        out = state.clone()
        out[:, 0] = state[:, 0].xor(u)
        out[:, 1] = state[:, 1].xor(v)
        out[:, 2] = state[:, 2].xor(u)
        out[:, 3] = state[:, 3].xor(v)

        return self.__mix_columns(out)


if __name__ == "__main__":
    aes = AES(0x2B7E151628AED2A6ABF7158809CF4F3C)
    print(hex(aes.encrypt(0x3243F6A8885A308D313198A2E0370734)))
    print(hex(aes.decrypt(0x3925841D02DC09FBDC118597196A0B32)))
