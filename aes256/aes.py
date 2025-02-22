from tinygrad.tensor import Tensor
from tinygrad import dtypes
from aes256.constants import (
    Sbox as Sbox_const,
    InvSbox as InvSbox_const,
    Rcon as Rcon_const,
)
from functools import wraps
import time
from collections import defaultdict
import atexit

Sbox = Tensor(Sbox_const, dtype=dtypes.uint8)
InvSbox = Tensor(InvSbox_const, dtype=dtypes.uint8)
Rcon = Tensor(Rcon_const, dtype=dtypes.uint8)

# Add these at the top after imports
timing_stats = defaultdict(lambda: {'total_time': 0, 'calls': 0})

def format_table(headers, rows):
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Create format string for rows
    row_format = '| ' + ' | '.join(f'{{:<{w}}}' for w in widths) + ' |'
    separator = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    
    # Build table
    table = [separator]
    table.append(row_format.format(*headers))
    table.append(separator)
    for row in rows:
        table.append(row_format.format(*row))
    table.append(separator)
    
    return '\n'.join(table)

def print_timing_stats():
    if not timing_stats:
        return
    
    # Prepare table data
    headers = ['Function', 'Avg Time (ms)', 'Calls', 'Total Time (ms)']
    table_data = []
    for func_name, stats in sorted(timing_stats.items()):
        avg_time = (stats['total_time'] * 1000) / stats['calls']
        table_data.append([
            func_name,
            f"{avg_time:.2f}",
            str(stats['calls']),
            f"{(stats['total_time'] * 1000):.2f}"
        ])
    
    print("\nTiming Statistics:")
    print(format_table(headers, table_data))

# Register the printing function to run at exit
atexit.register(print_timing_stats)

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Update statistics
        timing_stats[func.__name__]['total_time'] += end_time - start_time
        timing_stats[func.__name__]['calls'] += 1
        
        return result
    return wrapper

@timing_decorator
def xtime(a: int) -> int:
    return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)


@timing_decorator
def xtime_tensor(a: Tensor) -> Tensor:
    high_bit_mask = a.bitwise_and(0x80)
    shifted = a.lshift(1)

    condition = high_bit_mask != 0
    result = condition.where(shifted.xor(0x1B), shifted)

    return result.bitwise_and(0xFF)


@timing_decorator
def text2matrix(text: int) -> Tensor:
    return (
        Tensor([text >> (8 * (15 - i)) for i in range(16)], dtype=dtypes.uint8)
        .bitwise_and(0xFF)
        .reshape((4, 4))
    )


@timing_decorator
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

    @timing_decorator
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

    @timing_decorator
    def encrypt(self, plaintext: int) -> int:
        self.plain_state = text2matrix(plaintext)
        self.__add_round_key(self.plain_state, self.round_keys[:4])

        for i in range(1, 10):
            self.plain_state = self.__round_encrypt(
                self.plain_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

        self.__sub_bytes(self.plain_state)
        self.plain_state = self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])

        return matrix2text(self.plain_state)

    @timing_decorator
    def decrypt(self, ciphertext: int) -> int:
        self.cipher_state = text2matrix(ciphertext)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)
        for i in range(9, 0, -1):
            self.cipher_state = self.__round_decrypt(
                self.cipher_state, self.round_keys[4 * i : 4 * (i + 1)]
            )

        self.__add_round_key(self.cipher_state, self.round_keys[:4])

        return matrix2text(self.cipher_state)

    @timing_decorator
    def __round_encrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        self.__sub_bytes(state_matrix)
        state_matrix = self.__shift_rows(state_matrix)
        state_matrix = self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)
        return state_matrix

    @timing_decorator
    def __round_decrypt(self, state_matrix: Tensor, key_matrix: Tensor) -> Tensor:
        self.__add_round_key(state_matrix, key_matrix)
        state_matrix = self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)
        return state_matrix

    @timing_decorator
    def __add_round_key(self, s: Tensor, k: Tensor) -> Tensor:
        s.assign(s.xor(k))

    @timing_decorator
    def __sub_bytes(self, s: Tensor) -> Tensor:
        s.assign(Sbox[s])

    @timing_decorator
    def __inv_sub_bytes(self, s: Tensor) -> Tensor:
        s.assign(InvSbox[s])

    @timing_decorator
    def __shift_rows(self, s: Tensor) -> Tensor:
        state = s.clone()

        for i in range(1, 4):
            state[:, i] = state[:, i].roll(-i, dims=0)

        return state

    @timing_decorator
    def __inv_shift_rows(self, s: Tensor) -> Tensor:
        _s = s.contiguous()
        for i in range(1, 4):
            _s[:, i] = _s[:, i].roll(i, dims=0)

        s.assign(_s)

    @timing_decorator
    def __mix_columns(self, state: Tensor) -> Tensor:
        t = state[:, 0].xor(state[:, 1]).xor(state[:, 2]).xor(state[:, 3])
        xtimes = xtime_tensor(state.roll(-1, dims=1).xor(state))
        state = state.xor(t.unsqueeze(1)).xor(xtimes)

        return state

    @timing_decorator
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
    pt = 0x3243F6A8885A308D313198A2E0370734
    ct = aes.encrypt(pt)
    rec_pt = aes.decrypt(ct)
    print(f'pt: {hex(pt)}')
    print(f'ct: {hex(ct)}')
    print(f'rec_pt: {hex(rec_pt)}')

    assert pt == rec_pt
