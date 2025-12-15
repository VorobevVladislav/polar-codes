import numpy as np
import pandas as pd
from src.utils import tensor_power


class PolarEncoder:
    def __init__(self, N, R, K, file_path) -> None:
        self.N = N
        self.R = R
        self.K = K
        self.rank = pd.read_csv(file_path)
        self.freeze_positions = None
        self.info_positions = None
        pass

    def get_N(self):
        return self.N

    def get_R(self):
        return self.R

    def get_K(self):
        return self.K

    def get_rank(self):
        return self.rank

    def get_freeze_positions(self):
        return self.freeze_positions

    def get_info_positions(self):
        return self.info_positions

    def set_info_and_freeze_positions(self):
        reliability = np.array(self.rank["Q"])
        # Фильтруем только позиции, которые меньше N
        valid_positions = reliability[reliability < self.N]
        # Замороженные позиции - первые N-K (худшие надежности)
        self.freeze_positions = np.sort(
            np.array(valid_positions[: self.N - self.K], dtype=int)
        )
        # Информационные позиции - остальные (лучшие K позиций)
        self.info_positions = np.sort(
            np.array(valid_positions[self.N - self.K :], dtype=int)
        )
        return self.info_positions, self.freeze_positions

    def get_u_vector(self, message):
        # Вектор u
        u = np.zeros(self.N, dtype=int)
        # Вставляем значения из message в позиции info_positions
        u[self.info_positions] = message
        return u

    def fast_encode(self, u):
        """
        Быстрое преобразование для полярного кодирования.
        u — вектор длины N (N должно быть степени двойки).
        Возвращает кодовое слово x.
        """
        x = u.copy()
        stage = 1
        while stage < self.N:
            half = stage
            step = 2 * stage
            for i in range(0, self.N, step):
                for j in range(half):
                    x[i + j] ^= x[i + j + half]  # XOR комбинация
            stage *= 2
        return x

    def encode(self, u):
        # Применяем полярное преобразование
        F = np.array([[1, 0], [1, 1]])
        n = int(np.log2(self.N))
        G_N = tensor_power(F, n)
        x = (u @ G_N) % 2
        return x

    def bpsk_mod(self, x):
        s = 1 - 2 * x
        return s

    pass
