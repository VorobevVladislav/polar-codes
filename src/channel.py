import numpy as np


class Channel:
    def __init__(self, sigma2, N) -> None:
        self.sigma2 = sigma2
        self.N = N
        pass

    def effect(self, s):
        # Канал AWGN
        noise = np.random.normal(0, np.sqrt(self.sigma2), self.N)
        # print(f"noise = {noise}")
        y = s + noise
        # print(f"y = {y}")
        return y

    def calc_LLR(self, y):
        # LLR
        LLR = 2 * y / self.sigma2
        # print(f"LLR = {LLR}")
        return LLR
