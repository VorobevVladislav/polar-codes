# import numpy as np


# class Channel:
#     def __init__(self, sigma2, N) -> None:
#         self.sigma2 = sigma2
#         self.N = N
#         pass

#     def effect(self, s):
#         # Канал AWGN
#         noise = np.random.normal(0, np.sqrt(self.sigma2), self.N)
#         # print(f"noise = {noise}")
#         y = s + noise
#         # print(f"y = {y}")
#         return y

#     def calc_LLR(self, y):
#         # LLR
#         LLR = 2 * y / self.sigma2
#         # print(f"LLR = {LLR}")
#         return LLR
# file: channel.py
import numpy as np


class Channel:
    def __init__(self, EbN0_dB, code_rate, noise_variance=None) -> None:
        """
        Инициализация канала AWGN

        Args:
            EbN0_dB: SNR в dB (Eb/N0)
            code_rate: скорость кода R
            noise_variance: дисперсия шума (если None - вычисляется из EbN0_dB)
        """
        self.EbN0_dB = EbN0_dB
        self.code_rate = code_rate
        self.noise_variance = noise_variance

        if self.noise_variance is None:
            # Преобразование Eb/N0 (dB) в мощность шума sigma^2
            # Для BPSK: Es/N0 = R * Eb/N0
            # sigma^2 = N0/2 = 1/(2 * R * 10^(EbN0_dB/10))
            EbN0_linear = 10 ** (EbN0_dB / 10)
            EsN0_linear = code_rate * EbN0_linear
            # Для BPSK, мощность сигнала = 1
            self.noise_variance = 1.0 / (2.0 * EsN0_linear)

    def add_noise(self, signal):
        """
        Добавление шума к сигналу BPSK

        Args:
            signal: массив BPSK символов (+1/-1)
        Returns:
            y: зашумленные символы
        """
        noise = np.random.normal(0, np.sqrt(self.noise_variance), len(signal))
        y = signal + noise
        return y

    def calc_llr(self, y):
        """
        Расчет LLR (Log-Likelihood Ratio) для BPSK в AWGN

        Args:
            y: зашумленные символы
        Returns:
            LLR значения
        """
        # LLR = 2 * y / sigma^2
        llr = 2.0 * y / self.noise_variance
        return llr

    def get_noise_variance(self):
        """Возвращает дисперсию шума"""
        return self.noise_variance
