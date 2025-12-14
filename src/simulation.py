import numpy as np

# Длина кода
N = 512

# Скорость кода
R = 1 / 2

# Количество информационных бит
K = int(R * N)

# Генерируем сообщение, состощее из случайной последовательности 0 и 1
message = np.random.randint(0, 2, K)
# message = np.array([1, 0, 1, 0])
print(f"message = {message}")
