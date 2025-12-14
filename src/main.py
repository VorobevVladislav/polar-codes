import numpy as np
import pandas as pd

rank = pd.read_csv("rank.csv")


# def tensor_power(F, n):
#     """Возвращает F^⊗n (кронекерово произведение n раз)"""
#     if n == 0:
#         return np.array([[1]])
#     elif n == 1:
#         return F
#     else:
#         # Рекурсивно: F^⊗n = F ⊗ (F^⊗(n-1))
#         return np.kron(F, tensor_power(F, n - 1))


# # Пример:
# F = np.array([[1, 0], [1, 1]])
# n = 3  # log₂(8)
# G_8 = tensor_power(F, n)
# print(f"F^⊗{n} (размер {G_8.shape[0]}x{G_8.shape[1]}):")
# print(G_8)

# Длина кода
N = 8

# Скорость кода
R_speed = 1 / 2

# Количество информационных бит
K = int(R_speed * N)

reliability = np.array(rank["Q"])
# --- выбор замороженных позиций ---
freeze_positions = []
for idx in reliability:
    if idx < N:
        freeze_positions.append(idx)
        if len(freeze_positions) == K:
            break

freeze_positions = np.sort(np.array(freeze_positions, dtype=int))
info_positions = np.sort(np.setdiff1d(np.arange(N), freeze_positions))

print("K =", K)
print("info_positions (len={}):".format(len(info_positions)), info_positions)
print("freeze_positions (len={}):".format(len(freeze_positions)), freeze_positions)

# Генерируем сообщение, состощее из случайной последовательности 0 и 1
# message = np.random.randint(0, 2, K)
message = np.array([1, 0, 1, 0])
print(f"message = {message}")

# Вектор u
u = np.zeros(N, dtype=int)
print(f"u = {u}")

# Вставляем значения из message в позиции info_positions
u[info_positions] = message
print(f"u = {u}")


def tensor_power(F, n):
    """Возвращает F^⊗n (кронекерово произведение n раз)"""
    if n == 0:
        return np.array([[1]])
    elif n == 1:
        return F
    else:
        # Рекурсивно: F^⊗n = F ⊗ (F^⊗(n-1))
        return np.kron(F, tensor_power(F, n - 1))


# Применяем полярное преобразование
F = np.array([[1, 0], [1, 1]])
n = int(np.log2(N))
G_N = tensor_power(F, n)
print(f"F^⊗{n} (размер {G_N.shape[0]}x{G_N.shape[1]}):")
print(f"G_N = {G_N}")
x = (u @ G_N) % 2
print(f"x = {x}")


def polar_encode(u):
    """
    Быстрое преобразование для полярного кодирования.
    u — вектор длины N (N должно быть степени двойки).
    Возвращает кодовое слово x.
    """
    N = len(u)
    x = u.copy()

    stage = 1
    while stage < N:
        half = stage
        step = 2 * stage
        for i in range(0, N, step):
            for j in range(half):
                x[i + j] ^= x[i + j + half]  # XOR комбинация
        stage *= 2

    return x


x = polar_encode(u)
print(f"x = {x}")


def bpsk_mod(x):
    return 1 - 2 * x  # 0->+1, 1->-1


s = bpsk_mod(x)
print(f"x = {s}")

# Канал AWGN
sigma2 = 0.5
noise = np.random.normal(0, np.sqrt(sigma2), N)
print(f"noise = {noise}")
y = s + noise
print(f"y = {y}")

# LLR
LLR = 2 * y / sigma2
print(f"LLR = {LLR}")


def L(x, y):
    return np.sign(x * y) * np.min([np.abs(x), np.abs(y)])


def R(x, y, b):
    # b - предыдущий бит
    if b == 0:
        return np.float64(y + x)
    elif b == 1:
        return np.float64(y - x)


# def R(x, y, b):
#     if b == 0:
#         return y + x
#     elif b == 1:
#         return y - x

# Правило:
# LLR > 0 → вероятнее бит 0
# LLR < 0 → вероятнее бит 1


# (𝑢, 𝑣) → (𝑢 + 𝑣, 𝑢)
# # Поэлементный XOR (сумма по модулю 2)
# result = np.bitwise_xor(a, b)
def u_v(u, v):
    u = list(u)
    v = list(v)
    # Поэлементный XOR для целых чисел
    # u_plus_v = list(np.bitwise_xor(u, v))
    u_plus_v = [int(a) ^ int(b) for a, b in zip(u, v)]
    result = u_plus_v + u
    print(f"({u}, {v}) → ({u_plus_v}, {u}) = {result}")
    return result


def decompose(code, b=[], path_metrics=[]):
    print(f"code lenght: {len(code)}, bits: {b}, code: {code}, pm: {path_metrics}")

    if len(code) == 1:
        # попали в лист
        if len(path_metrics) in freeze_positions:
            # лист на замороженной позиции
            b.append(0)
        elif code[0] < 0:
            b.append(1)
        elif code[0] > 0:
            b.append(0)
        print(f"In len=1: code: {code[0]}, b: {b}")
        path_metrics.append({"id": len(path_metrics), "pm": code[0], "bit": b[-1]})
        return code[0], b, path_metrics

    # попали в узел
    center = int(len(code) / 2)
    left_part_copy = code[:center]
    right_part_copy = code[center:]

    # Делаем левый шаг
    left_b = []
    left_part = [L(left_part_copy[i], right_part_copy[i]) for i in range(center)]
    result_left, left_b, path_metrics = decompose(left_part, left_b, path_metrics)

    # Делаем правый шаг
    right_b = []
    right_part = [
        R(left_part_copy[i], right_part_copy[i], left_b[i]) for i in range(center)
    ]
    result_right, right_b, path_metrics = decompose(right_part, right_b, path_metrics)

    # Вычисляем биты после шагов, чтобы передать их вверх по дереву
    b = u_v(right_b, left_b)

    return [result_left, result_right], b, path_metrics


def find(binary_tree, N, id):
    if N <= 1:
        # выдаем значение
        return binary_tree
    if id < N / 2:
        # попали в левую часть дерева
        return find(binary_tree[0], N / 2, id % (N / 2))
    else:
        # попали в правую часть дерева
        return find(binary_tree[1], N / 2, id % (N / 2))


def recursive_to_array(binary_tree, N):
    result = []
    for i in range(N):
        result.append(np.float64(find(binary_tree, N, i)))
    return result


LLR = [-0.3, -1.2, 0.7, -0.8, -1.1, 0.9, -1.6, -0.5]
# print(LLR)
d_LLR, b, path_metrics = decompose(LLR)
print(d_LLR)
print(recursive_to_array(d_LLR, N))
print()
u_hat = []
for el in path_metrics:
    u_hat.append(el["bit"])
    print(f"{el["id"]}: pm = {el["pm"]}, bit = {el["bit"]}")

u_hat = np.array(u_hat)
print(u_hat)
decoded = u_hat[info_positions]
print(message)
print(decoded)
if np.array_equal(decoded, message):
    print("УСПЕШНОЕ ДЕКОДИРОВАНИЕ")
else:
    print("ОШИБКА ДЕКОДИРОВАНИЯ")
# # f_LLR = find(d_LLR, 9, 2)
# # print(f_LLR)
# for i in range(N):
#     print(find(d_LLR, int(np.log2(N)), i), i)


# SCL
L_length = 16
paths = [{"u_hat": [], "pm": 0.0}]
for i in range(N):
    if i in freeze_positions:
        for path in paths:
            bit = 0
            path["u_hat"].append(bit)
            # Обновление метрики пути
            path["pm"] += path_metrics[i]["pm"]
    else:
        # 2B. Информационный бит - РАЗВЕТВЛЕНИЕ
        expanded_paths = []

        for path in paths:
            # Два варианта: бит=0 и бит=1
            for bit in [0, 1]:
                new_path = path.copy()
                new_path["u_hat"] = path["u_hat"].copy()
                new_path["u_hat"].append(bit)
                new_path["pm"] = path["pm"] + path_metrics[i]["pm"]
                expanded_paths.append(new_path)

        # 3. Отбор L лучших путей
        expanded_paths.sort(key=lambda x: x["pm"])  # Сортировка по метрике
        paths = expanded_paths[:L_length]  # Оставляем L лучших

# 4. В конце выбираем путь с лучшей метрикой
best_path = min(paths, key=lambda x: x["pm"])
print(best_path["u_hat"])
