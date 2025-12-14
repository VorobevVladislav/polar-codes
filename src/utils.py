import numpy as np


def tensor_power(F, n):
    """Возвращает F^⊗n (кронекерово произведение n раз)"""
    if n == 0:
        return np.array([[1]])
    elif n == 1:
        return F
    else:
        # Рекурсивно: F^⊗n = F ⊗ (F^⊗(n-1))
        return np.kron(F, tensor_power(F, n - 1))


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
