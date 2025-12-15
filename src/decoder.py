import numpy as np
from src.utils import recursive_to_array


class SCLDecoder:
    def __init__(self, N, R, K, list_lenght, freeze_positions, info_positions) -> None:
        self.N = N
        self.R = R
        self.K = K
        self.list_lenght = list_lenght
        self.freeze_positions = freeze_positions
        self.info_positions = info_positions
        pass

    def get_N(self):
        return self.N

    def get_R(self):
        return self.R

    def get_K(self):
        return self.K

    def get_freeze_positions(self):
        return self.freeze_positions

    def get_info_positions(self):
        return self.info_positions

    def L_step(self, x, y):
        return np.sign(x * y) * np.min([np.abs(x), np.abs(y)])

    def R_step(self, x, y, b):
        if b == 0:
            return np.float64(y + x)
        elif b == 1:
            return np.float64(y - x)

    def u_v(self, u, v):
        # (𝑢, 𝑣) → (𝑢 + 𝑣, 𝑢)
        u = list(u)
        v = list(v)
        # Поэлементный XOR для целых чисел
        u_plus_v = [int(a) ^ int(b) for a, b in zip(u, v)]
        result = u_plus_v + u
        return result

    def hard_decision(self, L):
        if L > 0:
            return 0
        elif L < 0:
            return 1

    def sc_decode(self, LLR, message):
        def decompose(LLR, b=None, path_metrics=None):
            if b is None:
                b = []
            if path_metrics is None:
                path_metrics = []

            if len(LLR) == 1:
                # попали в лист
                if len(path_metrics) in self.freeze_positions:
                    # лист на замороженной позиции
                    b.append(0)
                else:
                    b.append(self.hard_decision(LLR[0]))
                path_metrics.append(
                    {"id": len(path_metrics), "pm": LLR[0], "bit": b[-1]}
                )
                return LLR[0], b, path_metrics

            # попали в узел
            center = int(len(LLR) / 2)
            left_part_copy = LLR[:center]
            right_part_copy = LLR[center:]

            # Делаем левый шаг
            left_b = []
            left_part = [
                self.L_step(left_part_copy[i], right_part_copy[i])
                for i in range(center)
            ]
            result_left, left_b, path_metrics = decompose(
                left_part, left_b, path_metrics
            )

            # Делаем правый шаг
            right_b = []
            right_part = [
                self.R_step(left_part_copy[i], right_part_copy[i], left_b[i])
                for i in range(center)
            ]
            result_right, right_b, path_metrics = decompose(
                right_part, right_b, path_metrics
            )

            # Вычисляем биты после шагов, чтобы передать их вверх по дереву
            b = self.u_v(right_b, left_b)

            return [result_left, result_right], b, path_metrics

        d_LLR, b, path_metrics = decompose(LLR)
        u_hat = []
        for el in path_metrics:
            u_hat.append(el["bit"])
        u_hat = np.array(u_hat)
        decoded = u_hat[self.info_positions]
        successfully_decoded = np.array_equal(decoded, message)
        return u_hat, decoded, successfully_decoded

    def calc_pm(self, bits: list, LLR):
        def partial_decompose(bits: list, LLR, b=None, path_metrics=None):
            if b is None:
                b = []
            if path_metrics is None:
                path_metrics = []

            # выход из рекурсии
            if len(path_metrics) == len(bits):
                return bits, LLR[0], b, path_metrics

            if len(LLR) == 1:
                # попали в лист
                b.append(bits[len(path_metrics)])
                path_metrics.append(
                    {"id": len(path_metrics), "pm": LLR[0], "bit": b[-1]}
                )
                return bits, LLR[0], b, path_metrics

            # попали в узел
            center = int(len(LLR) / 2)
            left_part_copy = LLR[:center]
            right_part_copy = LLR[center:]

            # Делаем левый шаг
            left_b = []
            left_part = [
                self.L_step(left_part_copy[i], right_part_copy[i])
                for i in range(center)
            ]
            bits, result_left, left_b, path_metrics = partial_decompose(
                bits, left_part, left_b, path_metrics
            )
            # выход из рекурсии
            if len(path_metrics) == len(bits):
                return bits, [result_left, None], left_b, path_metrics

            # Делаем правый шаг
            right_b = []
            right_part = [
                self.R_step(left_part_copy[i], right_part_copy[i], left_b[i])
                for i in range(center)
            ]
            bits, result_right, right_b, path_metrics = partial_decompose(
                bits, right_part, right_b, path_metrics
            )

            # Вычисляем биты после шагов, чтобы передать их вверх по дереву
            b = self.u_v(right_b, left_b)

            # выход из рекурсии
            if len(path_metrics) == len(bits):
                return bits, [result_left, result_right], b, path_metrics

            return bits, [result_left, result_right], b, path_metrics

        bits, d_LLR, b, path_metrics = partial_decompose(bits, LLR)
        return path_metrics[-1]

    def scl_decode(self, LLR, message):
        paths = [{"path": [], "pm": 0}]
        for i in range(self.N):
            if i in self.freeze_positions:
                # на замороженных битах
                for p in paths:
                    p["path"].append(0)
                    pm = self.calc_pm(p["path"], LLR)
                    if self.hard_decision(pm["pm"]) != 0:
                        p["pm"] += abs(pm["pm"])
            else:
                # на информационных битах
                new_paths = []
                for b in [0, 1]:
                    # разветвление
                    for p in paths:
                        new_p = {
                            "path": p["path"].copy(),
                            "pm": p["pm"],
                        }
                        new_p["path"].append(b)
                        updated_pm = self.calc_pm(new_p["path"], LLR)
                        # Проверяем совпдение с hard decision
                        if self.hard_decision(updated_pm["pm"]) != b:
                            # если нет совпадения, то ухудшаем метрику пути
                            new_p["pm"] += abs(updated_pm["pm"])
                        new_paths.append(new_p)
                new_paths.sort(key=lambda x: x["pm"])
                # Оставляем только L лучших путей в списке
                paths = new_paths[: self.list_lenght]

        best_path = min(paths, key=lambda x: x["pm"])
        u_hat = best_path["path"].copy()
        u_hat = np.array(u_hat)
        decoded = u_hat[self.info_positions]
        successfully_decoded = np.array_equal(decoded, message)
        return u_hat, decoded, successfully_decoded

    pass
