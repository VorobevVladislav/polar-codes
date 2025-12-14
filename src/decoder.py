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
        # b - предыдущий бит
        if b == 0:
            return np.float64(y + x)
        elif b == 1:
            return np.float64(y - x)

    # (𝑢, 𝑣) → (𝑢 + 𝑣, 𝑢)
    # # Поэлементный XOR (сумма по модулю 2)
    # result = np.bitwise_xor(a, b)
    def u_v(self, u, v):
        u = list(u)
        v = list(v)
        # Поэлементный XOR для целых чисел
        # u_plus_v = list(np.bitwise_xor(u, v))
        u_plus_v = [int(a) ^ int(b) for a, b in zip(u, v)]
        result = u_plus_v + u
        # print(f"({u}, {v}) → ({u_plus_v}, {u}) = {result}")
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
            # print(f"code lenght: {len(code)}, bits: {b}, code: {code}, pm: {path_metrics}")

            if len(LLR) == 1:
                # попали в лист
                # Правило:
                # LLR > 0 → вероятнее бит 0
                # LLR < 0 → вероятнее бит 1
                if len(path_metrics) in self.freeze_positions:
                    # лист на замороженной позиции
                    b.append(0)
                elif LLR[0] < 0:
                    b.append(1)
                elif LLR[0] > 0:
                    b.append(0)
                # print(f"In len=1: code: {code[0]}, b: {b}")
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
            # print("выход после левого шага")

            # Делаем правый шаг
            right_b = []
            right_part = [
                self.R_step(left_part_copy[i], right_part_copy[i], left_b[i])
                for i in range(center)
            ]
            result_right, right_b, path_metrics = decompose(
                right_part, right_b, path_metrics
            )
            # print("выход после правого шага")

            # Вычисляем биты после шагов, чтобы передать их вверх по дереву
            b = self.u_v(right_b, left_b)

            return [result_left, result_right], b, path_metrics

        d_LLR, b, path_metrics = decompose(LLR)
        u_hat = []
        for el in path_metrics:
            u_hat.append(el["bit"])
        u_hat = np.array(u_hat)
        decoded = u_hat[self.info_positions]
        print(f"message = {message}")
        print(f"decoded = {decoded}")
        successfully_decoded = np.array_equal(decoded, message)
        if successfully_decoded:
            print("=" * 100)
            print("SC: УСПЕШНОЕ ДЕКОДИРОВАНИЕ")
            print("=" * 100)
        else:
            print("=" * 100)
            print("SC: ОШИБКА ДЕКОДИРОВАНИЯ")
            print("=" * 100)
        return u_hat, decoded, successfully_decoded

    def calc_pm(self, bits: list, LLR):
        def partial_decompose(bits: list, LLR, b=None, path_metrics=None):
            if b is None:
                b = []
            if path_metrics is None:
                path_metrics = []
            # print(f"LLR lenght: {len(LLR)}, bits: {b}, LLR: {LLR}, pm: {path_metrics}")

            # выход из рекурсии
            if len(path_metrics) == len(bits):
                # print("выход в начале")
                return bits, LLR[0], b, path_metrics

            if len(LLR) == 1:
                # попали в лист
                b.append(bits[len(path_metrics)])
                # print(f"In len=1: LLR: {LLR[0]}, b: {b}")
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
                # print("выход после левого шага")
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
                # print("выход после правого шага")
                return bits, [result_left, result_right], b, path_metrics

            return bits, [result_left, result_right], b, path_metrics

        bits, d_LLR, b, path_metrics = partial_decompose(bits, LLR)
        return path_metrics[-1]

    def scl_decode(self, LLR, message):
        paths = [{"path": [], "pm": 0}]

        for i in range(self.N):
            # print(f"Обрабатывается бит: {i}")
            if i in self.freeze_positions:
                # print("f")
                for p in paths:
                    p["path"].append(0)
                    pm = self.calc_pm(p["path"], LLR)
                    # print(f"pm = {pm}")
                    if self.hard_decision(pm["pm"]) != 0:
                        p["pm"] += abs(pm["pm"])
                    # print(p)

            else:
                # print("i")
                new_paths = []
                for b in [0, 1]:
                    # print("Разветвление:", b)
                    for p in paths:
                        new_p = {
                            "path": p["path"].copy(),
                            "pm": p["pm"],
                        }
                        new_p["path"].append(b)
                        updated_pm = self.calc_pm(new_p["path"], LLR)
                        # print(f"updated_pm = {updated_pm}")
                        if self.hard_decision(updated_pm["pm"]) != b:
                            # print("Сработал if для", updated_pm)
                            new_p["pm"] += abs(updated_pm["pm"])
                        # print("Добавляем путь к списку", new_p)
                        new_paths.append(new_p)
                new_paths.sort(key=lambda x: x["pm"])
                paths = new_paths[: self.list_lenght]
                # print("Остались пути:")
                # for p in paths:
                #     print(p)
        best_path = min(paths, key=lambda x: x["pm"])
        u_hat = best_path["path"].copy()
        u_hat = np.array(u_hat)
        decoded = u_hat[self.info_positions]
        # print(f"message = {message}")
        # print(f"decoded = {decoded}")
        successfully_decoded = np.array_equal(decoded, message)
        # if successfully_decoded:
        #     print("=" * 100)
        #     print("SCL: УСПЕШНОЕ ДЕКОДИРОВАНИЕ")
        #     print("=" * 100)
        # else:
        #     print("=" * 100)
        #     print("SCL: ОШИБКА ДЕКОДИРОВАНИЯ")
        #     print("=" * 100)
        return u_hat, decoded, successfully_decoded

    pass
