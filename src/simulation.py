import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from src.channel import Channel
from src.encoder import PolarEncoder
from src.decoder import SCLDecoder
from src.visualize import plot_fer_results


def simulate_polar_code(N, R, L, EbN0_dB_list, max_errors=100, max_frames=10000):
    """
    Симуляция полярного кода с SCL декодированием

    Args:
        N: длина кода
        R: скорость кода
        L: размер списка SCL декодера
        EbN0_dB_list: список значений SNR в dB
        max_errors: максимальное количество ошибок для остановки на каждом SNR
        max_frames: максимальное количество кадров для симуляции на каждом SNR

    Returns:
        DataFrame с результатами симуляции
    """
    K = int(R * N)  # Количество информационных бит

    # Инициализация кодера и декодера
    polar_encoder = PolarEncoder(N, R, K, "rank.csv")
    info_positions, freeze_positions = polar_encoder.set_info_and_freeze_positions()
    scl_decoder = SCLDecoder(N, R, K, L, freeze_positions, info_positions)

    results = []

    for EbN0_dB in EbN0_dB_list:
        print(f"\nСимуляция: N={N}, R={R:.3f}, L={L}, EbN0={EbN0_dB} dB")

        # Инициализация канала для данного SNR
        channel = Channel(EbN0_dB, R)

        error_count = 0
        frame_count = 0

        # Прогресс-бар для симуляции
        pbar = tqdm(total=max_errors, desc=f"SNR={EbN0_dB}dB")

        while error_count < max_errors and frame_count < max_frames:
            # Генерация случайного сообщения
            message = np.random.randint(0, 2, K)

            # Кодирование
            u = polar_encoder.get_u_vector(message)
            x = polar_encoder.encode(u)
            s = polar_encoder.bpsk_mod(x)

            # Передача через канал
            y = channel.add_noise(s)

            # Расчет LLR
            llr = channel.calc_llr(y)

            # Декодирование
            u_hat, decoded_bits, successfully_decoded = scl_decoder.scl_decode(
                llr, message
            )

            # Проверка на ошибку декодирования
            if not successfully_decoded:
                error_count += 1

            frame_count += 1

            # Обновление прогресс-бара
            pbar.n = error_count
            pbar.refresh()

            if error_count >= max_errors:
                break

        pbar.close()

        # Расчет FER
        FER = error_count / frame_count

        # Сохранение результатов
        results.append(
            {
                "N": N,
                "R": R,
                "K": K,
                "L": L,
                "EbN0_dB": EbN0_dB,
                "NoiseVariance": channel.get_noise_variance(),
                "FrameCount": frame_count,
                "ErrorCount": error_count,
                "FER": FER,
            }
        )

        print(f"  Frames: {frame_count}, Errors: {error_count}, FER: {FER:.2e}")

    return pd.DataFrame(results)


def run_simulation_series():
    """
    Запуск серии симуляций для разных параметров
    """
    # Параметры симуляции
    N_list = [16, 32, 64]  # Длины кодов
    R_list = [1 / 3, 1 / 2, 2 / 3]  # Скорости кодов
    L_list = [4, 8, 16]  # Размеры списка

    # Диапазон SNR (в dB)
    EbN0_dB_min = -3
    EbN0_dB_max = 3
    EbN0_dB_step = 1
    EbN0_dB_list = np.arange(EbN0_dB_min, EbN0_dB_max + EbN0_dB_step, EbN0_dB_step)

    # Параметры симуляции
    max_errors = 30  # Останавливаемся после 30 ошибок
    max_frames = 10000  # Максимум 10000 кадров на точку

    all_results = []

    print("=" * 70)
    print("Запуск симуляции полярных кодов с SCL декодированием")
    print("=" * 70)

    # Запуск симуляций
    for N in N_list:
        for R in R_list:
            for L in L_list:
                K = int(R * N)
                if K < 1:  # Пропускаем если слишком мало информационных бит
                    continue

                print(f"\n{'='*60}")
                print(f"Симуляция: N={N}, R={R:.3f}, K={K}, L={L}")
                print(f"{'='*60}")

                try:
                    df = simulate_polar_code(
                        N=N,
                        R=R,
                        L=L,
                        EbN0_dB_list=EbN0_dB_list,
                        max_errors=max_errors,
                        max_frames=max_frames,
                    )
                    all_results.append(df)
                except Exception as e:
                    print(f"Ошибка при симуляции N={N}, R={R}, L={L}: {e}")

    # Объединение всех результатов
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Сохранение результатов в файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"polar_code_simulation_results_{timestamp}.csv"
        final_df.to_csv(filename, index=False)
        print(f"\nРезультаты сохранены в {filename}")

        # Создание графиков FER(SNR)
        plot_fer_results(final_df)

        return final_df
    else:
        print("Нет результатов для сохранения")
        return None


def quick_test():
    """
    Быстрый тест для проверки работы
    """
    print("Быстрый тест полярного кода (8, 4) с SCL декодированием")

    N = 8
    R = 0.5
    K = 4
    L = 4

    # Тестовые SNR
    EbN0_dB_list = [0, 1, 2, 3, 4]

    df = simulate_polar_code(
        N=N, R=R, L=L, EbN0_dB_list=EbN0_dB_list, max_errors=20, max_frames=1000
    )

    print("\nРезультаты теста:")
    print(df[["EbN0_dB", "FrameCount", "ErrorCount", "FER"]])

    # Построение графика для теста
    plt.figure(figsize=(8, 6))
    plt.semilogy(df["EbN0_dB"], df["FER"], "bo-", linewidth=2, markersize=8)
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("FER")
    plt.title(f"Полярный код ({N}, {K}), SCL декодер L={L}")
    plt.grid(True)
    plt.show()
