import matplotlib.pyplot as plt
import seaborn as sns


def plot_fer_results(df):
    """
    Построение графиков FER от SNR
    """

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # Группировка по N и L
    for N in df["N"].unique():
        df_N = df[df["N"] == N]

        plt.figure(figsize=(10, 6))
        plt.title(f"FER vs SNR для N={N}", fontsize=14)

        for L in sorted(df_N["L"].unique()):
            df_L = df_N[df_N["L"] == L]

            for R in sorted(df_L["R"].unique()):
                df_R = df_L[df_L["R"] == R]
                df_R = df_R.sort_values("EbN0_dB")

                # Фильтрация точек с достаточной статистикой
                # df_R_valid = df_R[df_R['FrameCount'] >= 100]
                df_R_valid = df_R

                if len(df_R_valid) > 1:
                    plt.semilogy(
                        df_R_valid["EbN0_dB"],
                        df_R_valid["FER"],
                        marker="o",
                        linestyle="-",
                        linewidth=2,
                        markersize=6,
                        label=f"R={R:.2f}, L={L}",
                    )

        plt.xlabel("Eb/N0 (dB)", fontsize=12)
        plt.ylabel("Frame Error Rate (FER)", fontsize=12)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.ylim(1e-4, 1)

        # Сохранение графика
        plt.tight_layout()
        plt.savefig(f"FER_vs_SNR_N{N}.png", dpi=150, bbox_inches="tight")
        plt.show()
