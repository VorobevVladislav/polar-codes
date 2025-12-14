from simulation import run_simulation_series, quick_test

if __name__ == "__main__":
    # Выбор режима работы
    print("Выберите режим работы:")
    print("1 - Полная симуляция (длительно)")
    print("2 - Быстрый тест (код (8,4))")
    # print("3 - Тест канала (генерация и демодуляция)")
    
    choice = input("Ваш выбор (1-2): ").strip()
    
    if choice == "1":
        results = run_simulation_series()
        if results is not None:
            print("\nСводка результатов:")
            print(results.groupby(['N', 'R', 'L']).agg({
                'FER': 'min',
                'EbN0_dB': 'count'
            }))
    elif choice == "2":
        quick_test()
    else:
        print("Неверный выбор. Запуск быстрого теста...")
        quick_test()