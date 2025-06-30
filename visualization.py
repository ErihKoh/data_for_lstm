import matplotlib.pyplot as plt


def visualization(history, y_test_real, y_pred_real, forecast_real=None, history_points=100):
    """
    Побудова 3 графіків:
    1. Тренування/валідація (MAE)
    2. Прогноз на тесті
    """

    # === 1. MAE: Train vs Validation ===
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('LSTM: Train vs Validation Loss')
    plt.xlabel('Епоха')
    plt.ylabel('MAE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 2. Прогноз на тестових даних ===
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_real, label='Реальна ціна (Close)', color='blue')
    plt.plot(y_pred_real, label='Прогнозована ціна', color='orange')
    plt.title('LSTM: прогноз ціни EUR/USD у доларах США (тестові дані)')
    plt.xlabel('Час')
    plt.ylabel('Ціна (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
