import matplotlib.pyplot as plt


def visualization(history, y_test_real, y_pred_real):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM: Train vs Validation Loss')
    plt.xlabel('Епоха')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ВІЗУАЛІЗАЦІЯ
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label='Реальна ціна (Close)', color='blue')
    plt.plot(y_pred_real, label='Прогнозована ціна', color='orange')
    plt.title('LSTM: прогноз ціни EUR/USD у доларах США')
    plt.xlabel('Тестові зразки')
    plt.ylabel('Ціна (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
