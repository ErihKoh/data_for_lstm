import numpy as np


def live_forecast_multi_step(model, scaled_data, scaler, window_size=120, steps_ahead=48):
    """
    Багатокроковий прогноз (iterative) на кілька кроків уперед.
    :param model: натренована модель LSTM
    :param scaled_data: нормалізовані дані (наприклад, з StandardScaler)
    :param scaler: fitted StandardScaler
    :param window_size: кількість кроків у вікні (наприклад, 120 = 10 годин)
    :param steps_ahead: кількість кроків уперед (48 = 4 години для таймфрейму M5)
    :return: реальні (денормалізовані) прогнозовані значення
    """
    input_window = scaled_data[-window_size:].copy()  # (window_size, n_features)
    predictions_scaled = []

    for _ in range(steps_ahead):
        input_seq = np.expand_dims(input_window, axis=0)  # (1, window_size, n_features)
        pred_scaled = model.predict(input_seq, verbose=0)
        predictions_scaled.append(pred_scaled[0, 0])  # передбачене значення (Close)

        # Додати нову точку до input_window
        if input_window.shape[1] == 1:  # тільки ціна
            next_input = np.array([[pred_scaled[0, 0]]])
        else:  # якщо є кілька фічей — заповнюємо лише першу (інше — нулі)
            next_input = np.zeros((1, input_window.shape[1]))
            next_input[0, 0] = pred_scaled[0, 0]

        input_window = np.vstack([input_window, next_input])[-window_size:]

    # Денормалізація
    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    if scaled_data.shape[1] > 1:
        zeros = np.zeros((steps_ahead, scaled_data.shape[1] - 1))
        predictions_full = np.hstack([predictions_scaled, zeros])
    else:
        predictions_full = predictions_scaled

    predictions_real = scaler.inverse_transform(predictions_full)[:, 0]

    # Вивід у консоль
    print(f"\n📈 Прогноз на {steps_ahead * 5} хв (≈ {steps_ahead // 12} год):")
    for i, p in enumerate(predictions_real[:48]):
        print(f"⏩ +{(i + 1) * 5} хв: {p:.5f} USD")

    return predictions_real
