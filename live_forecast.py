import numpy as np


def live_forecast(model, scaled_data, scaler, window_size=100):
    last_window = scaled_data[-window_size:]  # останні 100 точок
    input_data = np.expand_dims(last_window, axis=0)  # (1, 100, 3)

    pred_scaled = model.predict(input_data)
    pred_full = np.concatenate([pred_scaled, np.zeros((1, 2))], axis=1)
    pred_real = scaler.inverse_transform(pred_full)[:, 0][0]

    print(f"\n📈 Прогноз ціни EUR/USD на наступний день: {pred_real:.5f} USD")
