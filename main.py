import tensorflow as tf

from data_preprocecing import create_data, create_sequences, train_and_test
from live_forecast import live_forecast_multi_step
from train_lstm import train_lstm
from visualization import visualization


def main():
    window_size = 120
    scaled_data, scaler, df = create_data()
    X, y = create_sequences(scaled_data, window_size)
    X_train, X_test, y_train, y_test = train_and_test(X, y)
    print(f"Дані підготовлено: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")

    model, history, y_pred_real, y_test_real = train_lstm(X_train, y_train, X_test, y_test, scaler)

    forecast = live_forecast_multi_step(model, scaled_data, scaler, window_size=120, steps_ahead=48)

    visualization(history, y_test_real, y_pred_real, forecast, 100)


if __name__ == '__main__':
    main()