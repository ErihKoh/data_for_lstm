from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import joblib


def train_lstm(X_train, y_train, X_test, y_test, scaler, epochs=20, batch_size=16):

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))  # опціонально
    model.add(Dense(units=1))  # прогноз одного значення (напр. ціна)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)

    # Прогноз
    y_pred = model.predict(X_test)

    # ВІДНОВЛЕННЯ РЕАЛЬНИХ ЗНАЧЕНЬ CLOSE (через scaler.inverse_transform)
    # reshape до (samples, 1), щоб можна було об'єднати назад
    y_pred_full = np.concatenate([y_pred, np.zeros((len(y_pred), 2))], axis=1)
    y_test_full = np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), 2))], axis=1)

    y_pred_real = scaler.inverse_transform(y_pred_full)[:, 0]
    y_test_real = scaler.inverse_transform(y_test_full)[:, 0]

    # Метрики
    mae = mean_absolute_error(y_test_real, y_pred_real)

    print(f"\n📊 MAE: {mae:.5f}")

    model.save("model.keras")
    joblib.dump(scaler, "scaler.save")

    return model, history, y_pred_real, y_test_real
