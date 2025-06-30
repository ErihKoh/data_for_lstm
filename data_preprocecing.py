import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def calculate_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.Series:
    delta = df[column].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df: pd.DataFrame, column: str = "Close",
                   fast: int = 8, slow: int = 21, signal: int = 5) -> pd.DataFrame:
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["macd"] = macd_line
    df["macd_Signal"] = signal_line
    df["macd_Hist"] = histogram

    return df


def create_data():
    # Завантаження даних
    df = pd.read_csv('EURUSD_M5_202501021000_202506301725.csv', delimiter='\t')
    df.drop(columns=['<TICKVOL>', '<VOL>', '<SPREAD>'], inplace=True)
    df = df.rename(columns={'<DATE>': 'Date', '<TIME>': 'Time', '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low',
                            '<CLOSE>': 'Close'})


    # Обчислення індикаторів

    df['rsi'] = calculate_rsi(df)
    df = calculate_macd(df)

    # Вибір ознак
    features = ['Close', 'rsi', 'macd']
    df.dropna(inplace=True)
    data = df[features].values

    # Масштабування
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler, df


# Формування послідовностей
def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])  # прогнозуємо ціну закриття
    return np.array(X), np.array(y)


# Розділення на train/test
def train_and_test(x, y):
    split = int(0.8 * len(x))
    X_train, X_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test






