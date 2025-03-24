import numpy as np
import talib


def compute_oscillators(data):
    """Compute various financial oscillators and indicators.

    This function extracts classical Technical Analysis features
    commonly used to analyse financial datasets. 

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing 'Close', 'High', 'Low', and 'Volume' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added oscillator and indicator columns.
    """
    log_return = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data = data.assign(
        Z_score=((log_return - log_return.rolling(20).mean()) / log_return.rolling(20).std())
    )
    data = data.assign(RSI=talib.RSI(data['Close']) / 100)
    upper_band, _, lower_band = talib.BBANDS(data['Close'], nbdevup=2, nbdevdn=2, matype=0)
    data = data.assign(boll=(data['Close'] - lower_band) / (upper_band - lower_band))
    data = data.assign(ULTOSC=talib.ULTOSC(data['High'], data['Low'], data['Close']) / 100)
    data = data.assign(pct_change=data['Close'].pct_change())
    data = data.assign(zsVol=(data['Volume'] - data['Volume'].mean()) / data['Volume'].std())
    data = data.assign(
        PR_MA_Ratio_short=(data['Close'] - talib.SMA(data['Close'], 21)) / talib.SMA(data['Close'], 21)
    )
    data = data.assign(
        MA_Ratio_short=(talib.SMA(data['Close'], 21) - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50)
    )
    data = data.assign(
        MA_Ratio=(talib.SMA(data['Close'], 50) - talib.SMA(data['Close'], 100)) / talib.SMA(data['Close'], 100)
    )
    data = data.assign(
        PR_MA_Ratio=(data['Close'] - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50)
    )

    return data


def compute_oscillators_cryptocurrency(data):
    """Compute various financial oscillators and indicators for cryptocurrency data.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing 'Close', 'High', 'Low', 'Volume_contracts',
        'Volume_currency', and 'Volume_quote' columns. These values are
        related to the OKX API.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added oscillator and indicator columns specific to cryptocurrency.
    """
    log_return = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data = data.assign(
        Z_score=((log_return - log_return.rolling(20).mean()) / log_return.rolling(20).std())
    )
    data = data.assign(RSI=talib.RSI(data['Close']) / 100)
    upper_band, _, lower_band = talib.BBANDS(data['Close'], nbdevup=2, nbdevdn=2, matype=0)
    data = data.assign(boll=(data['Close'] - lower_band) / (upper_band - lower_band))
    data = data.assign(ULTOSC=talib.ULTOSC(data['High'], data['Low'], data['Close']) / 100)
    data = data.assign(pct_change=data['Close'].pct_change())
    data = data.assign(
        zsVolume_contracts=(data['Volume_contracts'] - data['Volume_contracts'].mean()) / data['Volume_contracts'].std()
    )
    data = data.assign(
        Volume_currency=(data['Volume_currency'] - data['Volume_currency'].mean()) / data['Volume_currency'].std()
    )
    data = data.assign(
        Volume_quote=(data['Volume_quote'] - data['Volume_quote'].mean()) / data['Volume_quote'].std()
    )
    data = data.assign(
        PR_MA_Ratio_short=(data['Close'] - talib.SMA(data['Close'], 21)) / talib.SMA(data['Close'], 21)
    )
    data = data.assign(
        MA_Ratio_short=(talib.SMA(data['Close'], 21) - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50)
    )
    data = data.assign(
        MA_Ratio=(talib.SMA(data['Close'], 50) - talib.SMA(data['Close'], 100)) / talib.SMA(data['Close'], 100)
    )
    data = data.assign(
        PR_MA_Ratio=(data['Close'] - talib.SMA(data['Close'], 50)) / talib.SMA(data['Close'], 50)
    )

    return data


def find_patterns(data):
    """
    Identify various candlestick patterns in the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing 'Open', 'High', 'Low', and 'Close' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with added columns for each candlestick pattern.
    """
    data = data.assign(CDL2CROWS=talib.CDL2CROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100)
    data = data.assign(
        CDL3BLACKCROWS=talib.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDL3WHITESOLDIERS=talib.CDL3WHITESOLDIERS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLABANDONEDBABY=talib.CDLABANDONEDBABY(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(CDLBELTHOLD=talib.CDLBELTHOLD(data['Open'], data['High'], data['Low'], data['Close']) / 100)
    data = data.assign(
        CDLCOUNTERATTACK=talib.CDLCOUNTERATTACK(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLDARKCLOUDCOVER=talib.CDLDARKCLOUDCOVER(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLDRAGONFLYDOJI=talib.CDLDRAGONFLYDOJI(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLENGULFING=talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLEVENINGDOJISTAR=talib.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLEVENINGSTAR=talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLGRAVESTONEDOJI=talib.CDLGRAVESTONEDOJI(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLHANGINGMAN=talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLHARAMICROSS=talib.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLINVERTEDHAMMER=talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLMARUBOZU=talib.CDLMARUBOZU(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLMORNINGDOJISTAR=talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLMORNINGSTAR=talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLPIERCING=talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLRISEFALL3METHODS=talib.CDLRISEFALL3METHODS(
            data['Open'], data['High'], data['Low'], data['Close']
        ) / 100
    )
    data = data.assign(
        CDLSHOOTINGSTAR=talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLSPINNINGTOP=talib.CDLSPINNINGTOP(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )
    data = data.assign(
        CDLUPSIDEGAP2CROWS=talib.CDLUPSIDEGAP2CROWS(data['Open'], data['High'], data['Low'], data['Close']) / 100
    )

    return data
