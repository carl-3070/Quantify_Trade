import pandas as pd 
import numpy as np

def bollinger_bands(df, price_name='close', window=20, std_times=2):
    # Middle Band = 20-day simple moving average (SMA)
    # Upper Band = 20-day SMA + (20-day standard deviation of price x 2) 
    # Lower Band = 20-day SMA - (20-day standard deviation of price x 2)

    #Calculate rolling mean and standard deviation using number of days set above
    rolling_mean = df[price_name].rolling(window).mean()
    rolling_std = df[price_name].rolling(window).std()

    #create two new DataFrame columns to hold values of upper and lower Bollinger bands
    ret_df = pd.DataFrame()
    ret_df[price_name] = df[price_name]
    ret_df['Middle_Band'] = rolling_mean
    ret_df['Upper_Band'] = rolling_mean + (rolling_std * std_times)
    ret_df['Lower_Band'] = rolling_mean - (rolling_std * std_times)
    # ((Upper Band - Lower Band) / Middle Band) * 100
    ret_df['Band_Width'] = ( (ret_df['Upper_Band'] - ret_df['Lower_Band']) / ret_df['Middle_Band']) * 100
    # %B = (Price - Lower Band)/(Upper Band - Lower Band)
    ret_df['Percent_Band'] =  (df[price_name] - ret_df['Lower_Band']) / (ret_df['Upper_Band'] -ret_df['Lower_Band'])
    
    return ret_df

def adl(df):
    # 1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
    flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close']))/(df['high'] - df['low'])
    
    # 2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
    flow_vol = flow_multiplier * df['vol']

    # 3. ADL = Previous ADL + Current Period's Money Flow Volume
    adl = flow_vol.rolling(2).sum()
    ret_df = pd.DataFrame()
    ret_df['close'] = df['close']
    ret_df['flow_multiplier'] = flow_multiplier
    ret_df['adl'] = adl
    return ret_df

def aroon(df, n=25):
    # Aroon-Up = ((25 - Days Since 25-day High)/25) x 100
    # Aroon-Down = ((25 - Days Since 25-day Low)/25) x 100
    # Aroon Oscillator = Aroon-Up  -  Aroon-Down
    up = 100 * (25 - df['close'].rolling(n).apply(lambda x: x.argmax(), raw=True))/n
    down = 100 * (25 - df['close'].rolling(n).apply(lambda x: x.argmin(), raw=True))/n
    oscillator = up -down
    return pd.DataFrame(dict(up=up, down=down, oscillator=oscillator))

def true_range(df, period=14):
    tr = (np.maximum((df['high'] - df['low']), abs(df['high'] - df['pre_close']))).rolling(period).sum()
    atr = tr/period
    return tr, atr

def adx(df, period=14):
    '''
    TR = SUM(MAX(MAX(HIGH - LOW, ABS(HIGH-REF(CLOSE,1))), ABS(LOW - REF(CLOSE, 1))), N)
    HD = HIGH - REF(HIGH, 1)
    LD = REF(LOW, 1) - LOW
    DMP = SUM(IF(HD>0 AND HD>LD, HD, 0), N)
    DMM = SUM(IF(LD>0 AND LD>HD, LD, 0), N)
    PDI = DMP*100/TR
    MDI = DMM*100/TR
    DX = ABS(MDI - PDI)/(MDI + PDI)*100
    ADX = MA(ABS(MDI - PDI)/(MDI + PDI)*100, M)
    REF(X, N)：引用X在N个周期前的值
    '''
    #tr = (np.maximum((df['high'] - df['low']), abs(df['high'] - df['pre_close']))).rolling(period).sum()
    tr, atr = true_range(df, period)
    hd = df['high'].diff()
    ld = df['low'].diff(-1)
    ld[1:] = ld[:-1]
    ld[0]=np.nan
    dmp = pd.Series([(x if c else 0) for x,c in zip(hd,(hd>0) & (hd>ld))]).rolling(period).sum()
    dmm = pd.Series([(x if c else 0) for x,c in zip(ld,(ld>0) & (ld>hd))]).rolling(period).sum()
    pdi = dmp*100/TR
    mdi = dmm*100/TR
    dx = np.abs(mdi - pdi)/(mdi + pdi) *100
    adx = dx.rolling(period).mean()
    return pd.DataFrame(dict(adx=adx, pdi=pdi, mdi=mdi))

def cmf(df, period=20):
    # 1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
    # 2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
    # 3. 20-period CMF = 20-period Sum of Money Flow Volume / 20 period Sum of Volume 
    multiplier = ((df['close']-df['low']) - (df['high'] - df['close']))/(df['high'] - df['low'])
    flow_vol = multiplier * df['vol']
    cmf = flow_vol.rolling(period).sum()/ df['vol'].rolling(period).sum()
    return pd.DataFrame(dict(cmf=cmf))
    
def chaikin_oscillator(df):
    # Chaikin Oscillator = (3-day EMA of ADL)  -  (10-day EMA of ADL)
    df_adl = adl(df)
    #EMA: pd.ewm()
    co = df_adl['adl'].ewm(span=3, adjust=False).mean() - df_adl['adl'].ewm(span=10, adjust=False).mean()
    return co

def ctm(df):
    # Bollinger bands in four different timeframes (20-day, 50-day, 75-day and 100-day)
    # The price change relative to the standard deviation over the past 100 days
    # The 14-day RSI value
    # The existence of any short-term (2-day) price channel breakouts
    return 'not suport'
    
def rsi(df, period=14):
    '''            
                      100
        RSI = 100 - --------
                     1 + RS

        RS = Average Gain / Average Loss
    ''' 

    gain_avg = df['change'].rolling(period).apply(lambda x:sum(p for p in x if p > 0), raw=True)/period
    loss_avg = -1*df['change'].rolling(period).apply(lambda x:sum(n for n in x if n < 0), raw=True)/period
    rs = gain_avg / loss_avg
    rsi = 100 - 100/(1+rs)
    return rsi

def price_channel(df, period=20):
    # Upper Channel Line: 20-day high
    # Lower Channel Line: 20-day low
    # Centerline: (20-day high + 20-day low)/2 
    upper_channel = df['high'].rolling(period).max()
    lower_channel = df['low'].rolling(period).max()
    center_line = (upper_channel + lower_channel)/2
    return pd.DataFrame(dict(upper_channel=upper_channel, lower_channel=lower_channel, center_line=center_line))

def cci(df, period=20):
    # CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)
    # Typical Price (TP) = (High + Low + Close)/3
    # Constant = .015
    # There are four steps to calculating the Mean Deviation: 
    # First, subtract the most recent 20-period average of the typical price from each period's typical price. 
    # Second, take the absolute values of these numbers. 
    # Third, sum the absolute values. 
    # Fourth, divide by the total number of periods (20).
    c = 0.015
    typical_price = (df['high'] + df['low'] + df['close'])/3
    md = typical_price.rolling(period).apply(lambda x: abs(x-x.mean()).sum(), raw=True)/period
    cci = (typical_price - typical_price.rolling(period).mean()) / (c * md)
    return cci

def roc(df, period=10):
    '''
    ROC = [(Close - Close n periods ago) / (Close n periods ago)] * 100
    '''
    
    roc = 100 * df['close'].diff(period) / df['close'].shift(period)
    return roc

def coppock_curve(df):
    '''
    Coppock Curve = 10-period WMA of (14-period RoC + 11-period RoC)
    WMA = Weighted Moving Average
    RoC = Rate-of-Change
    '''
    roc_14 = roc(df, 14)
    roc_11 = roc(df, 11)
    weight = np.arange(1, 11)
    cv = (roc_14 + roc_11).rolling(10).apply(lambda x:np.average(x, weights=weight), raw=True)
    return cv

def pmo(df, period):
    '''
    Price Momentum Oscillator
    Smoothing Multiplier = (2 / Time period)
    Custom Smoothing Function = {Close - Smoothing Function(previous day)} *
    Smoothing Multiplier + Smoothing Function(previous day) 

    PMO Line = 20-period Custom Smoothing of
    (10 * 35-period Custom Smoothing of
    ( ( (Today's Price/Yesterday's Price) * 100) - 100) )

    PMO Signal Line = 10-period EMA of the PMO Line
    '''
    return 'not suport'
    
def dpo(df, period=20):
    '''
    Detrended Price Oscillator (DPO) = Price {X/2 + 1} periods ago - the X-period simple moving average.
    '''
    dpo = df['close'].shift(int(period/2+1))- df['close'].rolling(period).mean()
    return dpo

def emv(df, period=14):
    '''
    Distance Moved = ((H + L)/2 - (Prior H + Prior L)/2) 

    Box Ratio = ((V/100,000,000)/(H - L))

    1-Period EMV = ((H + L)/2 - (Prior H + Prior L)/2) / ((V/100,000,000)/(H - L))

    14-Period Ease of Movement = 14-Period simple moving average of 1-period EMV
    '''
    distanc_moved = ((df['high']+df['low'])/2).diff()
    box_ratio = (df['vol']/1000000)/(df['high']-df['low'])
    period_1_emv = distanc_moved/box_ratio
    emv_period = period_1_emv.rolling(period).mean()
    return emv_period

def force_index(df, period=13):
    '''
    Force Index(1) = {Close (current period)  -  Close (prior period)} x Volume
    Force Index(13) = 13-period EMA of Force Index(1)
    '''
    force_index_1 = df['change'] *df['vol']
    force_index_period = force_index_1.ewm(span=period, adjust=False).mean()
    return force_index_period

def mass_index(df):
    '''
    Single EMA = 9-period exponential moving average (EMA) of the high-low differential  

    Double EMA = 9-period EMA of the 9-period EMA of the high-low differential 

    EMA Ratio = Single EMA divided by Double EMA 

    Mass Index = 25-period sum of the EMA Ratio 
    '''
    single_ema = (df['high']-df['low']).ewm(span=9, adjust=False).mean()
    double_ema = single_ema.ewm(span=9, adjust=False).mean()
    ema_ratio = single_ema / double_ema
    mass_index = ema_ratio.ewm(span=25, adjust=False).mean()
    return mass_index

def macd(df,short_period=12, long_period=26, signal_period=9):
    '''
    MACD Line: (12-day EMA - 26-day EMA)

    Signal Line: 9-day EMA of MACD Line

    MACD Histogram: MACD Line - Signal Line
    '''
    ema_short = df['close'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return pd.DataFrame(dict(macd=macd_line, signal_line=signal_line, macd_histogram=macd_histogram))

def mfi(df):
    '''
    Typical Price = (High + Low + Close)/3

    Raw Money Flow = Typical Price x Volume
    Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)

    Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
    '''
    tp = (df['high'] + df['low'] + df['close'])/3
    raw_money_flow = tp * df['vol']
    #money_flow_ratio = 
    return 'not suport'
    
def nvi(df):
    '''
    1. Cumulative NVI starts at 1000

    2. Add the Percentage Price Change to Cumulative NVI when Volume Decreases

    3. Cumulative NVI is Unchanged when Volume Increases

    4. Apply a 255-day EMA for Signals
    '''
    return 'not suport'

def obv(df):
    '''
    If the closing price is above the prior close price then: 
    Current OBV = Previous OBV + Current Volume

    If the closing price is below the prior close price then: 
    Current OBV = Previous OBV  -  Current Volume

    If the closing prices equals the prior close price then:
    Current OBV = Previous OBV (no change)
    '''
    curretn_obv_list = []
    for index, row in df.iterrows():
        if index == 0:
            curretn_obv = row['vol']
            curretn_obv_list.append(curretn_obv)
            continue
        if row['change']>0:
            curretn_obv = curretn_obv + row['vol']
            curretn_obv_list.append(curretn_obv)
        elif row['change']<0:
            curretn_obv = curretn_obv - row['vol']
            curretn_obv_list.append(curretn_obv)
        else:
            curretn_obv_list.append(curretn_obv)
    return curretn_obv_list

def ppo(df, short_period=12, long_period=26, signal_period=9):
    '''
    Percentage Price Oscillator (PPO): {(12-day EMA - 26-day EMA)/26-day EMA} x 100

    Signal Line: 9-day EMA of PPO

    PPO Histogram: PPO - Signal Line
    '''
    ema_short = df['close'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_period, adjust=False).mean()
    ppo_line = (ema_short - ema_long)/ema_long
    signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
    ppo_histogram = ppo_line - signal_line
    return pd.DataFrame(dict(ppo=ppo_line, signal_line=signal_line, ppo_histogram=ppo_histogram))

def pvo(df, short_period=12, long_period=26, signal_period=9):
    '''
    Percentage Volume Oscillator (PVO): 
    ((12-day EMA of Volume - 26-day EMA of Volume)/26-day EMA of Volume) x 100

    Signal Line: 9-day EMA of PVO

    PVO Histogram: PVO - Signal Line
    '''
    ema_short = df['vol'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['vol'].ewm(span=long_period, adjust=False).mean()
    pvo_line = (ema_short - ema_long)/ema_long
    signal_line = pvo_line.ewm(span=signal_period, adjust=False).mean()
    pvo_histogram = pvo_line - signal_line
    return pd.DataFrame(dict(pvo=pvo_line, signal_line=signal_line, pvo_histogram=pvo_histogram))

def relate_strength(first_df, second_df):
    '''
    Price Relative = Base Security / Comparative Security

    Ratio Symbol Close = Close of First Symbol / Close of Second Symbol
    Ratio Symbol Open  = Open of First Symbol / Close of Second Symbol
    Ratio Symbol High  = High of First Symbol / Close of Second Symbol
    Ratio Symbol Low   = Low of First Symbol / Close of Second Symbol
    '''
    ratio_close = first_df['close']/second_df['close']
    ratio_open = first_df['open']/second_df['open']
    ratio_high = first_df['high']/second_df['high']
    ratio_low = first_df['low']/second_df['low']
    return pd.DataFrame(dict(ratio_close=ratio_close, 
                             ratio_open=ratio_open, 
                             ratio_high=ratio_high, 
                             ratio_low=ratio_low))

def kst(df):
    '''
    RCMA1 = 10-Period SMA of 10-Period Rate-of-Change 
    RCMA2 = 10-Period SMA of 15-Period Rate-of-Change 
    RCMA3 = 10-Period SMA of 20-Period Rate-of-Change 
    RCMA4 = 15-Period SMA of 30-Period Rate-of-Change 

    KST = (RCMA1 x 1) + (RCMA2 x 2) + (RCMA3 x 3) + (RCMA4 x 4)  

    Signal Line = 9-period SMA of KST
    '''
    roc_10 = roc(df, 10)
    rcma1 = roc_10.rolling(10).mean()
    roc_15 = roc(df, 15)
    rcma2 = roc_15.rolling(10).mean()
    roc_20 = roc(df, 20)
    rcma3 = roc_20.rolling(10).mean()
    roc_30 = roc(df, 30)
    rcma4 = roc_30.rolling(15).mean()
    
    kst = rcma1 + rcma2*2 + rcma3*3 + rcma4*4
    kst_signal_line = kst.rolling(9).mean()
    return pd.DataFrame(dict(kst=kst, kst_signal_line=kst_signal_line))

def special_k(df):
    '''
     Special K =  10 Period Simple Moving Average of ROC(10) * 1
                + 10 Period Simple Moving Average of ROC(15) * 2
                + 10 Period Simple Moving Average of ROC(20) * 3
                + 15 Period Simple Moving Average of ROC(30) * 4
                + 50 Period Simple Moving Average of ROC(40) * 1
                + 65 Period Simple Moving Average of ROC(65) * 2
                + 75 Period Simple Moving Average of ROC(75) * 3
                +100 Period Simple Moving Average of ROC(100)* 4
                +130 Period Simple Moving Average of ROC(195)* 1
                +130 Period Simple Moving Average of ROC(265)* 2
                +130 Period Simple Moving Average of ROC(390)* 3
                +195 Period Simple Moving Average of ROC(530)* 4
    '''
    special_k  =  roc(df, 10).rolling(10).mean()
    + roc(df, 15).rolling(10).mean() * 2
    + roc(df, 20).rolling(10).mean() * 3
    + roc(df, 30).rolling(15).mean() * 4
    + roc(df, 40).rolling(50).mean() 
    + roc(df, 65).rolling(65).mean() * 2
    + roc(df, 75).rolling(75).mean() * 3
    + roc(df, 100).rolling(100).mean() * 4
    + roc(df, 195).rolling(130).mean() 
    + roc(df, 265).rolling(130).mean() * 2
    + roc(df, 390).rolling(130).mean() * 3
    if len(df)>725:
        special_k = special_k + roc(df, 530).rolling(195).mean() * 4
    return special_k

def stochastic_oscillator(df, period=14):
    '''
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %K is multiplied by 100 to move the decimal point two places
    '''
    highest = df['high'].rolling(period).max()
    lowest = df['low'].rolling(period).min()
    
    k_percent = (df['close'] - lowest)/(highest - lowest) *100
    d_percent = k_percent.rolling(3).mean()
    return pd.DataFrame(dict(k_percent=k_percent, d_percent=d_percent))