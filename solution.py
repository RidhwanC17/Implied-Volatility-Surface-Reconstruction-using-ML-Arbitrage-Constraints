import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


# Load Data

train = pd.read_csv('train.csv', parse_dates=['date'])
test  = pd.read_csv('test.csv',  parse_dates=['date'])

# Regime Detection

def calculate_regime_features_train(df):
    df  = df.copy()
    atm = (df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)

    atm_iv = (
        df[atm & df['iv_observed'].notna()]
        .groupby('date')['iv_observed']
        .median()
        .rename('atm_iv')
    )

    slope_list = []
    for d in df['date'].unique():
        ddata  = df[(df['date'] == d) & atm & df['iv_observed'].notna()]
        iv1    = ddata[ddata['maturity_label'] == '1M']['iv_observed'].median()
        iv6    = ddata[ddata['maturity_label'] == '6M']['iv_observed'].median()
        slope  = iv6 - iv1 if pd.notna(iv1) and pd.notna(iv6) else np.nan
        slope_list.append({'date': d, 'slope': slope})

    slope_df = pd.DataFrame(slope_list)
    df = df.merge(atm_iv,   on='date', how='left')
    df = df.merge(slope_df, on='date', how='left')
    return df


def calculate_regime_features_test(df, train_median_atm, train_median_slope):
    df  = df.copy()
    atm = (df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)

    atm_iv = (
        df[atm & df['iv_observed'].notna()]
        .groupby('date')['iv_observed']
        .median()
        .rename('atm_iv')
    )

    slope_list = []
    for d in df['date'].unique():
        ddata = df[(df['date'] == d) & atm & df['iv_observed'].notna()]
        iv1   = ddata[ddata['maturity_label'] == '1M']['iv_observed'].median()
        iv6   = ddata[ddata['maturity_label'] == '6M']['iv_observed'].median()
        slope = iv6 - iv1 if pd.notna(iv1) and pd.notna(iv6) else np.nan
        slope_list.append({'date': d, 'slope': slope})

    slope_df = pd.DataFrame(slope_list)
    df = df.merge(atm_iv,   on='date', how='left')
    df = df.merge(slope_df, on='date', how='left')

    # Impute dates where test has no observable ATM options
    df['atm_iv'] = df['atm_iv'].fillna(train_median_atm)
    df['slope']  = df['slope'].fillna(train_median_slope)
    return df


train = calculate_regime_features_train(train)

train_median_atm   = train['atm_iv'].median()
train_median_slope = train['slope'].median()

train['atm_iv'] = train['atm_iv'].fillna(train_median_atm)
train['slope']  = train['slope'].fillna(train_median_slope)

test = calculate_regime_features_test(test, train_median_atm, train_median_slope)

scaler           = StandardScaler()
X_regime_train   = scaler.fit_transform(train[['atm_iv', 'slope']])

kmeans                   = KMeans(n_clusters=3, random_state=42, n_init=10)
train['regime_cluster']  = kmeans.fit_predict(X_regime_train)

X_regime_test           = scaler.transform(test[['atm_iv', 'slope']])
test['regime_cluster']  = kmeans.predict(X_regime_test)


# Feature Engineering

def create_features(df):
    df = df.copy()

    # Smile features
    df['log_m']  = np.log(df['moneyness'])
    df['m2']     = df['moneyness'] ** 2
    df['m3']     = df['moneyness'] ** 3
    df['dist']   = np.abs(df['moneyness'] - 1.0)
    df['dist2']  = df['dist'] ** 2

    # Term-structure features
    df['log_tau']  = np.log(df['tau'])
    df['sqrt_tau'] = np.sqrt(df['tau'])

    # Cross / interaction features
    df['m_tau']          = df['moneyness'] * df['tau']
    df['dist_tau']       = df['dist'] * df['tau']
    df['log_m_log_tau']  = df['log_m'] * df['log_tau']

    # Regime-conditioned features
    df['atm_dist']   = df['atm_iv'] * df['dist']
    df['regime_tau'] = df['atm_iv'] * df['tau']

    # Option type flag
    df['is_call'] = (df['option_type'] == 'call').astype(int)

    # Cyclical calendar features 
    df['dow']   = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df


train = create_features(train)
test  = create_features(test)

features = [
    'moneyness', 'log_m', 'm2', 'm3', 'dist', 'dist2',
    'tau', 'log_tau', 'sqrt_tau',
    'm_tau', 'dist_tau', 'log_m_log_tau',
    'strike', 'spot', 'is_call',
    'atm_iv', 'slope', 'atm_dist', 'regime_tau',
    'regime_cluster', 'dow', 'month'
]

# Train XGBoost

train_clean = train.dropna(subset=features + ['iv_observed'])

X_train = train_clean[features]
y_train = train_clean['iv_observed']

model = xgb.XGBRegressor(
    n_estimators=1100,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2.0,
    reg_lambda=2.0,
    random_state=42,
    tree_method='hist'
)

model.fit(X_train, y_train)

# Predict Missing IVs

test_missing              = test[test['iv_observed'].isna()].copy()
X_test                    = test_missing[features]
preds                     = model.predict(X_test)
test_missing['iv_predicted'] = preds

# Arbitrage-Free Constraints

test_known = test[test['iv_observed'].notna()].copy()
test_all   = pd.concat([test_known, test_missing], ignore_index=True)

test_all['iv_final']       = test_all['iv_observed'].fillna(test_all['iv_predicted'])
test_all['total_variance'] = (test_all['iv_final'] ** 2) * test_all['tau']


def enforce_calendar(df):

    groups    = df.groupby(['date', 'strike', 'option_type'])
    corrected = []

    for _, group in groups:
        group = group.sort_values('tau').copy()
        for i in range(1, len(group)):
            if group.iloc[i]['total_variance'] < group.iloc[i-1]['total_variance']:
                prev_var  = group.iloc[i-1]['total_variance']
                curr_tau  = group.iloc[i]['tau']
                group.iloc[i, group.columns.get_loc('iv_final')] = np.sqrt(prev_var / curr_tau)
        corrected.append(group)

    return pd.concat(corrected)


def smooth_strikes(df):

    groups  = df.groupby(['date', 'maturity_label', 'option_type'])
    smoothed = []

    for _, group in groups:
        group = group.sort_values('strike').copy()
        vals  = group['iv_final'].values

        if len(vals) >= 3:
            for i in range(1, len(vals) - 1):
                vals[i] = 0.85 * vals[i] + 0.15 * 0.5 * (vals[i-1] + vals[i+1])
            group['iv_final'] = vals

        smoothed.append(group)

    return pd.concat(smoothed)


def enforce_put_call_parity(df):

    groups    = df.groupby(['date', 'strike', 'tau'])
    corrected = []

    for _, group in groups:
        group = group.copy()

        if len(group) < 2:
            corrected.append(group)
            continue

        calls = group[group['option_type'] == 'call']
        puts  = group[group['option_type'] == 'put']

        if len(calls) == 1 and len(puts) == 1:
            call_idx = calls.index[0]
            put_idx  = puts.index[0]

            call_iv  = group.loc[call_idx, 'iv_final']
            put_iv   = group.loc[put_idx,  'iv_final']

            alpha = 0.2   # soft blend weight
            group.loc[call_idx, 'iv_final'] = (1 - alpha) * call_iv + alpha * put_iv
            group.loc[put_idx,  'iv_final'] = (1 - alpha) * put_iv  + alpha * call_iv

        corrected.append(group)

    return pd.concat(corrected)


test_all = enforce_calendar(test_all)
test_all = smooth_strikes(test_all)
test_all = enforce_put_call_parity(test_all)

test_final                   = test_all[test_all['iv_observed'].isna()].copy()
test_final['iv_predicted']   = test_final['iv_final'].clip(5.0, 50.0)

# Time-Based Validation

train_sorted = train.sort_values('date')
split_date   = train_sorted['date'].quantile(0.8)

val_data     = train_sorted[train_sorted['date'] > split_date]
val_data     = val_data.dropna(subset=features + ['iv_observed'])

X_val   = val_data[features]
y_val   = val_data['iv_observed']

val_preds = model.predict(X_val)
rmse      = np.sqrt(mean_squared_error(y_val, val_preds))
print(f"RMSE : {rmse:.4f}")

# Submission

submission = test_final[['row_id', 'iv_predicted']].sort_values('row_id')
submission.to_csv('submission.csv', index=False)
print(f"Submission written: {len(submission):,} rows")