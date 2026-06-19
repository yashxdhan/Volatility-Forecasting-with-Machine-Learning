# ============================================================
# VOLATILITY FORECASTING WITH MACHINE LEARNING
# Target: EURO STOXX 50 — Predicting 21-day Realized Volatility
# Author: Yashodhan Sharma
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. LOAD & CLEAN DATA
# ============================================================

data = pd.read_csv(r'c:/Users/us536/Documents/development/vol_forecasting/EUSTOXX_50.csv')

data.columns = data.columns.str.strip()

print("Raw columns:", data.columns.tolist())
print("First 3 rows:")
print(data.head(3))

data['Date'] = pd.to_datetime(data['Date'], format='mixed', dayfirst=False)
data = data.set_index('Date')

for col in ['Price', 'Open', 'High', 'Low']:
    data[col] = data[col].astype(str).str.replace(',', '').astype(float)

if 'Vol.' in data.columns:
    def parse_volume(val):
        if pd.isna(val) or str(val).strip() in ['', '-']:
            return np.nan
        val = str(val).strip().upper()
        if 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        elif 'K' in val:
            return float(val.replace('K', '')) * 1_000
        elif 'B' in val:
            return float(val.replace('B', '')) * 1_000_000_000
        else:
            try:
                return float(val.replace(',', ''))
            except:
                return np.nan

    data['Volume'] = data['Vol.'].apply(parse_volume)
    data = data.drop(columns=['Vol.'])
else:
    data['Volume'] = np.nan

if 'Change %' in data.columns:
    data = data.drop(columns=['Change %'])

data = data.rename(columns={'Price': 'Close'})

data = data.sort_index(ascending=True)

has_volume = data['Volume'].notna().sum() > 0

print(f"\nCleaned data shape: {data.shape}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Has volume data: {has_volume}")
print(f"\nFirst 5 rows:")
print(data.head())

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df = pd.DataFrame(index=data.index)

df['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

forward_returns = pd.DataFrame({
    f'ret_t+{i}': df['log_return'].shift(-i) for i in range(1, 22)
})
df['target_rv_21'] = forward_returns.std(axis=1) * np.sqrt(252)

df['rv_5d'] = df['log_return'].rolling(5).std() * np.sqrt(252)
df['rv_10d'] = df['log_return'].rolling(10).std() * np.sqrt(252)
df['rv_21d'] = df['log_return'].rolling(21).std() * np.sqrt(252)
df['rv_63d'] = df['log_return'].rolling(63).std() * np.sqrt(252)

df['return_1d'] = df['log_return']
df['return_5d'] = df['log_return'].rolling(5).sum()
df['return_21d'] = df['log_return'].rolling(21).sum()
df['abs_return_1d'] = df['log_return'].abs()
df['abs_return_5d'] = df['log_return'].abs().rolling(5).mean()

df['high_low_range'] = np.log(data['High'] / data['Low'])
df['parkinson_vol'] = (
    df['high_low_range'].rolling(21).apply(
        lambda x: np.sqrt(1 / (4 * 21 * np.log(2)) * (x**2).sum())
    ) * np.sqrt(252)
)

if has_volume:
    df['volume'] = data['Volume']
    df['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(21).mean()
    df['volume_std_21d'] = data['Volume'].rolling(21).std()

df['vol_of_vol'] = df['rv_21d'].rolling(21).std()

df['negative_return_5d'] = (df['log_return'].rolling(5).apply(
    lambda x: (x[x < 0]**2).sum() if len(x[x < 0]) > 0 else 0
)) * np.sqrt(252)

df['day_of_week'] = data.index.dayofweek
df['month'] = data.index.month

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

df['vol_regime'] = (df['rv_21d'] > df['rv_63d']).astype(int)

df['rv_5d_21d_ratio'] = df['rv_5d'] / df['rv_21d']
df['rv_21d_63d_ratio'] = df['rv_21d'] / df['rv_63d']
df['max_return_5d'] = df['log_return'].rolling(5).max()
df['min_return_5d'] = df['log_return'].rolling(5).min()
df['return_range_5d'] = df['max_return_5d'] - df['min_return_5d']

df = df.dropna()

print(f"\nFinal dataset shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Number of features: {len([c for c in df.columns if c != 'target_rv_21'])}")
print(f"\nFeatures:")
for i, col in enumerate([c for c in df.columns if c != 'target_rv_21'], 1):
    print(f"  {i:2d}. {col}")

# ============================================================
# 3. WALK-FORWARD TRAIN/TEST SPLIT
# ============================================================

feature_cols = [c for c in df.columns if c != 'target_rv_21']
X = df[feature_cols]
y = df['target_rv_21']

split_date = '2020-01-01'

X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]
y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]

print(f"\n{'='*50}")
print(f"  TRAIN/TEST SPLIT")
print(f"{'='*50}")
print(f"Training: {X_train.shape[0]} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
print(f"Testing:  {X_test.shape[0]} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
print(f"\nTarget stats (train): mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Target stats (test):  mean={y_test.mean():.4f}, std={y_test.std():.4f}")

# ============================================================
# 4. BENCHMARK: HAR MODEL (Corsi, 2009)
# ============================================================

har_features = ['rv_5d', 'rv_10d', 'rv_21d', 'rv_63d']

har_model = LinearRegression()
har_model.fit(X_train[har_features], y_train)
y_pred_har = har_model.predict(X_test[har_features])

print(f"\n{'='*50}")
print(f"  HAR MODEL (Benchmark)")
print(f"{'='*50}")
print("Coefficients:")
for feat, coef in zip(har_features, har_model.coef_):
    print(f"  {feat}: {coef:.4f}")
print(f"  Intercept: {har_model.intercept_:.4f}")

# ============================================================
# 5. RANDOM FOREST
# ============================================================

print(f"\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("Top 10 Features (Random Forest):")
for feat, imp in importance.head(10).items():
    print(f"  {feat}: {imp:.4f}")

# ============================================================
# 6. GRADIENT BOOSTING (replaces XGBoost)
# ============================================================

print(f"\nTraining Gradient Boosting...")

gb_model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)

gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# ============================================================
# 7. LSTM NEURAL NETWORK (PyTorch)
# ============================================================

print(f"\nTraining LSTM...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

SEQUENCE_LENGTH = 21


def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


X_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train.values, SEQUENCE_LENGTH
)
X_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test.values, SEQUENCE_LENGTH
)

X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)


class VolatilityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(VolatilityLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = VolatilityLSTM(
    input_size=X_train_seq.shape[2],
    hidden_size=64,
    num_layers=2,
    dropout=0.2
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5
)

EPOCHS = 100
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

model.eval()
with torch.no_grad():
    y_pred_lstm = model(X_test_tensor.to(device)).cpu().numpy().flatten()

# ============================================================
# 8. MODEL EVALUATION
# ============================================================

test_dates_lstm = y_test.index[SEQUENCE_LENGTH:]


def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        dir_accuracy = np.mean(actual_direction == pred_direction)
    else:
        dir_accuracy = np.nan

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  RMSE:                  {rmse:.4f}")
    print(f"  MAE:                   {mae:.4f}")
    print(f"  R²:                    {r2:.4f}")
    print(f"  Directional Accuracy:  {dir_accuracy:.2%}")

    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae,
            'R2': r2, 'Dir_Accuracy': dir_accuracy}

test_dates_lstm = y_test.index[SEQUENCE_LENGTH:]

results = []
results.append(evaluate_model(y_test.values, y_pred_har, "HAR (Benchmark)"))
results.append(evaluate_model(y_test.values, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test.values, y_pred_gb, "Gradient Boosting"))
results.append(evaluate_model(y_test_seq, y_pred_lstm, "LSTM (PyTorch)"))

results_df = pd.DataFrame(results).set_index('Model')
print("\n\n" + "=" * 60)
print("  FINAL SUMMARY — MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string())

# ============================================================
# 9. VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle(
    'Volatility Forecasting: EURO STOXX 50\n'
    'ML Models vs. HAR Benchmark | Yashodhan Sharma',
    fontsize=16, fontweight='bold', y=1.02
)

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.plot(y_test.index, y_test.values, color='black', alpha=0.7,
        linewidth=1, label='Actual RV (21d)')
ax.plot(y_test.index, y_pred_har, alpha=0.7, linewidth=1, label='HAR')
ax.plot(y_test.index, y_pred_rf, alpha=0.7, linewidth=1, label='Random Forest')
ax.plot(y_test.index, y_pred_gb, alpha=0.7, linewidth=1, label='Gradient Boosting')
ax.plot(test_dates_lstm, y_pred_lstm, alpha=0.7, linewidth=1, label='LSTM')
ax.set_title('Predicted vs Actual Realized Volatility')
ax.set_ylabel('Annualized Volatility')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: RMSE Bar Chart
ax = axes[0, 1]
colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']
results_df['RMSE'].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
ax.set_title('RMSE Comparison (Lower is Better)')
ax.set_ylabel('RMSE')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Feature Importance
ax = axes[1, 0]
importance.head(10).plot(kind='barh', ax=ax, color='#3498db', edgecolor='black')
ax.set_title('Top 10 Features (Random Forest)')
ax.set_xlabel('Importance')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Gradient Boosting Error Over Time
ax = axes[1, 1]
error_gb = y_test.values - y_pred_gb
ax.fill_between(y_test.index, error_gb, 0, alpha=0.5, color='#2ecc71')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Gradient Boosting Prediction Error Over Time')
ax.set_ylabel('Error (Actual - Predicted)')
ax.grid(True, alpha=0.3)

# Plot 5: Scatter — Gradient Boosting
ax = axes[2, 0]
ax.scatter(y_test.values, y_pred_gb, alpha=0.3, s=10, color='#2ecc71')
min_val = min(y_test.min(), min(y_pred_gb))
max_val = max(y_test.max(), max(y_pred_gb))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
ax.set_xlabel('Actual Realized Volatility')
ax.set_ylabel('Predicted Realized Volatility')
ax.set_title('Gradient Boosting: Predicted vs Actual')
ax.grid(True, alpha=0.3)

# Plot 6: LSTM Training Loss
ax = axes[2, 1]
ax.plot(train_losses, color='#e74c3c', linewidth=1)
ax.set_title('LSTM Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('volatility_forecast_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nChart saved as 'volatility_forecast_results.png'")

# ============================================================
# 10. SAVE RESULTS
# ============================================================

output = pd.DataFrame({
    'actual_rv_21d': y_test.values,
    'pred_har': y_pred_har,
    'pred_rf': y_pred_rf,
    'pred_gb': y_pred_gb,
}, index=y_test.index)

output.to_csv('volatility_predictions.csv')
results_df.to_csv('model_comparison.csv')

print("Predictions saved to 'volatility_predictions.csv'")
print("Model comparison saved to 'model_comparison.csv'")
print("\n✅ Project complete.")