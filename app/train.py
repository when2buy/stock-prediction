import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 下载AAPL一年的股票数据
df = yf.download('AAPL', start='2023-01-01', end='2024-01-01', interval='1d')

# 计算百分比变化
df_pct_change = df[['Open', 'High', 'Low', 'Close']].pct_change().dropna()

# 增加交易量等特征（可以选择是否包含交易量特征）
df_pct_change['Volume'] = df['Volume'].iloc[1:].values

# 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_pct_change.values)


# 准备数据集
def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, :4])  # 只使用前四列作为标签，忽略Volume
    return np.array(X), np.array(y)


look_back = 10  # 使用前10天的数据来预测下一天的K线变化
X, y = create_dataset(scaled_data, look_back)

# 划分训练集和测试集（80%训练，20%测试）
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 重新调整模型的输入维度
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out[:, -1, :])  # 取最后一个LSTM输出，并加入dropout层
        out = self.fc(out)   # 预测K线的4个数值（Open, High, Low, Close）
        return out


model = LSTMModel()


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets):
        return self.mse(outputs[:, 0], targets[:, 0]) * 0.5 + \
               self.mse(outputs[:, 1], targets[:, 1]) * 0.2 + \
               self.mse(outputs[:, 2], targets[:, 2]) * 0.2 + \
               self.mse(outputs[:, 3], targets[:, 3]) * 0.1


criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# 在测试集上进行评估
model.eval()
with torch.no_grad():
    predicted_pct_change = model(X_test).cpu().numpy()
    
    # 为了使形状与scaler一致，添加伪Volume列，然后移除
    predicted_with_volume = np.concatenate((predicted_pct_change, np.zeros((predicted_pct_change.shape[0], 1))), axis=1)
    predicted_pct_change = scaler.inverse_transform(predicted_with_volume)[:, :4]
    
    actual_with_volume = np.concatenate((y_test.cpu().numpy(), np.zeros((y_test.shape[0], 1))), axis=1)
    actual_pct_change = scaler.inverse_transform(actual_with_volume)[:, :4]

    # 通过前一天的价格计算原始的价格
    last_prices = df[['Open', 'High', 'Low', 'Close']].values[train_size + look_back:train_size + look_back + len(predicted_pct_change)]
    predicted_prices = last_prices * (1 + predicted_pct_change)

    # 计算评估指标
    mse = np.mean((predicted_prices - (last_prices * (1 + actual_pct_change))) ** 2)
    print(f'MSE on test set: {mse:.4f}')

# 构建DataFrame以便可视化
predicted_df = pd.DataFrame(predicted_prices, columns=['Open', 'High', 'Low', 'Close'])
predicted_df.index = df.index[-len(predicted_prices):]  # 使用与实际数据匹配的日期索引

# 可视化实际的K线图和预测的K线图
appl_actual = df.iloc[-len(predicted_prices):]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 绘制实际的K线图
ax1.set_title("Actual AAPL Stock Price")
mpf.plot(appl_actual, type='candle', ax=ax1)

# 绘制预测的K线图
ax2.set_title("Predicted AAPL Stock Price")
mpf.plot(predicted_df, type='candle', ax=ax2)

plt.show()
