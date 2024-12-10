import yfinance as yf
import pandas as pd
import numpy as np
from keras.api.models import Sequential
from keras.api.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Tải dữ liệu cổ phiếu từ Yahoo Finance
ticker = 'AAPL'  # Ticker cổ phiếu (Apple)
start_date = '2014-01-01'  # Ngày bắt đầu
end_date = '2023-12-31'    # Ngày kết thúc

# Tải dữ liệu cổ phiếu
stock = yf.download(ticker, start=start_date, end=end_date)

# Hiển thị thông tin cơ bản về dữ liệu
print(stock.head())

# Chỉ lấy cột 'Close' để làm dự đoán
stock = stock[['Close']]  # Chỉ sử dụng giá đóng cửa

# Chuẩn hóa giá cổ phiếu bằng MinMaxScaler
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock)
stock['Scaled_Close'] = scaled_values

# Tạo các chuỗi dữ liệu (50 ngày lịch sử) và 1 ngày sau đó làm dự đoán
window_size = 50
X = []
Y = []

for i in range(0, len(stock) - window_size - 1, 1):
    first = stock.iloc[i, 0]  # Giá đóng cửa ngày đầu tiên trong cửa sổ
    temp = []
    for j in range(window_size):
        temp.append((stock.iloc[i + j, 0] - first) / first)  # Chuẩn hóa tỷ lệ thay đổi giá cổ phiếu
    temp2 = [(stock.iloc[i + window_size, 0] - first) / first]  # Giá cổ phiếu ngày tiếp theo
    X.append(np.array(temp).reshape(50, 1))  # X chứa 50 ngày dữ liệu
    Y.append(np.array(temp2).reshape(1, 1))  # Y chứa giá cổ phiếu ngày tiếp theo

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_X, test_X, train_label, test_label = train_test_split(X, Y, test_size=0.1, shuffle=False)

train_X = np.array(train_X)
test_X = np.array(test_X)
train_label = np.array(train_label)
test_label = np.array(test_label)

# Reshape lại dữ liệu đầu vào
train_X = train_X.reshape(train_X.shape[0], 50, 1)  # Hình dạng (None, 50, 1)
test_X = test_X.reshape(test_X.shape[0], 50, 1)  # Hình dạng (None, 50, 1)

# Định nghĩa lại mô hình
model = Sequential()

# Định nghĩa lại input_shape cho Conv1D
model.add(Conv1D(128, kernel_size=1, activation='relu', input_shape=(50, 1)))  # Sửa lại ở đây
model.add(MaxPooling1D(2))
model.add(Conv1D(256, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(512, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(2))

# Không cần Flatten nữa, giữ nguyên 3 chiều để cho LSTM hoạt động
model.add(Bidirectional(LSTM(200, return_sequences=True)))  # Đầu vào có dạng (batch_size, timesteps, features)
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200, return_sequences=False)))
model.add(Dropout(0.5))

# Lớp Dense để dự đoán giá trị tiếp theo
model.add(Dense(1, activation='linear'))

# Biên dịch mô hình
model.compile(optimizer='RMSprop', loss='mse')

# Huấn luyện mô hình
model.fit(train_X, train_label, validation_data=(test_X, test_label), epochs=14000, batch_size=64, shuffle=False)

# Đánh giá mô hình
print(model.evaluate(test_X, test_label))

# Dự đoán giá cổ phiếu
predicted = model.predict(test_X)

# Khôi phục giá trị thực từ tỷ lệ chuẩn hóa (chỉ cho cột 'Close')
test_label = test_label.reshape(-1, 1)  # Làm phẳng test_label thành 1 chiều
predicted = predicted.reshape(-1, 1)  # Làm phẳng predicted thành 1 chiều

# Khôi phục giá trị từ chuẩn hóa cho cột 'Close'
test_label = scaler.inverse_transform(test_label)  # Khôi phục giá trị gốc cho test label
predicted = scaler.inverse_transform(predicted)  # Khôi phục giá trị gốc cho giá trị dự đoán

# Vẽ biểu đồ
plt.plot(test_label, color='black', label='Stock Price')
plt.plot(predicted, color='green', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
