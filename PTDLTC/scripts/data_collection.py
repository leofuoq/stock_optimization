import pandas as pd
import schedule
import time
from vnstock3 import Vnstock
import os

# Đường dẫn đến thư mục 'data'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

# Tạo thư mục 'data' nếu chưa tồn tại
os.makedirs(data_dir, exist_ok=True)

# Đường dẫn đầy đủ đến file CSV
file_path = os.path.join(data_dir, "company_info.csv")

# Hàm lưu DataFrame vào file CSV
def save_to_csv(df, file_path):
    try:
        # Kiểm tra xem file đã tồn tại hay chưa
        if os.path.exists(file_path):
            # Nếu tồn tại, ghi thêm dữ liệu mà không xóa dữ liệu cũ
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Nếu chưa tồn tại, ghi dữ liệu với header
            df.to_csv(file_path, mode='w', header=True, index=False)
        print(f"Dữ liệu đã được lưu vào file {file_path}!")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào CSV: {e}")

# Hàm chính để thu thập dữ liệu và lưu vào CSV
def run_task():
    # Tạo đối tượng Vnstock
    stock = Vnstock().stock(symbol='VN30F1M', source='VCI')

    # Lấy danh sách các ngành và mã chứng khoán theo ICB
    stock_icb = stock.listing.symbols_by_industries()[['symbol', 'organ_name', 'icb_code1']]

    # Lấy dữ liệu các ngành ICB
    icb_all = stock.listing.industries_icb()[['icb_name', 'icb_code']]

    # Tạo bảng result từ các thông tin lấy được
    result2 = pd.merge(stock_icb, icb_all, left_on='icb_code1', right_on='icb_code', how='inner')[['symbol', 'organ_name', 'icb_name']]

    # Lấy tất cả các mã chứng khoán từ các sàn HOSE, HNX, UPCOM, VN30, HNX30
    def get_stocks_by_exchange(stock, exchange):
        symbols = stock.listing.symbols_by_group(exchange).tolist()
        return pd.DataFrame({"symbol": symbols, "exchange": exchange})

    def get_all_stocks(stock):
        exchanges = ['HOSE', 'HNX', 'VN30', 'HNX30']
        stock_dfs = [get_stocks_by_exchange(stock, exchange) for exchange in exchanges]
        return pd.concat(stock_dfs, ignore_index=True)

    # Tạo bảng kết hợp với các mã chứng khoán từ các sàn
    all_stocks_df = get_all_stocks(stock)
    
    # Ghép bảng result2 với all_stocks_df
    result3 = pd.merge(result2, all_stocks_df, on='symbol', how='inner')

    # Lưu kết quả vào CSV
    save_to_csv(result3, file_path)

    print("Công việc đã hoàn thành và dữ liệu được cập nhật!")

# Thiết lập lịch chạy hàng ngày vào lúc 00:00
schedule.every().day.at("08:00").do(run_task)

# Vòng lặp chính để kiểm tra công việc và thực hiện nếu đến giờ
while True:
    schedule.run_pending()
    time.sleep(1)
