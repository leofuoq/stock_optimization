import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from vnstock import Vnstock
import datetime  # Import thêm thư viện datetime để lấy ngày hiện tại
from scipy.optimize import minimize
from pypfopt import EfficientFrontier, risk_models, expected_returns, DiscreteAllocation
from pypfopt import EfficientCVaR, EfficientCDaR, expected_returns, HRPOpt
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

# Đường dẫn đến thư mục data và file CSV
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
file_path = os.path.join(data_dir, "company_info.csv")

# Hàm truy vấn dữ liệu từ file CSV
def fetch_data_from_csv():
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)  # Đọc dữ liệu từ CSV
            return df
        else:
            st.error(f"File {file_path} không tồn tại. Vui lòng kiểm tra lại.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu từ file CSV: {e}")
        return pd.DataFrame()

# Lấy dữ liệu từ file CSV
df = fetch_data_from_csv()
# Tạo đối tượng Vnstock (để lấy dữ liệu giá cổ phiếu)
def create_vnstock_instance():
    return Vnstock().stock(symbol='VN30F1M', source='VCI')
# def fetch_stock_data( start_date, end_date):
#     tickers = st.session_state.selected_stocks
#     data = pd.DataFrame()
#     skipped_tickers = []  # Danh sách các mã bị bỏ qua do không có dữ liệu
#     for ticker in tickers:
#         stock = Vnstock().stock(symbol=ticker, source='VCI')
#         try:
#             stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
#             if stock_data is not None and not stock_data.empty:
#                 stock_data = stock_data[['time', 'close']]
#                 stock_data.columns = ['time', ticker]
#                 stock_data.set_index('time', inplace=True)
#                 # Hợp nhất dữ liệu từng mã vào DataFrame chính
#                 if data.empty:
#                     data = stock_data
#                 else:
#                     data = pd.merge(data, stock_data, how='outer', on='time')  # Dùng merge thay vì join
#             else:
#                 skipped_tickers.append(ticker)
#         except Exception:
#             skipped_tickers.append(ticker)
#     return data, skipped_tickers


def markowitz_optimization(price_data, total_investment):
    tickers = price_data.columns.tolist()  # Đồng bộ tickers với cột của price_data
    num_assets = len(tickers)

    if num_assets == 0:
        st.error("Danh sách mã cổ phiếu đã chọn không hợp lệ. Vui lòng kiểm tra lại.")
        return


    log_ret = np.log(price_data / price_data.shift(1)).dropna()
    n_portfolios = 10000
    all_weights = np.zeros((n_portfolios, num_assets))
    ret_arr = np.zeros(n_portfolios)
    vol_arr = np.zeros(n_portfolios)
    sharpe_arr = np.zeros(n_portfolios)

    mean_returns = log_ret.mean() * 252  # Lợi nhuận kỳ vọng hàng năm
    cov_matrix = log_ret.cov() * 252  # Ma trận hiệp phương sai hàng năm


    np.random.seed(42)  # Thiết lập giá trị seed để kết quả ổn định

    for i in range(n_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights

        ret_arr[i] = np.dot(mean_returns, weights)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]

    max_sharpe_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_sharpe_idx]

    weight2 = dict(zip(tickers, optimal_weights))
    latest_prices = get_latest_prices(tickers)
    latest_prices_series = pd.Series(latest_prices)
    total_portfolio_value = total_investment
    allocation_lp, leftover_lp = run_integer_programming(weight2, latest_prices_series, total_portfolio_value)


    result = {
        "Trọng số danh mục": dict(zip(tickers, optimal_weights)),
        "Lợi nhuận kỳ vọng": ret_arr[max_sharpe_idx],
        "Rủi ro (Độ lệch chuẩn)": vol_arr[max_sharpe_idx],
        "Tỷ lệ Sharpe": sharpe_arr[max_sharpe_idx],
        "Số cổ phiếu cần mua": allocation_lp,
        "Số tiền còn lại": leftover_lp,
        "Giá cổ phiếu": latest_prices
    }
    plot_efficient_frontier(ret_arr, vol_arr, sharpe_arr, all_weights, tickers, max_sharpe_idx, optimal_weights)

    return result



    #return ret_arr, vol_arr, sharpe_arr, optimal_weights, max_sharpe_idx, mean_returns, cov_matrix, all_weights
# Vẽ biểu đồ giá cổ phiếu tương tác
def plot_interactive_stock_chart(data):
    """
    Hàm vẽ biểu đồ giá cổ phiếu tương tác sử dụng Plotly Express.
    """
    tickers = st.session_state.selected_stocks
    if data.empty:
        st.warning("Không có dữ liệu để hiển thị biểu đồ.")
        return

    # Reset index để hiển thị cột 'time' dưới dạng trục X
    data_reset = data.reset_index()
    
    # Định dạng dữ liệu cho biểu đồ dạng dài
    data_long = pd.melt(
        data_reset,
        id_vars=['time'],
        value_vars=tickers,
        var_name='Mã cổ phiếu',
        value_name='Giá đóng cửa'
    )

    # Sử dụng Plotly Express để vẽ biểu đồ
    fig = px.line(
        data_long,
        x='time',
        y='Giá đóng cửa',
        color='Mã cổ phiếu',
        title="Biểu đồ giá cổ phiếu",
        labels={"time": "Thời gian", "Giá đóng cửa": "Giá cổ phiếu (VND)"},
        template="plotly_white"
    )

    # Tuỳ chỉnh giao diện
    fig.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Giá cổ phiếu (VND)",
        legend_title="Mã cổ phiếu",
        hovermode="x unified"
    )

    # Hiển thị biểu đồ trên Streamlit
    st.plotly_chart(fig, width='stretch')

# Vẽ biểu đồ đường biên hiệu quả
def plot_efficient_frontier(ret_arr, vol_arr, sharpe_arr, all_weights, tickers, max_sharpe_idx, optimal_weights):
    # Chuẩn bị thông tin hover
    hover_texts = [
        ", ".join([f"{tickers[j]}: {weight * 100:.2f}%" for j, weight in enumerate(weights)])
        for weights in all_weights
    ]

    fig = px.scatter(
        x=vol_arr,
        y=ret_arr,
        color=sharpe_arr,
        hover_data={
            'Tỷ lệ Sharpe': sharpe_arr,
            'Thông tin danh mục': hover_texts
        },
        labels={'x': 'Rủi ro (Độ lệch chuẩn)', 'y': 'Lợi nhuận kỳ vọng', 'color': 'Tỷ lệ Sharpe'},
        title='Đường biên hiệu quả Markowitz'
    )

    # Đánh dấu danh mục tối ưu
    fig.add_scatter(
        x=[vol_arr[max_sharpe_idx]],
        y=[ret_arr[max_sharpe_idx]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Danh mục tối ưu',
        hovertext=[", ".join([f"{tickers[j]}: {optimal_weights[j] * 100:.2f}%" for j in range(len(tickers))])]
    )
    st.plotly_chart(fig)

# Streamlit UI
df = fetch_data_from_csv()

# Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
def display_selected_stocks(df):
    if st.button("Xóa hết các cổ phiếu"):
        # Xóa danh sách mã cổ phiếu trong cả hai chế độ
        st.session_state.selected_stocks = []
        st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
    if st.session_state.selected_stocks:
        st.markdown("### Danh sách mã cổ phiếu đã chọn:")
        for stock in st.session_state.selected_stocks:
            # Tìm thông tin chi tiết từ DataFrame dựa trên symbol
            stock_info = df[df['symbol'] == stock]
            if not stock_info.empty:
                organ_name = stock_info.iloc[0]['organ_name']
                icb_name = stock_info.iloc[0]['icb_name']
                exchange = stock_info.iloc[0]['exchange']

                # Hiển thị thông tin: Mã cổ phiếu, tên công ty, và ngành
                col1, col2, col3, col4, col5 = st.columns([2, 4, 3, 2, 1])  # Thêm cột để hiển thị thông tin
                 # Hiển thị tiêu đề cho các cột
                col1.write(stock)  # Mã cổ phiếu
                col2.write(organ_name)  # Tên công ty
                col3.write(icb_name)  # Tên ngành
                col4.write(exchange)
                if col5.button(f"❌", key=f"remove_{stock}"):  # Nút xóa
                    st.session_state.selected_stocks.remove(stock)
                    st.rerun()  # Làm mới lại giao diện sau khi xóa
    else:
        st.write("Chưa có mã cổ phiếu nào được chọn.")


def display_selected_stocks_2(df):
    """
    Hiển thị danh sách mã cổ phiếu đã được chọn và cung cấp tùy chọn xóa từng mã hoặc toàn bộ.
    """
    # Nút xóa toàn bộ danh sách cổ phiếu
    if st.button("Xóa hết các cổ phiếu trong danh mục"):
        if "selected_stocks" in st.session_state:
            st.session_state.selected_stocks_2 = []  # Xóa toàn bộ danh sách mã cổ phiếu
            st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
        else:
            st.warning("Không có mã cổ phiếu nào để xóa.")

    # Hiển thị danh sách mã cổ phiếu đã chọn
    if "selected_stocks" in st.session_state and st.session_state.selected_stocks_2:
        st.markdown("### Danh sách cổ phiếu trong danh mục đầu tư:")

        # Duyệt qua từng mã cổ phiếu trong danh sách
        for stock in st.session_state.selected_stocks_2:
            # Lấy thông tin chi tiết từ DataFrame dựa trên mã cổ phiếu
            stock_info = df[df['symbol'] == stock]
            if not stock_info.empty:
                organ_name = stock_info.iloc[0]['organ_name']
                icb_name = stock_info.iloc[0]['icb_name']
                exchange = stock_info.iloc[0]['exchange']

                # Hiển thị thông tin cổ phiếu
                col1, col2, col3, col4, col5 = st.columns([2, 4, 3, 2, 1])
                col1.write(stock)  # Mã cổ phiếu
                col2.write(organ_name)  # Tên công ty
                col3.write(icb_name)  # Tên ngành
                col4.write(exchange)  # Sàn giao dịch
                with col5:
                    if st.button(f"❌", key=f"remove_{stock}"):  # Nút xóa từng mã cổ phiếu
                        st.session_state.selected_stocks_2.remove(stock)
                        st.success(f"Đã xóa mã cổ phiếu '{stock}' khỏi danh sách!")
                        st.rerun()  # Làm mới giao diện sau khi xóa
    else:
        st.info("Chưa có mã cổ phiếu nào được chọn.")

# Hàm tính lợi nhuận kỳ vọng và phương sai
def calculate_metrics(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    volatility = returns.std()
    return mean_returns, volatility

# Hàm vẽ biểu đồ giá cổ phiếu
def plot_interactive_stock_chart(data):
    data_reset = data.reset_index()
    data_long = pd.melt(data_reset, id_vars=['time'], var_name='Mã cổ phiếu', value_name='Giá đóng cửa')
    fig = px.line(data_long, x='time', y='Giá đóng cửa', color='Mã cổ phiếu', title='Biểu đồ giá cổ phiếu')
    st.plotly_chart(fig, width='stretch')

if "selected_stocks_2" not in st.session_state:
    st.session_state.selected_stocks_2 = []  # Chỉ khởi tạo khi chưa tồn tại




def fetch_stock_data2(symbols, start_date, end_date):
    """
    Lấy dữ liệu giá lịch sử cho danh sách cổ phiếu từ Vnstock.
    Args:
        symbols (list): Danh sách mã cổ phiếu.
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD').
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD').

    Returns:
        data (pd.DataFrame): Dữ liệu giá lịch sử, mỗi cổ phiếu là một cột.
        skipped_tickers (list): Danh sách cổ phiếu không tải được dữ liệu.
    """
    data = pd.DataFrame()
    skipped_tickers = []

    def fetch_single_stock(ticker):
        """
        Lấy dữ liệu cho một cổ phiếu.
        """
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            if stock_data is not None and not stock_data.empty:
                stock_data = stock_data[['time', 'close']].rename(columns={'close': ticker})
                stock_data['time'] = pd.to_datetime(stock_data['time'])
                return stock_data.set_index('time')
            else:
                skipped_tickers.append(ticker)
                return pd.DataFrame()
        except Exception as e:
            skipped_tickers.append(ticker)
            print(f"Lỗi khi lấy dữ liệu {ticker}: {e}")
            return pd.DataFrame()

    # Tải dữ liệu song song cho tất cả cổ phiếu
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_single_stock, symbols)

    # Gộp tất cả các DataFrame
    for result in results:
        if not result.empty:
            if data.empty:
                data = result
            else:
                data = pd.merge(data, result, how='outer', on='time')

    # Xử lý giá trị bị thiếu bằng nội suy
    if not data.empty:
        data = data.interpolate(method='linear', limit_direction='both')

    return data, skipped_tickers


# Hàm lấy giá cổ phiếu mới nhất từ vnstock3
def get_latest_prices(tickers):
    latest_prices = {}  # Khởi tạo dictionary để lưu trữ giá cổ phiếu
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)

    for ticker in tickers:
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')  # Tạo đối tượng cổ phiếu
            stock_data = stock.quote.history(start=str(start_date), end=str(end_date))
            if stock_data is not None and not stock_data.empty:
                # Lấy giá đóng cửa (close) của ngày cuối cùng trong dữ liệu
                latest_price = stock_data['close'].iloc[-1] * 1000  # Lấy giá của ngày mới nhất
                latest_prices[ticker] = latest_price  # Lưu vào dictionary
            else:
                print(f"Không có dữ liệu cho cổ phiếu {ticker}")
        except Exception as e:
            print(f"Lỗi khi lấy giá cổ phiếu {ticker}: {e}")
    return latest_prices

def run_integer_programming(weights, latest_prices, total_portfolio_value):
    # Khởi tạo DiscreteAllocation với trọng số và giá cổ phiếu gần nhất
    allocation = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)

    # Sử dụng Integer Programming (LP) để tối ưu phân bổ cổ phiếu
    allocation_lp, leftover_lp = allocation.lp_portfolio(reinvest=False, verbose=True, solver='ECOS_BB')

    return allocation_lp, leftover_lp

# Hàm mô hình Max Sharpe Ratio
def max_sharpe(data,total_investment):
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.max_sharpe()  # Tối ưu hóa tỷ lệ Sharpe
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(weights, latest_prices_series, total_portfolio_value)

        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Max Sharpe: {e}")
        return None

# Hàm mô hình Min Volatility
def min_volatility(data,total_investment):
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        cov_matrix = risk_models.sample_cov(data)

        ef = EfficientFrontier(mean_returns, cov_matrix)
        weights = ef.min_volatility()  # Tối ưu hóa độ lệch chuẩn thấp nhất
        performance = ef.portfolio_performance(verbose=False)
        cleaned_weights = ef.clean_weights()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(weights, latest_prices_series, total_portfolio_value)


        return {
            "Trọng số danh mục": cleaned_weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min Volatility: {e}")
        return None

# Hàm mô hình Min CVaR
def min_cvar(data, total_investment, beta=0.95):
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cvar_optimizer = EfficientCVaR(mean_returns, returns, beta=beta)
        weights = cvar_optimizer.min_cvar()
        performance = cvar_optimizer.portfolio_performance()
        # Tính ma trận hiệp phương sai
        cov_matrix = risk_models.sample_cov(data)

        # Tính độ lệch chuẩn của danh mục
        weights_array = np.array(list(weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf)/ portfolio_std

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(weights, latest_prices_series, total_portfolio_value)

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CVaR": performance[1],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio

        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min CVaR: {e}")
        return None

# Hàm mô hình Min CDaR
def min_cdar(data, total_investment, beta=0.95):
    try:
        mean_returns = expected_returns.mean_historical_return(data)
        returns = expected_returns.returns_from_prices(data).dropna()

        cdar_optimizer = EfficientCDaR(mean_returns, returns, beta=beta)
        weights = cdar_optimizer.min_cdar()
        performance = cdar_optimizer.portfolio_performance()

        cov_matrix = risk_models.sample_cov(data)
        weights_array = np.array(list(weights.values()))
        portfolio_std = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
        rf = 0.02
        sharpe_ratio = (performance[0] - rf)/ portfolio_std

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(weights, latest_prices_series, total_portfolio_value)

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro CDaR": performance[1],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices,
            "Rủi ro (Độ lệch chuẩn)": portfolio_std,
            "Tỷ lệ Sharpe": sharpe_ratio
        }
    except Exception as e:
        print(f"Lỗi trong mô hình Min CDaR: {e}")
        return None
# Hàm Semivariance

# Hàm HRP
def hrp_model(data,total_investment):
    try:
        returns = data.pct_change().dropna(how="all")
        hrp = HRPOpt(returns)
        weights = hrp.optimize(linkage_method="single")
        performance = hrp.portfolio_performance()

        tickers = data.columns.tolist()
        latest_prices = get_latest_prices(tickers)
        latest_prices_series = pd.Series(latest_prices)
        total_portfolio_value = total_investment
        allocation_lp, leftover_lp = run_integer_programming(weights, latest_prices_series, total_portfolio_value)        

        return {
            "Trọng số danh mục": weights,
            "Lợi nhuận kỳ vọng": performance[0],
            "Rủi ro (Độ lệch chuẩn)": performance[1],
            "Tỷ lệ Sharpe": performance[2],
            "Số cổ phiếu cần mua": allocation_lp,
            "Số tiền còn lại": leftover_lp,
            "Giá cổ phiếu": latest_prices
        }
    except Exception as e:
        print(f"Lỗi trong mô hình HRP: {e}")
        return None

import plotly.express as px
import streamlit as st

def backtest_portfolio(symbols, weights, start_date, end_date, benchmark_symbols=["VNINDEX", "VN30", "HNX30", "HNXINDEX"]):
    """
    Hàm backtesting danh mục đầu tư, hỗ trợ nhiều chỉ số benchmark và hiển thị biểu đồ tương tác.

    Args:
        symbols (list): Danh sách mã cổ phiếu trong danh mục.
        weights (list): Trọng số của mỗi mã cổ phiếu.
        start_date (str): Ngày bắt đầu (định dạng 'YYYY-MM-DD').
        end_date (str): Ngày kết thúc (định dạng 'YYYY-MM-DD').
        benchmark_symbols (list): Danh sách các chỉ số benchmark.

    Returns:
        dict: Kết quả backtesting bao gồm Sharpe Ratio, Maximum Drawdown, và lợi suất tích lũy.
    """
    # Lấy dữ liệu giá cổ phiếu trong danh mục
    stock_data, skipped_tickers = fetch_stock_data2(symbols, start_date, end_date)
    if skipped_tickers:
        st.warning(f"Các mã không tải được dữ liệu: {', '.join(skipped_tickers)}")

    if stock_data.empty:
        st.error("Không có dữ liệu để backtesting.")
        return

    # Tính lợi suất hàng ngày của danh mục
    returns = stock_data.pct_change().dropna()
    portfolio_returns = returns.dot(weights)  # Lợi suất danh mục đầu tư
    cumulative_returns = (1 + portfolio_returns).cumprod()  # Lợi suất tích lũy

    # Lấy dữ liệu benchmark
    benchmark_data = {}
    for benchmark in benchmark_symbols:
        benchmark_df, _ = fetch_stock_data2([benchmark], start_date, end_date)
        if not benchmark_df.empty:
            benchmark_returns = benchmark_df.pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns[benchmark]).cumprod()
            benchmark_data[benchmark] = benchmark_cumulative
        else:
            st.warning(f"Không có dữ liệu benchmark cho {benchmark}.")

    # Gộp dữ liệu lợi suất tích lũy của danh mục và các benchmark
    results_df = pd.DataFrame({
        "time": cumulative_returns.index,
        "Danh mục đầu tư": cumulative_returns.values
    }).set_index("time")

    for benchmark, benchmark_cumulative in benchmark_data.items():
        results_df[benchmark] = benchmark_cumulative

    # Chuyển đổi dữ liệu sang dạng dài (long format) để vẽ biểu đồ
    results_df = results_df.reset_index().melt(id_vars=["time"], var_name="Danh mục", value_name="Lợi suất tích lũy")

    # Vẽ biểu đồ lợi suất tích lũy
    fig = px.line(
        results_df,
        x="time",
        y="Lợi suất tích lũy",
        color="Danh mục",
        title="Biểu đồ So sánh Lợi suất Tích lũy",
        labels={"time": "Thời gian", "Lợi suất tích lũy": "Lợi suất (%)"},
        template="plotly_white"
    )
    fig.update_layout(
        xaxis_title="Thời gian",
        yaxis_title="Lợi suất tích lũy (%)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, width='stretch')

    # Tính toán chỉ số hiệu suất
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Cumulative Returns": cumulative_returns,
        "Skipped Tickers": skipped_tickers,
    }

# Tạo session state để lưu mã cổ phiếu đã chọn
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'final_selected_stocks' not in st.session_state:
    st.session_state.final_selected_stocks = {}
# Hàm chạy các mô hình tối ưu hóa
# Chạy các mô hình tối ưu hóa
def run_models(data):
    """
    Hàm xử lý các chiến lược tối ưu hóa danh mục và tích hợp backtesting tự động.
    """
    if data.empty:
        st.error("Dữ liệu cổ phiếu bị thiếu hoặc không hợp lệ.")
        return
    st.sidebar.title("Chọn chiến lược đầu tư")
    total_investment = st.sidebar.number_input(
        "Nhập số tiền đầu tư (VND)", 
        min_value=1000, 
        value=1000000, 
        step=100000,
        key="number_input_2"
    )

    models = {
        "Tối ưu hóa giữa lợi nhuận và rủi ro": {"function": markowitz_optimization, "original_name": "Mô hình Markowitz"},
        "Hiệu suất tối đa": {"function": max_sharpe, "original_name": "Mô hình Max Sharpe Ratio"},
        "Đầu tư an toàn": {"function": min_volatility, "original_name": "Mô hình Min Volatility"},
        "Đa dạng hóa thông minh": {"function": hrp_model, "original_name": "Mô hình HRP"},
        "Phòng ngừa tổn thất cực đại": {"function": min_cvar, "original_name": "Mô hình Min CVaR"},
        "Kiểm soát tổn thất kéo dài": {"function": min_cdar, "original_name": "Mô hình Min CDaR"},
    }

    for strategy_name, model_details in models.items():
        if st.sidebar.button(f"Chiến lược {strategy_name}"):
            try:
                # Chạy mô hình tối ưu hóa
                result = model_details["function"](data, total_investment)
                if result:
                    # Hiển thị kết quả tối ưu hóa
                    display_results(model_details["original_name"],result)

                    # Lấy thông tin cổ phiếu và trọng số từ kết quả
                    symbols = list(result["Trọng số danh mục"].keys())
                    weights = list(result["Trọng số danh mục"].values())

                    # Chạy backtesting ngay sau tối ưu hóa
                    st.subheader("Kết quả Backtesting")
                    with st.spinner("Đang chạy Backtesting..."):
                        end_date = datetime.date.today()
                        start_date = end_date - datetime.timedelta(days=1852)
                        backtest_result = backtest_portfolio(symbols, weights, start_date, end_date)

                        # Hiển thị kết quả backtesting
                        if backtest_result:
                            st.write(f"Mean Sharpe Ratio: {backtest_result['Sharpe Ratio']:.2f}")
                            st.write(f"Maximum Drawdown: {backtest_result['Maximum Drawdown']:.2%}")

                            # Biểu đồ tương tác lợi suất tích lũy
                            cumulative_returns = backtest_result["Cumulative Returns"]
                            # Chuyển cumulative_returns thành DataFrame với cột tên rõ ràng
                            cumulative_returns_df = pd.DataFrame({
                                "time": cumulative_returns.index,
                                "Cumulative Returns": cumulative_returns.values
                            })

                        else:
                            st.error("Không thể thực hiện Backtesting. Vui lòng kiểm tra dữ liệu đầu vào.")
                else:
                    st.error(f"Không thể chạy {strategy_name}.")
            except Exception as e:
                st.error(f"Lỗi khi chạy {strategy_name}: {e}")


# Hàm hiển thị kết quả với giao diện đẹp
def display_results(original_name, result):
    if result:
        st.markdown(f"## {original_name}")
        st.markdown("### Hiệu suất danh mục:")
        
        # Lợi nhuận kỳ vọng
        st.write(f"- **Lợi nhuận kỳ vọng:** {result.get('Lợi nhuận kỳ vọng', 0):.2%}")
        
        # Rủi ro (Độ lệch chuẩn)
        risk_std = result.get('Rủi ro (Độ lệch chuẩn)', 0)
        if risk_std == 0:
            st.write("- **Rủi ro (Độ lệch chuẩn):** Chỉ số không áp dụng cho mô hình này")
        else:
            st.write(f"- **Rủi ro (Độ lệch chuẩn):** {risk_std:.2%}")
        
        # Rủi ro CVaR
        if "Rủi ro CVaR" in result:
            st.write(f"- **Mức tổn thất trung bình trong tình huống xấu nhất:** {result['Rủi ro CVaR']:.2%}")
        
        # Rủi ro CDaR
        if "Rủi ro CDaR" in result:
            st.write(f"- **Mức giảm giá trị trung bình trong giai đoạn có sự giảm giá trị sâu:** {result['Rủi ro CDaR']:.2%}")
        
        # Tỷ lệ Sharpe
        sharpe_ratio = result.get('Tỷ lệ Sharpe', 0)
        if sharpe_ratio == 0:
            st.write("- **Tỷ lệ Sharpe:** Chỉ số không áp dụng cho mô hình này")
        else:
            st.write(f"- **Tỷ lệ Sharpe:** {sharpe_ratio:.2f}")


        # Trọng số danh mục
        weights = result["Trọng số danh mục"]
        tickers = list(weights.keys())

        # Tạo bảng trọng số
        weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Trọng số (%)"])
        weights_df["Trọng số (%)"] = weights_df["Trọng số (%)"] * 100

        # Giá cổ phiếu và phân bổ cổ phiếu
        latest_prices = result.get("Giá cổ phiếu", {})
        allocation = result.get("Số cổ phiếu cần mua", {})

        # Nếu không có phân bổ, mặc định là 0
        allocation = {ticker: allocation.get(ticker, 0) for ticker in tickers}
        latest_prices = {ticker: latest_prices.get(ticker, 0) for ticker in tickers}

        # Tạo DataFrame kết hợp các thông tin
        combined_data = {
            "Cổ phiếu": tickers,
            "Giá cổ phiếu": [f"{latest_prices.get(ticker, 0):.2f}" for ticker in tickers],
            "Trọng số (%)": [f"{weights_df.loc[ticker, 'Trọng số (%)']:.2f}" for ticker in tickers],
            "Số cổ phiếu cần mua": [allocation.get(ticker, 0) for ticker in tickers]
        }
        
        # Chuyển đổi thành DataFrame và hiển thị
        combined_df = pd.DataFrame(combined_data)

        # Hiển thị bảng kết hợp
        st.markdown("### Bảng phân bổ danh mục đầu tư:")
        st.table(combined_df)
        st.write(f"- **Số tiền còn lại:** {round(result.get('Số tiền còn lại', 0))}")





# Sidebar
st.sidebar.title("Lựa chọn phương thức")

# Tùy chọn giữa "Chủ động" và "Thụ động"
option = st.sidebar.radio("Chọn phương thức", ["Tự chọn cổ phiếu", "Hệ thống đề xuất cổ phiếu tự động"])

if option == "Tự chọn cổ phiếu":

    # Giao diện người dùng để lọc từ file CSV
    st.title("Dashboard hỗ trợ tối ưu hóa danh mục đầu tư chứng khoán")
    # Sidebar
    st.sidebar.title("Bộ lọc và Cấu hình")
    # Bộ lọc theo sàn giao dịch (exchange)
    selected_exchange = st.sidebar.selectbox('Chọn sàn giao dịch', df['exchange'].unique())

    # Lọc dữ liệu dựa trên sàn giao dịch đã chọn
    filtered_df = df[df['exchange'] == selected_exchange]

    # Bộ lọc theo loại ngành (icb_name)
    selected_icb_name = st.sidebar.selectbox('Chọn ngành', filtered_df['icb_name'].unique())

    # Lọc dữ liệu dựa trên ngành đã chọn
    filtered_df = filtered_df[filtered_df['icb_name'] == selected_icb_name]

    # Bộ lọc theo mã chứng khoán (symbol)
    selected_symbols = st.sidebar.multiselect('Chọn mã chứng khoán', filtered_df['symbol'])

    # Lưu các mã chứng khoán đã chọn vào session state khi nhấn nút "Thêm mã"
    if st.sidebar.button("Thêm mã vào danh sách"):
        for symbol in selected_symbols:
            if symbol not in st.session_state.selected_stocks:
                st.session_state.selected_stocks.append(symbol)
        st.sidebar.success(f"Đã thêm {len(selected_symbols)} mã cổ phiếu vào danh mục!")

    # Hiển thị danh sách mã cổ phiếu đã chọn và xử lý thao tác xóa
    display_selected_stocks(df)


    # Lựa chọn thời gian lấy dữ liệu
    today = datetime.date.today()  # Ngày hiện tại
    start_date = st.sidebar.date_input("Ngày bắt đầu", value=pd.to_datetime("2020-01-01"),max_value=today)
    end_date = st.sidebar.date_input("Ngày kết thúc", value=pd.to_datetime("2023-01-01"),max_value=today)
    # Kiểm tra ngày bắt đầu và ngày kết thúc
    if start_date > today or end_date > today:
        st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
    elif start_date > end_date:
        st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
    else:
        st.sidebar.success("Ngày tháng hợp lệ.")
    
    def main2():
        st.title("Tối ưu hóa danh mục đầu tư")
        # Kiểm tra session state và lấy danh sách cổ phiếu đã chọn
        if "selected_stocks" in st.session_state and st.session_state.selected_stocks:
            selected_stocks = st.session_state.selected_stocks
            # Lấy dữ liệu giá cổ phiếu
            data, skipped_tickers = fetch_stock_data2(selected_stocks, start_date, end_date)

            if not data.empty:
                st.subheader("Giá cổ phiếu")
                # Vẽ biểu đồ giá cổ phiếu
                plot_interactive_stock_chart(data)
                # Chạy các mô hình
                run_models(data)

            else:
                st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có.")
        else:
            st.warning("Chưa có mã cổ phiếu nào trong danh mục. Vui lòng chọn mã cổ phiếu trước.")

    # Gọi hàm chính
    if __name__ == "__main__":
        main2()
                
elif option == "Hệ thống đề xuất cổ phiếu tự động":
    # Lựa chọn mục tiêu đầu tư
    # Sidebar
   # Giao diện Streamlit
    st.title("Hệ thống đề xuất cổ phiếu")
    st.sidebar.title("Cấu hình đề xuất cổ phiếu")

    # Bước 1: Chọn sàn giao dịch
    df = fetch_data_from_csv()  # Dữ liệu từ database

    # Bước 1: Chọn sàn giao dịch
    if not df.empty:
        selected_exchanges = st.sidebar.multiselect("Chọn sàn giao dịch", df['exchange'].unique(), default=None)

        # Lọc theo sàn giao dịch
    # Lọc dữ liệu theo nhiều sàn giao dịch
        filtered_df = df[df['exchange'].isin(selected_exchanges)]

        # Bước 2: Chọn nhiều ngành
        selected_sectors = st.sidebar.multiselect("Chọn ngành", filtered_df['icb_name'].unique())

        if selected_sectors:
            # Lọc theo các ngành đã chọn
            sector_df = filtered_df[filtered_df['icb_name'].isin(selected_sectors)]

            # Bước 3: Chọn số lượng cổ phiếu cho từng ngành
            stocks_per_sector = {}
            for sector in selected_sectors:
                num_stocks = st.sidebar.number_input(f"Số cổ phiếu muốn đầu tư trong ngành '{sector}'", min_value=1, max_value=10, value=3)
                stocks_per_sector[sector] = num_stocks

            # Bước 4: Chọn cách lọc
            filter_method = st.sidebar.radio("Cách lọc cổ phiếu", ["Lợi nhuận lớn nhất", "Rủi ro bé nhất"])

            # Lựa chọn thời gian lấy dữ liệu
            today = datetime.date.today()  # Ngày hiện tại
            start_date = st.sidebar.date_input("Ngày bắt đầu", value=datetime.date(2020, 1, 1),key="start_date_1")
            end_date = st.sidebar.date_input("Ngày kết thúc", value=datetime.date(2023, 1, 1),key="end_date_1")
            # Khởi tạo session state nếu chưa có
            # Kiểm tra ngày bắt đầu và ngày kết thúc
            if start_date > today or end_date > today:
                st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
            elif start_date > end_date:
                st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
            else:
                st.sidebar.success("Ngày tháng hợp lệ.")

            # Bộ lọc và xử lý nhiều sàn, nhiều ngành, và đề xuất cổ phiếu
            if st.sidebar.button("Đề xuất cổ phiếu"):
                final_selected_stocks = {}  # Lưu kết quả cuối cùng theo sàn và ngành

                for exchange in selected_exchanges:  # Duyệt qua từng sàn được chọn
                    st.subheader(f"Sàn giao dịch: {exchange}")
                    exchange_df = df[df['exchange'] == exchange]  # Lọc dữ liệu theo sàn

                    for sector, num_stocks in stocks_per_sector.items():  # Duyệt qua từng ngành
                        # Lọc cổ phiếu theo ngành trong từng sàn
                        sector_df = exchange_df[exchange_df['icb_name'] == sector]

                        if sector_df.empty:
                            st.warning(f"Không có cổ phiếu nào trong ngành '{sector}' của sàn '{exchange}' để phân tích.")
                            continue

                        symbols = sector_df['symbol'].tolist()

                        # Kéo dữ liệu giá cổ phiếu
                        data, skipped_tickers = fetch_stock_data2(symbols, start_date, end_date)

                        if data.empty:
                            st.warning(f"Không có dữ liệu giá cổ phiếu cho ngành '{sector}' của sàn '{exchange}'.")
                            continue

                        # Tính toán lợi nhuận kỳ vọng và phương sai
                        mean_returns, volatility = calculate_metrics(data)

                        # Tạo DataFrame kết quả
                        stock_analysis = pd.DataFrame({
                            "Mã cổ phiếu": mean_returns.index,
                            "Lợi nhuận kỳ vọng (%)": mean_returns.values * 100,
                            "Rủi ro (Phương sai)": volatility.values * 100
                        })

                        # Hiển thị lợi nhuận kỳ vọng và rủi ro của từng cổ phiếu
                        #st.subheader(f"Lợi nhuận và rủi ro của ngành '{sector}' (Sàn '{exchange}')")
                        #st.write(stock_analysis)

                        # Lọc cổ phiếu theo cách lọc và số lượng
                        if filter_method == "Lợi nhuận lớn nhất":
                            selected_stocks = stock_analysis.nlargest(num_stocks, "Lợi nhuận kỳ vọng (%)")["Mã cổ phiếu"].tolist()
                        elif filter_method == "Rủi ro bé nhất":
                            selected_stocks = stock_analysis.nsmallest(num_stocks, "Rủi ro (Phương sai)")["Mã cổ phiếu"].tolist()

                        # Lưu cổ phiếu được chọn theo sàn và ngành vào session_state
                        if exchange not in st.session_state.final_selected_stocks:
                            st.session_state.final_selected_stocks[exchange] = {}
                        st.session_state.final_selected_stocks[exchange][sector] = selected_stocks

            # Hiển thị danh mục cổ phiếu được lọc
    if st.session_state.final_selected_stocks:
        st.subheader("Danh mục cổ phiếu được lọc theo sàn và ngành")
        if st.button("Xóa hết các cổ phiếu đã được đề xuất"):
            if "selected_stocks" in st.session_state:
                st.session_state.final_selected_stocks = {}  # Xóa toàn bộ danh sách mã cổ phiếu
                st.success("Đã xóa hết tất cả cổ phiếu khỏi danh sách!")
            else:
                st.warning("Không có mã cổ phiếu nào để xóa.")
        for exchange, sectors in st.session_state.final_selected_stocks.items():
            st.write(f"### Sàn: {exchange}")
            for sector, stocks in sectors.items():
                st.write(f"#### Ngành: {sector}")
                for stock in stocks:
                    col1, col2 = st.columns([4, 1])  # Cột hiển thị và nút thêm
                    with col1:
                        st.write(f"- {stock}")
                    with col2:
                        if st.button("➕ Thêm", key=f"add_{exchange}_{sector}_{stock}"):
                            if stock not in st.session_state.selected_stocks_2:
                                st.session_state.selected_stocks_2.append(stock)
                                #st.session_state.selected_stocks.append(stock)                                       
                                st.success(f"Đã thêm mã cổ phiếu '{stock}' vào danh sách.")
                            else:
                                st.warning(f"Mã cổ phiếu '{stock}' đã tồn tại trong danh sách.")

            # Hiển thị danh sách mã cổ phiếu đã chọn
    display_selected_stocks_2(df)

    # Hàm chính để kiểm tra và chạy mô hình
    def main1():
        st.title("Tối ưu hóa danh mục đầu tư")
        # Kiểm tra session state và lấy danh sách cổ phiếu đã chọn
        if "selected_stocks_2" in st.session_state and st.session_state.selected_stocks_2:
            selected_stocks_2 = st.session_state.selected_stocks_2
            st.sidebar.title("Chọn thời gian tính toán")
            today = datetime.date.today()
            start_date_2 = st.sidebar.date_input("Ngày bắt đầu", value=datetime.date(2020, 1, 1), key = "start_date_2")
            end_date_2 = st.sidebar.date_input("Ngày kết thúc", value=datetime.date(2023, 1, 1), key = "end_date_2")
            # Kiểm tra ngày bắt đầu và ngày kết thúc
            if start_date_2 > today or end_date_2 > today:
                st.sidebar.error("Ngày bắt đầu và ngày kết thúc không được vượt quá ngày hiện tại.")
            elif start_date_2 > end_date_2:
                st.sidebar.error("Ngày bắt đầu không thể lớn hơn ngày kết thúc.")
            else:
                st.sidebar.success("Ngày tháng hợp lệ.")            
            # Lấy dữ liệu giá cổ phiếu
            data, skipped_tickers = fetch_stock_data2(selected_stocks_2, start_date_2, end_date_2)

            if not data.empty:
                st.subheader("Giá cổ phiếu")
                # Vẽ biểu đồ giá cổ phiếu
                plot_interactive_stock_chart(data)
                # Chạy các mô hình
                run_models(data)

            else:
                st.error("Dữ liệu cổ phiếu bị thiếu hoặc không có.")
        else:
            st.warning("Chưa có mã cổ phiếu nào trong danh mục. Vui lòng chọn mã cổ phiếu trước.")

    # Gọi hàm chính
    if __name__ == "__main__":
        main1()






