import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO")

# --- HÀM TÍNH TOÁN ---
def calculate_indicators(df):
    # 1. BB
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. ADX/DI
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    df['TR14'] = df['TR'].ewm(alpha=1/14, adjust=False).mean()
    df['+DM14'] = df['+DM'].ewm(alpha=1/14, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=1/14, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
    df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()
    
    return df

# --- LOGIC MUA BÁN ---
def analyze_strategy(df):
    if len(df) < 25: return "Không đủ dữ liệu", "NEUTRAL", "gray"
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Trigger conditions
    buy_trigger = (curr['Close'] <= curr['Lower'] * 1.01) and (curr['RSI'] < 30)
    sell_trigger = (curr['Close'] >= curr['Upper'] * 0.99) and (curr['RSI'] > 70)
    
    rec, reason, color = "QUAN SÁT (HOLD)", "Chưa có tín hiệu.", "blue"
    
    # CHIẾN LƯỢC MUA
    if buy_trigger:
        if curr['ADX'] < 25:
            if (curr['-DI'] > curr['+DI']) and (curr['-DI'] < prev['-DI']):
                rec, reason, color = "MUA NGAY", "Giá chạm đáy, RSI thấp, ADX yếu. DI- đang suy giảm.", "green"
            else:
                rec, reason, color = "CHỜ MUA", "Giá tốt nhưng lực bán chưa giảm nhiệt.", "orange"
        elif curr['ADX'] > 50:
            cooling = (curr['ADX'] < prev['ADX'] < prev2['ADX']) and (curr['-DI'] < prev['-DI'] < prev2['-DI'])
            if cooling:
                rec, reason, color = "MUA NGAY", "Bắt đáy sau đợt sập mạnh (ADX và DI- giảm 2 phiên).", "green"
            else:
                rec, reason, color = "ĐỨNG NGOÀI", f"Đang sập mạnh (ADX={curr['ADX']:.1f}). Chờ tín hiệu giảm nhiệt.", "red"
        else:
             if (curr['-DI'] > curr['+DI']) and (curr['-DI'] < prev['-DI']):
                rec, reason, color = "MUA THĂM DÒ", "Giá rẻ, xu hướng giảm trung bình.", "green"

    # CHIẾN LƯỢC BÁN
    elif sell_trigger:
        if curr['ADX'] < 25:
             if (curr['+DI'] > curr['-DI']) and (curr['+DI'] < prev['+DI']):
                rec, reason, color = "BÁN NGAY", "Giá đỉnh, RSI cao, lực tăng yếu.", "red"
        elif curr['ADX'] > 50:
            cooling = (curr['ADX'] < prev['ADX'] < prev2['ADX']) and (curr['+DI'] < prev['+DI'] < prev2['+DI'])
            if cooling:
                rec, reason, color = "BÁN CHỐT LỜI", "Siêu sóng kết thúc (ADX và DI+ giảm 2 phiên).", "red"
            else:
                rec, reason, color = "NẮM GIỮ", f"Trend tăng cực mạnh (ADX={curr['ADX']:.1f}). Gồng lãi tiếp.", "green"
        else:
             rec, reason, color = "CÂN NHẮC BÁN", "Vùng quá mua.", "orange"
             
    return rec, reason, color

# --- GIAO DIỆN ---
st.title("📈 Stock Advisor PRO (Web Version)")
st.markdown("Hệ thống đánh giá xu hướng và tìm điểm đảo chiều theo chiến lược **Mean Reversion (BB + RSI) kết hợp Bộ lọc ADX**.")

ticker = st.text_input("Nhập mã cổ phiếu (VN):", "HPG").upper()

if st.button("Phân Tích"):
    try:
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        data = yf.download(symbol, period="1y", interval="1d", progress=False)
        
        if data.empty:
            st.error("Không tìm thấy mã này!")
        else:
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            df = calculate_indicators(data)
            rec, reason, color = analyze_strategy(df)
            curr = df.iloc[-1]
            
            # Hiển thị kết quả
            st.divider()
            st.subheader(f"Kết quả phân tích: {ticker}")
            if color == 'green': st.success(f"## {rec}")
            elif color == 'red': st.error(f"## {rec}")
            elif color == 'orange': st.warning(f"## {rec}")
            else: st.info(f"## {rec}")
            st.write(f"**Lý do:** {reason}")

            # Metric
            c1, c2, c3 = st.columns(3)
            c1.metric("Giá", f"{curr['Close']:,.0f}")
            c2.metric("RSI", f"{curr['RSI']:.1f}")
            c3.metric("ADX", f"{curr['ADX']:.1f}")
            
            # Hiển thị biểu đồ
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                               subplot_titles=("Giá & Bollinger Bands", "RSI (14)", "ADX (14) & DI"))
            
            # Giá & BB
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', dash='dash'), name="Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', dash='dash'), name="Lower"), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
            fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
            
            # ADX
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='black'), name="ADX"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], line=dict(color='green'), name="+DI"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], line=dict(color='red'), name="-DI"), row=3, col=1)
            fig.add_hline(y=25, line_dash="dot", row=3, col=1, line_color="gray")
            fig.add_hline(y=50, line_dash="dot", row=3, col=1, line_color="red")
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Lỗi: {e}")

# --- DISCLAIMER ---
st.divider()
st.caption("⚠️ **Tuyên bố miễn trừ trách nhiệm:**")
st.caption("Công cụ này chỉ mang tính chất tham khảo dựa trên các thuật toán phân tích kỹ thuật và dữ liệu quá khứ. Đây không phải là lời khuyên đầu tư tài chính hay khuyến nghị mua bán chính thức. Người sử dụng tự chịu trách nhiệm hoàn toàn về các quyết định giao dịch và rủi ro tài chính của mình. Chúng tôi không chịu trách nhiệm cho bất kỳ khoản lỗ nào phát sinh từ việc sử dụng công cụ này.")
st.caption("Dữ liệu được cung cấp bởi Yahoo Finance.")
