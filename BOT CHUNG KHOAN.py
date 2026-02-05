import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO", page_icon="📈")

# --- CSS TÙY CHỈNH (LÀM ĐẸP GIAO DIỆN) ---
st.markdown("""
<style>
    /* Chỉnh Font chữ toàn bộ web sang Sans-serif cho đẹp, bỏ font code cũ */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Căn giữa tiêu đề */
    .main-title {
        text-align: center;
        font-weight: bold;
        color: #FF4B4B;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    
    /* Style cho khung báo cáo chi tiết để không bị xấu */
    .report-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #FF4B4B;
        margin-top: 20px;
        color: #31333F; /* Màu chữ tối cho dễ đọc trên nền sáng */
    }
    
    /* Dark mode support cho report box */
    @media (prefers-color-scheme: dark) {
        .report-box {
            background-color: #262730;
            color: #FAFAFA;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN ---
def calculate_indicators(df):
    # 1. BB
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['SMA20'] + (2 * df['StdDev'])
    df['Lower'] = df['SMA20'] - (2 * df['StdDev'])
    
    # 2. RSI (Wilder's Smoothing)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
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
    if len(df) < 25: return "Không đủ dữ liệu", "NEUTRAL", "gray", "Chưa đủ dữ liệu."
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    
    # Values
    price = curr['Close']
    rsi = curr['RSI']
    adx = curr['ADX']
    di_plus = curr['+DI']
    di_minus = curr['-DI']
    lower_band = curr['Lower']
    upper_band = curr['Upper']

    # Triggers
    buy_trigger = (price <= lower_band * 1.01) and (rsi < 30)
    sell_trigger = (price >= upper_band * 0.99) and (rsi > 70)
    
    rec, reason, color = "QUAN SÁT (HOLD)", "Chưa có tín hiệu giao dịch đặc biệt.", "blue"
    
    # --- LOGIC ---
    if buy_trigger:
        if adx < 25:
            if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color = "MUA NGAY", "Giá chạm đáy BB, RSI thấp. Xu hướng giảm yếu và đang suy thoái.", "green"
            else:
                rec, reason, color = "CHỜ MUA", "Giá rẻ nhưng lực bán vẫn còn. Chờ DI- giảm.", "orange"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_minus < prev['-DI'] < prev2['-DI'])
            if cooling:
                rec, reason, color = "MUA NGAY", "Bắt đáy sau sập mạnh (ADX & DI- giảm 2 phiên).", "green"
            else:
                rec, reason, color = "ĐỨNG NGOÀI", f"Đang sập mạnh (ADX={adx:.1f}). Đừng bắt dao rơi!", "red"
        else:
             if (di_minus > di_plus) and (di_minus < prev['-DI']):
                rec, reason, color = "MUA THĂM DÒ", "Giá rẻ, xu hướng giảm trung bình.", "green"

    elif sell_trigger:
        if adx < 25:
             if (di_plus > di_minus) and (di_plus < prev['+DI']):
                rec, reason, color = "BÁN NGAY", "Giá đỉnh BB, RSI cao. Lực tăng yếu.", "red"
        elif adx > 50:
            cooling = (adx < prev['ADX'] < prev2['ADX']) and (di_plus < prev['+DI'] < prev2['+DI'])
            if cooling:
                rec, reason, color = "BÁN CHỐT LỜI", "Siêu sóng kết thúc (ADX & DI+ giảm 2 phiên).", "red"
            else:
                rec, reason, color = "NẮM GIỮ", f"Trend tăng cực mạnh (ADX={adx:.1f}). Gồng lãi!", "green"
        else:
             rec, reason, color = "CÂN NHẮC BÁN", "Vùng quá mua, cân nhắc chốt lời.", "orange"

    # --- REPORT TEXT (Đã sửa format Markdown) ---
    trend_state = "TĂNG" if di_plus > di_minus else "GIẢM"
    trend_strength = "YẾU (Sideway)" if adx < 25 else ("CỰC MẠNH" if adx > 50 else "TRUNG BÌNH")
    
    price_pos = "trong biên độ an toàn"
    if price <= lower_band * 1.01: price_pos = "chạm dải dưới (Rẻ)"
    elif price >= upper_band * 0.99: price_pos = "chạm dải trên (Đắt)"
    
    rsi_state = "Trung tính"
    if rsi < 30: rsi_state = "QUÁ BÁN (Cơ hội mua)"
    elif rsi > 70: rsi_state = "QUÁ MUA (Rủi ro chỉnh)"

    # Sử dụng HTML/Markdown chuẩn để không bị lỗi font
    report = f"""
    <div class='report-box'>
        <h4>📝 Phân Tích Chi Tiết</h4>
        <ul>
            <li><b>Xu hướng:</b> Thị trường đang trong pha <b>{trend_state}</b> với cường độ <b>{trend_strength}</b> (ADX={adx:.1f}).</li>
            <li><b>Vị thế giá:</b> Giá hiện tại đang <b>{price_pos}</b> của Bollinger Bands.</li>
            <li><b>Động lượng (RSI):</b> Chỉ số RSI đạt {rsi:.1f}, trạng thái <b>{rsi_state}</b>.</li>
            <li><b>Tín hiệu ADX/DI:</b> { "Phe Mua đang kiểm soát (+DI > -DI)" if di_plus > di_minus else "Phe Bán đang kiểm soát (-DI > +DI)" }.</li>
        </ul>
    </div>
    """
             
    return rec, reason, color, report

# --- GIAO DIỆN CHÍNH ---

st.markdown("<h1 class='main-title'>📈 STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Hệ thống phân tích kỹ thuật tự động: BB + RSI + ADX + DI</p>", unsafe_allow_html=True)

# 1. CĂN GIỮA THANH TÌM KIẾM VÀ XỬ LÝ ENTER
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.form(key='search_form'):
        # Tạo 2 cột bên trong form để nút nằm cạnh ô nhập (nếu muốn) hoặc dưới
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            ticker_input = st.text_input("Nhập mã cổ phiếu (VN):", "HPG", placeholder="Ví dụ: VNM, SSI...")
        with col_btn:
            # Padding để nút bấm thẳng hàng với ô input
            st.write("") 
            st.write("")
            submit_button = st.form_submit_button(label='🔍 Phân Tích')

# Nút đổi giao diện (Mẹo)
st.sidebar.markdown("### ⚙️ Cài đặt")
st.sidebar.info("Để chuyển chế độ Sáng/Tối, vui lòng chọn **Settings** ở góc trên cùng bên phải màn hình (Dấu 3 chấm ⋮).")

# LOGIC KHI ẤN ENTER HOẶC NÚT BẤM
if submit_button:
    try:
        ticker = ticker_input.upper().strip()
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        with st.spinner(f'Đang phân tích mã {ticker}...'):
            data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            if data.empty:
                st.error(f"❌ Không tìm thấy dữ liệu cho mã **{ticker}**! Vui lòng kiểm tra lại.")
            else:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                df = calculate_indicators(data)
                rec, reason, color, report = analyze_strategy(df)
                curr = df.iloc[-1]
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.divider()
                
                # Header Kết quả
                st.markdown(f"### 📊 Kết quả phân tích: {ticker}")
                
                # Alert Box màu sắc
                if color == 'green': st.success(f"## {rec}")
                elif color == 'red': st.error(f"## {rec}")
                elif color == 'orange': st.warning(f"## {rec}")
                else: st.info(f"## {rec}")
                
                st.write(f"**Lý do:** {reason}")
                
                # Báo cáo chi tiết (HTML Render)
                st.markdown(report, unsafe_allow_html=True)

                # Metrics (Chỉ số)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Giá đóng cửa", f"{curr['Close']:,.0f}", f"{curr['Close'] - df.iloc[-2]['Close']:,.0f}")
                m2.metric("RSI (14)", f"{curr['RSI']:.1f}")
                m3.metric("ADX (14)", f"{curr['ADX']:.1f}")
                m4.metric("Xu hướng", "TĂNG" if curr['+DI'] > curr['-DI'] else "GIẢM")
                
                # --- BIỂU ĐỒ ---
                st.divider()
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03,
                                   subplot_titles=("Giá & Bollinger Bands", "RSI (14)", "ADX & DI"))
                
                # Chart 1
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Giá"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='gray', width=1, dash='dash'), name="Upper"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='gray', width=1, dash='dash'), name="Lower"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='orange', width=1), name="SMA20"), row=1, col=1)

                # Chart 2
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#9467bd', width=2), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red")
                fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green")
                
                # Chart 3
                fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='black', width=2), name="ADX"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['+DI'], line=dict(color='#2ca02c', width=1), name="+DI"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['-DI'], line=dict(color='#d62728', width=1), name="-DI"), row=3, col=1)
                fig.add_hline(y=25, line_dash="dot", row=3, col=1, line_color="gray")
                fig.add_hline(y=50, line_dash="dot", row=3, col=1, line_color="red")
                
                fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi hệ thống: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 0.8em;'>⚠️ Công cụ hỗ trợ phân tích kỹ thuật. Không phải lời khuyên đầu tư tài chính.</p>", unsafe_allow_html=True)
