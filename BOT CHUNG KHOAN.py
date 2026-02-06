import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
import random
from datetime import datetime, timedelta

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(layout="wide", page_title="Stock Advisor PRO", page_icon="📈")

# --- CSS TÙY CHỈNH (Giữ nguyên giao diện Neon) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', 'Segoe UI', sans-serif; }
    
    /* HEADER */
    .main-title {
        text-align: center; font-weight: 900;
        background: -webkit-linear-gradient(45deg, #00E676, #69F0AE); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 3.5rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 0px 0px 20px rgba(0, 230, 118, 0.3);
    }
    .sub-title {
        text-align: center; color: #E0E0E0 !important; font-size: 1.2rem;
        font-weight: 400; margin-bottom: 20px; letter-spacing: 0.5px;
    }

    /* DISCLAIMER */
    .disclaimer-box {
        background-color: #1E1E1E; border: 1px solid #444; border-radius: 8px;
        padding: 20px; margin: 0 auto 30px auto; text-align: center; max-width: 800px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .disclaimer-title { color: #FF5252; font-weight: bold; font-size: 1rem; text-transform: uppercase; margin-bottom: 12px; letter-spacing: 1px; }
    .d-line-1 { color: #AAA; font-size: 0.95rem; margin-bottom: 5px; }
    .d-line-2 { color: #E0E0E0; font-size: 1rem; font-weight: bold; margin-bottom: 5px; text-decoration: underline; text-decoration-color: #555; }
    .d-line-3 { color: #888; font-size: 0.85rem; font-style: italic; }

    /* RESULT CARD */
    .result-card {
        padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .bg-green { background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); }
    .bg-red { background: linear-gradient(135deg, #b71c1c 0%, #c62828 100%); }
    .bg-orange { background: linear-gradient(135deg, #e65100 0%, #ef6c00 100%); }
    .bg-blue { background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); }
    .result-title { font-size: 2.2rem; font-weight: 800; color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    .result-reason { font-size: 1.1rem; color: #EEE; margin-top: 10px; font-style: italic; }

    /* REPORT BOX */
    .report-box { background-color: #1E1E1E; border: 1px solid #444; border-radius: 12px; padding: 25px; margin-top: 10px; }
    .report-header { color: #00E676; font-size: 1.2rem; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #444; padding-bottom: 10px; text-transform: uppercase; }
    .report-item { margin-bottom: 12px; font-size: 1rem; color: #FAFAFA; display: flex; align-items: center; }
    .icon-dot { margin-right: 12px; font-size: 1.2rem; }

    /* METRIC CARDS */
    .metric-container {
        background-color: #262730; border: 1px solid #41424C; border-radius: 12px;
        padding: 15px 10px; text-align: center; height: 160px;
        display: flex; flex-direction: column; justify-content: flex-start; align-items: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-label { font-size: 0.9rem; color: #FFF; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; height: 20px; display: flex; align-items: center; }
    .metric-value-box { flex-grow: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; }
    .metric-value { font-size: 2.2rem; font-weight: 900; color: #FFF; line-height: 1; }
    .trend-badge { padding: 10px 30px; border-radius: 30px; font-size: 1.3rem; font-weight: 900; color: white; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    
    div.stButton > button { width: 100%; border-radius: 8px; font-weight: bold; height: 50px; font-size: 1.1rem; }
    
    /* BACKTEST RESULT BOX */
    .backtest-box {
        background: linear-gradient(135deg, #263238 0%, #37474F 100%);
        border-radius: 10px; padding: 20px; margin-top: 20px; text-align: center;
        border: 1px solid #546E7A;
    }
    .backtest-label { color: #CFD8DC; font-size: 1rem; margin-bottom: 5px; }
    .backtest-val { color: #00E676; font-size: 2.5rem; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# --- HÀM TÍNH TOÁN CƠ BẢN ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- HÀM BACKTEST CHO TỪNG ĐƯỜNG MA ---
def run_backtest_for_ma(prices, ma_series, rsi_series, stop_loss_pct):
    cash = 100_000_000
    initial_capital = cash
    shares = 0
    trade_count = 0
    wins = 0
    
    # Chuyển sang numpy array để loop nhanh
    p_arr = prices.values
    ma_arr = ma_series.values
    rsi_arr = rsi_series.values
    
    last_buy_price = 0
    use_sl = stop_loss_pct > 0
    
    # Bắt đầu từ index 205 (đảm bảo đủ dữ liệu cho MA lớn nhất)
    start_idx = 205 
    if len(p_arr) <= start_idx: return -999, 0, 0

    for i in range(start_idx, len(p_arr)):
        price = p_arr[i]
        ma = ma_arr[i]
        rsi = rsi_arr[i]
        
        if np.isnan(ma) or np.isnan(rsi): continue
        
        # LOGIC MUA: Giá < MA và RSI < 30
        if shares == 0:
            if price < ma and rsi < 30:
                shares = cash / price
                cash = 0
                last_buy_price = price
        
        # LOGIC BÁN: Giá > MA và RSI > 70 HOẶC Cắt lỗ
        elif shares > 0:
            is_sell = False
            
            # 1. Cắt lỗ
            if use_sl:
                pct_loss = (price - last_buy_price) / last_buy_price * 100
                if pct_loss <= -stop_loss_pct:
                    is_sell = True
            
            # 2. Chốt lời theo chiến thuật
            if not is_sell:
                if price > ma and rsi > 70:
                    is_sell = True
            
            if is_sell:
                sell_val = shares * price
                if sell_val > shares * last_buy_price:
                    wins += 1
                cash = sell_val
                shares = 0
                trade_count += 1
                
    final_val = cash + (shares * p_arr[-1])
    roi = ((final_val - initial_capital) / initial_capital) * 100
    return roi, trade_count, wins

# --- HÀM TỐI ƯU HÓA (TÌM BEST MA) ---
def optimize_ma_strategy(df, stop_loss_pct):
    prices = df['Close']
    rsi = df['RSI']
    
    results = []
    
    # Quét MA từ 5 đến 205, bước nhảy 10
    ma_ranges = range(5, 206, 10) 
    
    # Progress bar giả lập nếu cần, ở đây chạy backend nhanh
    for ma_period in ma_ranges:
        ma_series = prices.rolling(window=ma_period).mean()
        roi, trades, wins = run_backtest_for_ma(prices, ma_series, rsi, stop_loss_pct)
        
        results.append({
            'MA': ma_period,
            'ROI': roi,
            'Trades': trades,
            'Wins': wins
        })
        
    # Sắp xếp tìm ROI cao nhất
    results_df = pd.DataFrame(results)
    if results_df.empty: return None, None
    
    best_res = results_df.loc[results_df['ROI'].idxmax()]
    return best_res, results_df

# --- HÀM VẼ GIAO DIỆN CHỈ SỐ ---
def render_metric_card(label, value, delta=None, color=None):
    delta_html = ""
    if delta is not None:
        delta_color = "#00E676" if delta > 0 else ("#FF5252" if delta < 0 else "#888")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
        delta_val = f"{abs(delta):.1f}"
        delta_html = f"<div style='font-size:0.9rem; margin-top:5px; color:{delta_color}'>{arrow} {delta_val} vs phiên trước</div>"
    
    if color:
        value_html = f"<div class='trend-badge' style='background-color:{color}'>{value}</div>"
    else:
        value_html = f"<div class='metric-value'>{value}</div>"

    card_html = f"<div class='metric-container'><div class='metric-label'>{label}</div><div class='metric-value-box'>{value_html}{delta_html}</div></div>"
    st.markdown(card_html, unsafe_allow_html=True)

# --- GIAO DIỆN CHÍNH ---
st.markdown("<h1 class='main-title'>STOCK ADVISOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hệ thống Tối ưu hóa Chiến thuật MA & RSI</p>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-box'>
    <div class='disclaimer-title'>⚠️ TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM</div>
    <div class='d-line-1'>Công cụ tự động tối ưu hóa tham số MA quá khứ.</div>
    <div class='d-line-2'>KHÔNG phải lời khuyên đầu tư tài chính chính thức.</div>
    <div class='d-line-3'>Người dùng tự chịu trách nhiệm. Dữ liệu Yahoo Finance.</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1]) 
with col2:
    with st.form(key='search_form'):
        c_ticker, c_sl = st.columns([2, 1])
        with c_ticker:
            ticker_input = st.text_input("Mã cổ phiếu:", value="", placeholder="VD: HPG, VNM...").upper()
        with c_sl:
            stop_loss_input = st.number_input("Cắt lỗ % (0 = Tắt):", min_value=0.0, max_value=20.0, value=7.0, step=0.5)
        submit_button = st.form_submit_button(label='🚀 PHÂN TÍCH & TỐI ƯU HÓA', use_container_width=True)

if submit_button or 'data' in st.session_state:
    # Hack ẩn keyboard mobile
    js_hack = f"""<script>function forceBlur(){{const activeElement=window.parent.document.activeElement;if(activeElement){{activeElement.blur();}}window.parent.document.body.focus();}}forceBlur();setTimeout(forceBlur,200);</script><div style="display:none;">{random.random()}</div>"""
    components.html(js_hack, height=0)

    if submit_button:
        ticker = ticker_input.strip()
        st.session_state['ticker'] = ticker
        st.session_state['sl_pct'] = stop_loss_input
    elif 'ticker' in st.session_state:
        ticker = st.session_state['ticker']
        stop_loss_input = st.session_state.get('sl_pct', 7.0)

    if not ticker:
        st.warning("⚠️ Vui lòng nhập mã cổ phiếu!")
    else:
        symbol = ticker if ".VN" in ticker else f"{ticker}.VN"
        
        if 'data' not in st.session_state or st.session_state.get('current_symbol') != symbol:
            with st.spinner(f'Đang tải dữ liệu {ticker} và chạy tối ưu hóa...'):
                try:
                    # Lấy max lịch sử để tối ưu tốt nhất
                    df_full = yf.download(symbol, period="max", interval="1d", progress=False)
                    if df_full.empty:
                        st.error(f"❌ Không tìm thấy mã **{ticker}**!")
                        st.stop()
                    
                    if isinstance(df_full.columns, pd.MultiIndex): df_full.columns = df_full.columns.get_level_values(0)
                    
                    # Tính RSI trước
                    df_full['RSI'] = calculate_rsi(df_full['Close'], 14)
                    
                    # Chạy tối ưu hóa
                    best_res, results_df = optimize_ma_strategy(df_full, stop_loss_input)
                    
                    if best_res is None:
                        st.error("Không đủ dữ liệu để tính toán.")
                        st.stop()
                        
                    # Lưu thông tin Best MA vào session
                    st.session_state['data'] = df_full
                    st.session_state['best_ma'] = int(best_res['MA'])
                    st.session_state['best_roi'] = best_res['ROI']
                    st.session_state['top_mas'] = results_df.sort_values(by='ROI', ascending=False).head(5)
                    st.session_state['current_symbol'] = symbol
                    
                    # Intraday
                    df_intra = yf.download(symbol, period="1d", interval="5m", progress=False)
                    if isinstance(df_intra.columns, pd.MultiIndex): df_intra.columns = df_intra.columns.get_level_values(0)
                    if not df_intra.empty:
                        if df_intra.index.tzinfo is None:
                            df_intra.index = df_intra.index + timedelta(hours=7)
                        else:
                            df_intra.index = df_intra.index.tz_convert('Asia/Ho_Chi_Minh')
                    st.session_state['data_intra'] = df_intra

                except Exception as e:
                    st.error(f"Lỗi tải dữ liệu: {e}")
                    st.stop()

        try:
            df = st.session_state['data']
            df_intra = st.session_state['data_intra']
            best_ma_val = st.session_state['best_ma']
            best_roi_val = st.session_state['best_roi']
            top_mas_df = st.session_state['top_mas']
            
            # Tính đường Best MA cho toàn bộ dataframe để vẽ và check tín hiệu hiện tại
            df['BestSMA'] = df['Close'].rolling(window=best_ma_val).mean()
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            curr_price = curr['Close']
            curr_ma = curr['BestSMA']
            curr_rsi = curr['RSI']
            
            # --- LOGIC KHUYẾN NGHỊ (Dựa trên Best MA) ---
            rec = "QUAN SÁT (WAIT)"
            reason = "Chưa có tín hiệu."
            bg_class = "bg-blue"
            
            if curr_price < curr_ma and curr_rsi < 30:
                rec = "MUA NGAY"
                reason = f"Giá ({curr_price:,.0f}) < MA{best_ma_val} ({curr_ma:,.0f}) và RSI ({curr_rsi:.1f}) < 30."
                bg_class = "bg-green"
            elif curr_price > curr_ma and curr_rsi > 70:
                rec = "BÁN NGAY"
                reason = f"Giá ({curr_price:,.0f}) > MA{best_ma_val} ({curr_ma:,.0f}) và RSI ({curr_rsi:.1f}) > 70."
                bg_class = "bg-red"
            else:
                if curr_price > curr_ma:
                    reason = f"Giá đang nằm trên đường MA{best_ma_val} (Xu hướng tăng), chờ RSI > 70 để chốt lời."
                else:
                    reason = f"Giá đang nằm dưới đường MA{best_ma_val} (Xu hướng giảm), chờ RSI < 30 để mua."

            # HIỂN THỊ THẺ KẾT QUẢ
            st.markdown(f"<div class='result-card {bg_class}'><div class='result-title'>{rec}</div><div class='result-reason'>💡 {reason}</div></div>", unsafe_allow_html=True)
            
            # HIỂN THỊ BACKTEST BOX
            bk_color = "#00E676" if best_roi_val > 0 else "#FF5252"
            sl_text = f"Cắt lỗ {stop_loss_input}%" if stop_loss_input > 0 else "KHÔNG SL"
            
            st.markdown(f"""
            <div class='backtest-box'>
                <div style='display:flex; justify-content:space-around; align-items:center;'>
                    <div style='text-align:center;'>
                        <div class='backtest-label'>ĐƯỜNG MA TỐI ƯU</div>
                        <div class='backtest-val' style='color:#FFF; font-size: 2rem;'>MA {best_ma_val}</div>
                    </div>
                    <div style='border-left:1px solid #546E7A; height:50px;'></div>
                    <div style='text-align:center;'>
                        <div class='backtest-label'>LỢI NHUẬN BACKTEST</div>
                        <div class='backtest-val' style='font-size:2rem; color:{bk_color}'>{best_roi_val:+.1f}%</div>
                        <div style='font-size:0.8rem; color:#AAA;'>({sl_text})</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # REPORT BOX
            report = f"""
            <div class='report-box'>
                <div class='report-header'>📝 PHÂN TÍCH TỐI ƯU HÓA</div>
                <div class='report-item'><span class='icon-dot'>🏆</span> <span>Hệ thống đã quét các đường MA từ 5 đến 205.</span></div>
                <div class='report-item'><span class='icon-dot'>✅</span> <span>Đường <b>MA{best_ma_val}</b> cho hiệu quả cao nhất lịch sử với mã {ticker}.</span></div>
                <div class='report-item'><span class='icon-dot'>📊</span> <span>Chiến thuật sử dụng: Mua khi Giá < MA{best_ma_val} & RSI < 30. Bán khi Giá > MA{best_ma_val} & RSI > 70.</span></div>
            </div>
            """
            st.markdown(report, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # INTRADAY CHART
            if not df_intra.empty:
                st.divider()
                latest_date = df_intra.index[0].strftime('%d/%m/%Y')
                st.markdown(f"### ⏱️ Diễn biến trong ngày ({latest_date})")
                ref_price = df['Close'].iloc[-2]
                current_price = df_intra['Close'].iloc[-1]
                line_color = '#00E676' if current_price >= ref_price else '#FF5252'
                fig_intra = go.Figure()
                fig_intra.add_trace(go.Scatter(x=df_intra.index, y=df_intra['Close'], mode='lines', line=dict(color=line_color, width=2), name='Intraday'))
                fig_intra.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', tickformat="%H:%M"), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig_intra, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            # METRICS
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: render_metric_card("GIÁ ĐÓNG CỬA", f"{curr['Close']:,.0f}", curr['Close'] - prev['Close'])
            with col_m2: render_metric_card("RSI (14)", f"{curr['RSI']:.1f}", curr['RSI'] - prev['RSI'])
            with col_m3: render_metric_card("MA TỐI ƯU", f"MA {best_ma_val}", curr['Close'] - curr['BestSMA'])
            with col_m4:
                status = "UPTREND" if curr_price > curr_ma else "DOWNTREND"
                color_st = "#00E676" if status == "UPTREND" else "#FF5252"
                render_metric_card("XU HƯỚNG", status, None, color=color_st)

            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            
            # --- BIỂU ĐỒ KỸ THUẬT ---
            st.markdown(f"### 📊 Biểu đồ Kỹ Thuật & Top MA")
            time_tabs = st.radio("Khung thời gian:", ["3 Tháng", "6 Tháng", "1 Năm", "3 Năm", "Tất cả"], horizontal=True, index=2)
            
            df_chart = df.copy()
            if time_tabs == "3 Tháng": df_chart = df.iloc[-66:]
            elif time_tabs == "6 Tháng": df_chart = df.iloc[-132:]
            elif time_tabs == "1 Năm": df_chart = df.iloc[-252:]
            elif time_tabs == "3 Năm": df_chart = df.iloc[-756:]

            # Chart 1: Giá + Best MA
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BestSMA'], line=dict(color='#FF914D', width=2), name=f"MA {best_ma_val} (Tối ưu)"))
            fig1.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Giá"))
            fig1.update_layout(height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
            st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

            # 2 Cột: RSI và Bảng Top MA
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("### 🚀 Chỉ số RSI")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='#E040FB', width=2), name="RSI"))
                fig2.add_hline(y=70, line_dash="dot", line_color="#FF5252")
                fig2.add_hline(y=30, line_dash="dot", line_color="#00E676")
                fig2.update_layout(height=350, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#FAFAFA'), margin=dict(l=10, r=10, t=10, b=40), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333'), yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#333', autorange=True))
                st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

            with col_c2:
                st.markdown("### 🏆 Top 5 Đường MA Hiệu Quả")
                st.dataframe(
                    top_mas_df[['MA', 'ROI', 'Trades', 'Wins']].reset_index(drop=True),
                    column_config={
                        "MA": "Đường MA",
                        "ROI": st.column_config.NumberColumn("Lợi Nhuận (%)", format="%.2f %%"),
                        "Trades": "Số Lệnh",
                        "Wins": "Số Thắng"
                    },
                    use_container_width=True,
                    height=350
                )

        except Exception as e:
            st.error(f"Đã xảy ra lỗi hiển thị: {e}")
