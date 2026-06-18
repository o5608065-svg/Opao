import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 

st.set_page_config(page_title="残差互信息网络", layout="wide", initial_sidebar_state="expanded")

def calc_mutual_information(x, y, bins):
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    p_xy = c_xy / float(np.sum(c_xy))
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = p_x[:, None] * p_y[None, :]
    nzs = p_xy > 0
    return np.sum(p_xy[nzs] * np.log(p_xy[nzs] / p_x_p_y[nzs]))

@st.cache_data 
def analyze_network(data):
    returns_df = np.log(data / data.shift(1)).dropna()
    tickers = returns_df.columns.tolist()
    N_samples = len(returns_df)
    N_stocks = len(tickers)
    
    bins = max(3, int(np.floor(N_samples ** (1/3))))
    rho_matrix = returns_df.corr(method='pearson').values
    delta_I_matrix = np.zeros((N_stocks, N_stocks))
    
    progress_bar = st.progress(0)
    total_pairs = (N_stocks * (N_stocks - 1)) / 2
    current_pair = 0
    
    for i in range(N_stocks):
        for j in range(i + 1, N_stocks):
            x = returns_df.iloc[:, i].values
            y = returns_df.iloc[:, j].values
            
            I_emp = calc_mutual_information(x, y, bins)
            rho = np.clip(rho_matrix[i, j], -0.999, 0.999) 
            I_g = -0.5 * np.log(1 - rho**2)
            
            delta_I = max(0, I_emp - I_g)
            delta_I_matrix[i, j] = delta_I
            delta_I_matrix[j, i] = delta_I
            
            current_pair += 1
            progress_bar.progress(int(current_pair / total_pairs * 100))
            
    progress_bar.empty() 
    delta_I_df = pd.DataFrame(delta_I_matrix, index=tickers, columns=tickers)
    mean_delta_I = delta_I_df.sum(axis=1) / (N_stocks - 1)
    
    return delta_I_df, mean_delta_I

def plot_mst(delta_I_df):
    epsilon = 1e-5
    dist_matrix = 1 / (delta_I_df + epsilon)
    tickers = delta_I_df.columns
    
    G = nx.Graph()
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            stock_a = tickers[i]
            stock_b = tickers[j]
            weight = dist_matrix.loc[stock_a, stock_b]
            delta_i_val = delta_I_df.loc[stock_a, stock_b]
            if delta_i_val > 0:
                G.add_edge(stock_a, stock_b, weight=weight)
                
    MST = nx.minimum_spanning_tree(G)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(MST, k=0.5, seed=42)
    nx.draw_networkx_nodes(MST, pos, ax=ax, node_size=800, node_color='#1f77b4', alpha=0.9, edgecolors='white')
    nx.draw_networkx_labels(MST, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(MST, pos, ax=ax, width=2, alpha=0.6, edge_color='#ff7f0e')
    
    ax.axis('off')
    return fig

# ================= 界面设计 =================
st.title("📊 隐藏共振网络分析系统")
st.markdown("一键自动获取历史行情，计算剥离大盘贝塔后的核心共振标的。")

st.sidebar.header("⚙️ 参数设置")
# 提供一组消费电子、算力、通信基础设施相关的默认标的
default_tickers = "AAPL, QCOM, ARM, VZ, TMUS, ERIC, NOK, SMCI, NVDA"
tickers_input = st.sidebar.text_area("输入标的代码 (用英文逗号分隔)", default_tickers)

end_date = datetime.today()
start_date = end_date - timedelta(days=365)
start_input = st.sidebar.date_input("开始日期", start_date)
end_input = st.sidebar.date_input("结束日期", end_date)

if st.sidebar.button("🚀 获取数据并计算", type="primary", use_container_width=True):
    ticker_list = [t.strip() for t in tickers_input.split(',')]
    
    with st.spinner('正在从雅虎财经拉取数据...'):
        data = yf.download(ticker_list, start=start_input, end=end_input)['Adj Close']
        data = data.dropna(axis=1, how='all')
    
    if data.empty or data.shape[1] < 2:
        st.error("数据获取失败或有效标的不足两个，请检查代码拼写或网络状态。")
    else:
        st.sidebar.success(f"成功获取 {data.shape[1]} 只标的的数据！")
        with st.spinner('正在执行矩阵张量运算，请稍候...'):
            delta_i_matrix, factor_scores = analyze_network(data)
            
            col1, col2 = st.columns([1, 2.5])
            
            with col1:
                st.subheader("🏆 核心共动得分排行")
                factor_df = factor_scores.sort_values(ascending=False).reset_index()
                factor_df.columns = ['标的代码', '隐藏共动得分']
                st.dataframe(factor_df, use_container_width=True, height=600)
                
            with col2:
                st.subheader("🕸️ 最小生成树 (MST) 拓扑结构")
                fig = plot_mst(delta_i_matrix)
                st.pyplot(fig)
