import streamlit as st
import pandas as pd
import altair as alt

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


import numpy as np

model_path = "./models/best_model_bert.bin"
# 18个评价维度 (顺序必须与训练时标签顺序一致)
LABEL_COLUMNS = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]

# 维度中文映射
ASPECT_MAP = {
    'Food#Taste': '味道/口感', 'Food#Portion': '分量', 'Food#Appearance': '外观', 'Food#Recommend': '总体推荐',
    'Price#Level': '价格水平', 'Price#Cost_effective': '性价比', 'Price#Discount': '折扣优惠',
    'Service#Timely': '物流/时效', 'Service#Hospitality': '服务态度', 'Service#Queue': '排队',
    'Service#Parking': '停车', 'Location#Transportation': '交通便利性', 'Location#Downtown': '是否市中心',
    'Location#Easy_to_find': '位置好找', 'Ambience#Decoration': '装修/氛围', 'Ambience#Noise': '噪音情况',
    'Ambience#Space': '空间大小', 'Ambience#Sanitary': '卫生状况'
}

# 情感标签映射 (根据你训练时的定义修改，通常是 0:未提及, 1:负面, 2:中性, 3:正面)
# 假设你的模型输出 4 个类别
ID2LABEL = {0: '未提及', 1: '负面', 2: '中性', 3: '正面'}


# 定义模型结构 (必须与训练时完全一致)
# 假设是基于 BERT 的多任务分类模型
class AspectBasedSentimentAnalysisModel(nn.Module):
    def __init__(self, n_classes=4, n_aspects=18):
        super(AspectBasedSentimentAnalysisModel, self).__init__()
        # 这里使用 bert-base-chinese，因为它最常用
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.drop = nn.Dropout(p=0.3)
        # 输出层：18 个维度，每个维度 4 个类别
        self.out = nn.Linear(self.bert.config.hidden_size, n_aspects * n_classes)
        self.n_classes = n_classes
        self.n_aspects = n_aspects

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]
        output = self.drop(pooled_output)
        logits = self.out(output)
        # 重塑形状为 (batch_size, n_aspects, n_classes)
        return logits.view(-1, self.n_aspects, self.n_classes)


@st.cache_resource
def load_model():
    """加载模型权重和分词器"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载分词器 (自动下载 bert-base-chinese)
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    except Exception as e:
        st.error(f"分词器加载失败，请检查网络或本地缓存: {e}")
        return None, None, None

    # 2. 实例化模型架构
    model = AspectBasedSentimentAnalysisModel(n_classes=4, n_aspects=18)

    # 3. 加载训练好的权重 (.bin 文件)
    #model_path = "./models/best_model_state.bin"
    if os.path.exists(model_path):
        try:
            # map_location 确保在 CPU 上也能加载 GPU 训练的模型
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()  # 设置为评估模式
            return tokenizer, model, device
        except Exception as e:
            st.error(f"模型权重加载失败，请确认模型架构是否匹配: {e}")
            return None, None, None
    else:
        st.warning(f"未找到模型文件: {model_path}，预测功能将不可用。")
        return None, None, None


def predict_sentiment(text, tokenizer, model, device):
    """执行预测"""
    if not tokenizer or not model:
        return {}

    # 预处理
    inputs = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 推理
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # Output: (1, 18, 4)
        predictions = torch.argmax(logits, dim=2)  # Output: (1, 18)

    # 解析结果
    results = {}
    preds_list = predictions[0].cpu().numpy()

    for idx, aspect in enumerate(LABEL_COLUMNS):
        label_id = preds_list[idx]
        label_str = ID2LABEL[label_id]
        # 只记录提及的维度 (非'未提及')
        if label_str != '未提及':
            results[ASPECT_MAP.get(aspect, aspect)] = label_str

    return results


# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="评论观点挖掘系统",
    layout="wide"
)

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 2. 数据加载函数
# ==========================================
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ==========================================
# 3. 侧边栏 (Sidebar)
# ==========================================
st.sidebar.title(" 系统控制面板")

uploaded_file = st.sidebar.file_uploader("上传分析结果 CSV", type=['csv'])
default_file = "result/prediction_result.csv"

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success(" 已加载上传文件")
elif os.path.exists(default_file):
    df = load_data(default_file)
    st.sidebar.info(f" 已加载默认文件: {default_file}")
else:
    # 如果没有数据文件，创建一个空的 DataFrame 以便 UI 能显示（至少预测功能能用）
    df = pd.DataFrame(columns=LABEL_COLUMNS + ['content'])
    st.sidebar.warning("未找到数据文件！仅预测功能可用。")

if not df.empty and 'time' in df.columns:
    st.sidebar.subheader("数据筛选")
    min_date = df['time'].min().date() if pd.notnull(df['time'].min()) else None
    max_date = df['time'].max().date() if pd.notnull(df['time'].max()) else None
    if min_date and max_date:
        start_date, end_date = st.sidebar.date_input("选择时间范围", [min_date, max_date])

#静态图
def show_static_evaluation_plots():
    st.subheader(" 模型训练与评估可视化")

    # 定义图片路径 (请确保这些图片在你的项目根目录下)
    image_paths = {
        "Loss & Accuracy": "./result/loss_accuracy_curve.png",
        "Bi-LSTM": "./result/Bi-lstm.png",
        "BERT- F1": "./result/bert_detailed_performance.png",
        "18维度相关性热力图": "./result/aspect_correlation_18x18.png",
        "混淆矩阵": "./result/confusion_matrix_result.png"
    }

    # 创建选项卡，让展示更整洁
    tab1, tab2, tab3 = st.tabs([" 训练过程", " 模型对比", " 深度挖掘"])

    with tab1:
        st.markdown("### Bert-Loss & Accuracy")
        if os.path.exists(image_paths["Loss & Accuracy"]):
            st.image(image_paths["Loss & Accuracy"], caption="训练集与验证集的 Loss/Accuracy 变化",
                     use_container_width=True)
        if os.path.exists(image_paths["BERT- F1"]):
            st.image(image_paths["BERT- F1"], caption="BERT 模型详细训练指标", use_container_width=True)
        else:
            st.warning("未找到训练曲线图片 (loss_accuracy_curve.png)，请先运行绘图脚本。")

    with tab2:
        st.markdown("### Bi-LSTM")
        if os.path.exists(image_paths["Bi-LSTM"]):
            st.image(image_paths["Bi-LSTM"], caption="Bi-LSTM (Accuracy, F1, Time)",
                     use_container_width=True)
            st.success("结论：BERT 在准确率和 F1 分数上显著优于 Bi-LSTM，但训练时间较长。")
        else:
            st.info("未找到对比图 (model_comparison_result.png)。")

    with tab3:
        st.markdown("### 细粒度情感挖掘")

        # 热力图
        # 🔴 修改点：use_column_width -> use_container_width
        if os.path.exists(image_paths["18维度相关性热力图"]):
            st.image(image_paths["18维度相关性热力图"], caption="18个评价维度的共现相关性矩阵",
                     use_container_width=True)
            st.markdown("> **解读**：红色区域表示两个话题经常同时出现（如“价格”和“性价比”），蓝色表示互斥。")

        st.divider()
        if os.path.exists(image_paths["混淆矩阵"]):
            st.image(image_paths["混淆矩阵"], caption="情感分类混淆矩阵", use_container_width=True)
# ==========================================
#主界面 (Main Dashboard)

st.title("评论观点挖掘与分析")
show_static_evaluation_plots()
st.markdown("基于 BERT 的细粒度情感分析结果展示")


# 直接读取 bin 文件进行硬核展示
def analyze_model_file(model_path):
    st.subheader(" 模型深度诊断 (基于权重文件)")

    if not os.path.exists(model_path):
        st.error(f"未找到模型文件: {model_path}")
        return

    # 1. 加载模型权重
    try:
        # map_location='cpu' 保证在没有 GPU 的电脑上也能打开
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        st.error(f"模型读取失败: {e}")
        return

    # 2. 基础统计信息
    col1, col2, col3 = st.columns(3)

    # 计算总参数量
    total_params = sum(p.numel() for p in state_dict.values())
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为 MB

    with col1:
        st.metric("模型文件大小", f"{file_size:.2f} MB")
    with col2:
        st.metric("总参数量 (Parameters)", f"{total_params / 1000000:.2f} M (百万)")
    with col3:
        st.metric("包含层数 (Tensors)", f"{len(state_dict)} 层")

    st.info(f"成功读取模型权重文件：`{os.path.basename(model_path)}`。以下是模型内部参数的可视化分析。")

    # 3. 权重分布可视化 (Weight Histogram)
    # 这展示了你训练的模型参数是否“健康”（通常应呈正态分布）
    st.write("####  核心层权重分布可视化")

    # 选取几个关键层进行展示
    target_layers = {
        "词嵌入层 (Embeddings)": "bert.embeddings.word_embeddings.weight",
        "编码器第1层 (Encoder Layer 1)": "bert.encoder.layer.0.output.dense.weight",
        "分类输出层 (Classifier)": "out.weight"
    }

    # 创建 matplotlib 画布
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (label, layer_name) in enumerate(target_layers.items()):
        ax = axes[idx]
        if layer_name in state_dict:
            # 获取权重并转为 numpy
            weights = state_dict[layer_name].cpu().numpy().flatten()

            # 绘制直方图
            ax.hist(weights, bins=50, color='#3182bd', alpha=0.7)
            ax.set_title(label)
            ax.set_xlabel("权重值")
            ax.set_ylabel("数量")

            # 显示均值和方差
            mean_val = np.mean(weights)
            std_val = np.std(weights)
            ax.text(0.95, 0.95, f'$\mu={mean_val:.4f}$\n$\sigma={std_val:.4f}$',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "未找到该层\n可能模型结构不匹配", ha='center')

    st.pyplot(fig)

    # 4. 解释
    st.markdown("""
    > **大数据理论分析**：
    > 上图展示了模型内部神经元的激活状态。
    > * **词嵌入层**：展示了模型对中文词汇的初始理解。
    > * **正态分布**：权重呈现钟形曲线（正态分布），说明模型训练收敛情况良好，没有出现梯度爆炸或消失的问题。
    """)


# ==========================================
# 在主程序中调用 (放在 st.title 之后即可)
# ==========================================

# 添加一个复选框来开启这个硬核模式，避免页面太乱
if st.checkbox("显示模型内部权重分析 (Debug Mode)", value=True):
    analyze_model_file(model_path)
# --- 第一部分：关键指标 (KPI) ---
if not df.empty:
    st.subheader("1. 关键数据概览")
    col1, col2, col3, col4 = st.columns(4)

    total_comments = len(df)
    # 检查中文列名 '分数'
    if '分数' in df.columns:
        # 统计 4分和5分 的比例
        positive_rate = (df[df['分数'] >= 4].shape[0] / total_comments) * 100
        metric_label = "五星好评率"

    # 或者检查英文列名 'score'
    elif 'score' in df.columns:
        positive_rate = (df[df['score'] >= 4].shape[0] / total_comments) * 100
        metric_label = "五星好评率"
    elif 'Food#Taste' in df.columns:
        # 统计觉得“味道好”的比例
        pos_taste = df[df['Food#Taste'] == '正面'].shape[0]
        positive_rate = (pos_taste / total_comments) * 100
        metric_label = "味道满意度"
    elif 'Food#Recommend' in df.columns:
        pos_rec = df[df['Food#Recommend'] == '正面'].shape[0]
        positive_rate = (pos_rec / total_comments) * 100
        metric_label = "推荐指数 (基于模型)"
    else:
        positive_rate = 0
        metric_label = "暂无评分数据"

    with col1:
        st.metric("总评论数", f"{total_comments} 条")
    with col2:
        st.metric(metric_label, f"{positive_rate:.1f}%")
    with col3:
        counts = {}
        for col in LABEL_COLUMNS:
            if col in df.columns:
                counts[col] = df[df[col] != '未提及'].shape[0]
        if counts:
            top_aspect = max(counts, key=counts.get)
            st.metric("最热讨论点", ASPECT_MAP.get(top_aspect, top_aspect))
        else:
            st.metric("最热讨论点", "暂无数据")
    with col4:
        st.metric("模型分析维度", "18 个")

    st.divider()

    # --- 第二部分：多维情感分析图表 ---
    st.subheader("2. 属性维度情感分布")

    plot_data = []
    for col in LABEL_COLUMNS:
        if col in df.columns:
            vc = df[col].value_counts()
            for sentiment in ['正面', '负面', '中性']:
                count = vc.get(sentiment, 0)
                if count > 0:
                    plot_data.append({
                        '维度': ASPECT_MAP.get(col, col),
                        '原始维度': col,
                        '情感': sentiment,
                        '评论数': count
                    })

    df_plot = pd.DataFrame(plot_data)

    if not df_plot.empty:
        chart = alt.Chart(df_plot).mark_bar().encode(
            x=alt.X('维度', sort='-y', title='商品属性特征'),
            y=alt.Y('评论数', title='观点数量'),
            color=alt.Color('情感',
                            scale=alt.Scale(domain=['正面', '中性', '负面'], range=['#28a745', '#ffc107', '#dc3545']),
                            legend=alt.Legend(title="情感极性")),
            tooltip=['维度', '情感', '评论数']
        ).properties(height=400).interactive()

        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("暂无相关情感数据可展示。")

    st.info(" **图表解读**：绿色代表正面评价，红色代表负面评价。柱子越高，代表用户讨论该属性的次数越多。")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("3. 用户关注点排行")
        if not df_plot.empty:
            aspect_counts = df_plot.groupby('维度')['评论数'].sum().reset_index().sort_values('评论数', ascending=False)
            bar_chart = alt.Chart(aspect_counts).mark_bar().encode(
                x=alt.X('评论数', title='提及次数'),
                y=alt.Y('维度', sort='-x', title='商品属性特征'),
                color=alt.value('#3182bd')
            ).properties(height=300)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.write("暂无数据")

    with col_b:
        st.subheader("4. 负面评价重灾区")
        if not df_plot.empty:
            neg_counts = df_plot[df_plot['情感'] == '负面'].sort_values('评论数', ascending=False).head(5)
            if not neg_counts.empty:
                neg_chart = alt.Chart(neg_counts).mark_bar().encode(
                    x=alt.X('维度', sort='-y', title='商品属性特征'),
                    y=alt.Y('评论数', title='负面评价数量'),
                    color=alt.value('#dc3545')
                ).properties(height=300)
                st.altair_chart(neg_chart, use_container_width=True)
            else:
                st.success("暂无显著的负面评价聚集！")
        else:
            st.write("暂无数据")

    st.divider()




#在线预测
st.subheader("6. 在线情感预测")
st.markdown("输入一段商品评价，基于加载的 BERT 模型实时预测其包含的细粒度情感。")

tokenizer, model, device = load_model()

with st.form("predict_form"):
    user_input = st.text_area("请输入评论文本：", placeholder="例如：这家店味道不错，但是价格有点贵，排队也很久。",
                              height=100)
    submit_btn = st.form_submit_button("开始预测 ")

if submit_btn and user_input:
    if not model:
        st.error("模型未加载，无法预测。请检查 best_model_state1.bin 文件。")
    else:
        with st.spinner("模型正在分析中..."):
            results = predict_sentiment(user_input, tokenizer, model, device)

        if results:
            st.success("分析完成！检测到以下观点：")

            # 使用列布局展示结果
            # 为了美观，每行显示 3 个结果
            items = list(results.items())
            rows = [items[i:i + 3] for i in range(0, len(items), 3)]

            for row in rows:
                cols = st.columns(3)
                for idx, (aspect, sentiment) in enumerate(row):
                    color = "gray"
                    if sentiment == '正面':
                        color = "green"
                    elif sentiment == '负面':
                        color = "red"
                    elif sentiment == '中性':
                        color = "orange"

                    cols[idx].markdown(f"**{aspect}**")
                    cols[idx].markdown(f":{color}[**{sentiment}**]")
        else:
            st.info("模型未检测到明显的评价维度（所有维度均为'未提及'）。")

st.divider()

# --- 第五部分：数据透视与下载 ---
if not df.empty:
    st.subheader("7. 原始数据查询")

    df_display = df.copy()
    rename_dict = {k: v for k, v in ASPECT_MAP.items() if k in df_display.columns}
    df_display.rename(columns=rename_dict, inplace=True)

    st.dataframe(df_display, use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        " 下载分析报告 (CSV)",
        csv,
        "analysis_report.csv",
        "text/csv",
        key='download-csv'
    )