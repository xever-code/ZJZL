import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 定义变量列表（可任意增减，代码会自适应）
vars = ['T', 'M', 'N', 'Tumor location', 'Differentiation', 'Perineural invasion', 'CA199 ', 'CA724', 'TP before the first treatment','HBcAb IGM']

# 初始化 session_state 中的 data
# 动态生成DataFrame列名：变量列表 + 预测相关列
df_columns = vars + ['Prediction Label', 'Label']
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=df_columns)

# 页面标题
st.header('GC-Prognosis')

# 创建五列布局用于显示logo
left_column, col1, col2, col3, right_column = st.columns(5)
left_column.write("")
# 显示logo（路径根据实际情况调整）
try:
    right_column.image('./logo.jpg', caption='', width=100)
except Exception as e:
    st.warning(f"Logo图片加载失败: {e}")

# 侧边栏输入区
st.sidebar.header('Parameters input')

# 🌟 核心修改1：自适应生成输入框
# 存储所有输入值的字典
input_values = {}
for var in vars:
    # 为每个变量生成独立的数字输入框
    input_values[var] = st.sidebar.number_input(
        label=var,
        min_value=0.0,
        value=0.0,
        key=f"input_{var}"  # 唯一key避免冲突
    )

# 加载模型（确保模型路径正确）
try:
    mm = joblib.load('./LightGBM.pkl')
except FileNotFoundError:
    st.error("模型文件 'LightGBM.pkl' 未找到，请检查路径！")
    st.stop()

# 提交按钮逻辑
if st.sidebar.button("Submit"):
    # 🌟 核心修改2：自适应构建输入数据框
    # 将输入值转换为DataFrame，列顺序与vars完全一致
    X = pd.DataFrame([list(input_values.values())], columns=vars)

    try:
        # 模型预测（概率）
        result_prob = mm.predict_proba(X)[0][1]  # 取正类概率
        result_prob_pos = round(float(result_prob) * 100, 2)  # 转换为百分比并保留2位小数

        # SHAP解释（可选，如需保留）
        # explainer = shap.TreeExplainer(mm)
        # shap_values = explainer.shap_values(X)

        # 核心逻辑：判断 result_prob_pos 是否大于0.74，大于则为1，否则为0
        binary_result = 1 if result_prob_pos >= 0.74 else 0

        # 显示预测结果
        st.text(f"The probability of LightGBM is: {result_prob_pos}%")


        # 🌟 核心修改3：自适应拼接新数据
        # 构造新数据行：输入值 + 预测概率 + Label（暂为空）
        new_data_row = list(input_values.values()) + [binary_result, None]
        new_data = pd.DataFrame([new_data_row], columns=df_columns)

        # 更新session_state中的数据
        st.session_state['data'] = pd.concat(
            [st.session_state['data'], new_data],
            ignore_index=True
        )
    except Exception as e:
        st.error(f"预测过程出错: {e}")

# 文件上传功能（自适应变量）
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)

        # 检查必需列是否存在
        missing_cols = [col for col in vars if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
        else:
            # 逐行预测
            for _, row in df.iterrows():
                # 🌟 核心修改4：自适应提取上传文件中的变量
                X = pd.DataFrame([row[vars].values], columns=vars)

                # 预测
                result_prob = mm.predict_proba(X)[0][1]
                result_prob_pos = round(float(result_prob) * 100, 2)

                # 获取Label（如果有）
                label = row['label'] if 'label' in df.columns else None

                # 构造新行并更新数据
                new_data_row = row[vars].tolist() + [result_prob_pos, label]
                new_data = pd.DataFrame([new_data_row], columns=df_columns)
                st.session_state['data'] = pd.concat(
                    [st.session_state['data'], new_data],
                    ignore_index=True
                )
            st.success("文件上传并预测完成！")
    except Exception as e:
        st.error(f"文件处理出错: {e}")

# 显示所有数据
# st.write("预测结果汇总：")
st.write(st.session_state['data'])

# 页脚
st.write(
    "<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>",
    unsafe_allow_html=True
)
st.markdown(
    '<div style="font-size: 12px; text-align: right;">Powered by MyLab+ X i-Research Consulting Team</div>',
    unsafe_allow_html=True
)


# pip list --format=freeze >requirements.txt