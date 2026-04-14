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
vars = ['Age', 'N', 'M', 'Stage', 'Tumor_size', 'Perineural_invasion', 'CEA', 'CA199', 'ALB']

# 🌟 二分类转换规则与输入提示
binary_rules = {
    'Age': {
        'hint': '≤65 → 0；＞65 → 1',
        'type': 'numeric',
        'min_val': 1,
        'max_val': 100,
        'step': 1,
        'threshold': 65,
        'convert': lambda x: 0 if int(x) <= 65 else 1
    },
    'N': {
        'hint': '0-1 → 0；2-3 → 1',
        'type': 'categorical',
        'options': [1, 2, 3],
        'convert': lambda x: 0 if int(x) <= 1 else 1
    },
    'M': {
        'hint': 'No → 0；Yes → 1',
        'type': 'categorical',
        'options': ['No', 'Yes'],
        'convert': lambda x: 0 if x == 'No' else 1
    },
    'Stage': {
        'hint': '0-2 → 0；3-4 → 1',
        'type': 'categorical',
        'options': [1, 2, 3, 4],
        'convert': lambda x: 0 if int(x) <= 2 else 1
    },
    'Tumor_size': {
        'hint': '≤3.9 cm → 0；＞3.9 cm → 1',
        'type': 'numeric',
        'convert': lambda x: 0 if float(x) <= 3.9 else 1
    },
    'Perineural_invasion': {
        'hint': 'No → 0；Yes → 1',
        'type': 'categorical',
        'options': ['No', 'Yes'],
        'convert': lambda x: 0 if x == 'No' else 1
    },
    'CEA': {
        'hint': '≤5.00 ng/ml → 0；＞5.00 ng/ml → 1',
        'type': 'numeric',
        'convert': lambda x: 0 if float(x) <= 5.0 else 1
    },
    'CA199': {
        'hint': '≤37.00 U/ml → 0；＞37.00 U/ml → 1',
        'type': 'numeric',
        'convert': lambda x: 0 if float(x) <= 37.0 else 1
    },
    'ALB': {
        'hint': '≤40.0 g/L → 0；＞40.0 g/L → 1',
        'type': 'numeric',
        'convert': lambda x: 0 if float(x) <= 40.0 else 1
    }
}

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

# 🌟 根据变量类型生成输入框，并显示二分类提示
input_values = {}
for var in vars:
    rule = binary_rules.get(var, {})
    hint = rule.get('hint', '')
    label_text = f"{var}  ({hint})" if hint else var

    if rule.get('type') == 'categorical':
        # 分类变量：下拉选择
        options = rule.get('options', [])
        input_values[var] = st.sidebar.selectbox(
            label=label_text,
            options=options,
            key=f"input_{var}"
        )
    else:
        # 数值变量：数字输入框
        input_values[var] = st.sidebar.number_input(
            label=label_text,
            min_value=rule.get('min_val', 0.0),
            max_value=rule.get('max_val', None),
            value=rule.get('min_val', 0.0),
            step=rule.get('step', 0.1),
            key=f"input_{var}"
        )

# 加载模型（确保模型路径正确）
try:
    mm = joblib.load('./LightGBM.pkl')
except FileNotFoundError:
    st.error("模型文件 'LightGBM.pkl' 未找到，请检查路径！")
    st.stop()

# 提交按钮逻辑
if st.sidebar.button("Submit"):
    # 🌟 将原始输入值转换为二分类值
    binary_values = {}
    for var in vars:
        rule = binary_rules.get(var, {})
        convert_fn = rule.get('convert', lambda x: x)
        binary_values[var] = convert_fn(input_values[var])

    # 构建输入数据框（使用二分类转换后的值）
    X = pd.DataFrame([list(binary_values.values())], columns=vars)

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
        st.markdown(f'<p style="font-size:28px; font-weight:bold; color:#1a3a6b;">5‑year poor prognosis risk : {result_prob_pos}%</p>', unsafe_allow_html=True)


        # 🌟 核心修改3：自适应拼接新数据
        # 存储二分类转换后的值到结果表中
        new_data_row = list(binary_values.values()) + [binary_result, None]
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
                # 🌟 对上传数据的每行做二分类转换
                binary_row = {}
                for var in vars:
                    rule = binary_rules.get(var, {})
                    convert_fn = rule.get('convert', lambda x: x)
                    raw_val = row[var]
                    # 分类变量直接传字符串，数值变量转float
                    if rule.get('type') == 'categorical':
                        binary_row[var] = convert_fn(str(raw_val).strip())
                    else:
                        binary_row[var] = convert_fn(float(raw_val))

                X = pd.DataFrame([list(binary_row.values())], columns=vars)

                # 预测
                result_prob = mm.predict_proba(X)[0][1]
                result_prob_pos = round(float(result_prob) * 100, 2)

                # 获取Label（如果有）
                label = row['label'] if 'label' in df.columns else None

                # 构造新行并更新数据（存储二分类转换后的值）
                row_binary_result = 1 if result_prob_pos >= 0.74 else 0
                new_data_row = list(binary_row.values()) + [row_binary_result, label]
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
    '<div style="font-size: 12px; text-align: right;">Powered by MyLab+ X i-Research Q Consulting Team</div>',
    unsafe_allow_html=True
)


# pip list --format=freeze >requirements.txt