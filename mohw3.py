# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 00:00:41 2025

@author: USER
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns
from prophet import Prophet


plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False


def millions(x, pos):
    return '%1.3f' % (x * 1e-7)

def hundred_thousands(x, pos):
    return '%1.1f' % (x * 1e-6)

# 105年-111年的檔案處理
data_folder = r'C:\Users\USER\Desktop\mypy\mohw'
# 定義你的檔案基礎路徑和年份範圍
files_name = '年西醫門診(包括急診)人數統計------按疾病別、性別及年齡別分.xlsx'
start_year = 105
end_year = 111

all_diabetes_dataframes = [] #糖尿病的容器
all_hypertension_dataframes = [] #高血壓的容器

for year in range(start_year, end_year + 1):
    file_name = f'{year}{files_name}'
    file_path = os.path.join(data_folder, file_name)

    print(f"正在處理檔案：{file_path}")

    try:
        df_raw = pd.read_excel(
            file_path,
            header=[4,5,6,7,8,9], # 表頭範圍
            engine='openpyxl'
        )

        # 清理多級索引的欄位名稱
        # 只處理了level0和level1
        level0 = [col[0] for col in df_raw.columns]
        level1 = [col[1] for col in df_raw.columns]

        filled_level0 = pd.Series(level0).ffill().tolist()

        new_columns = []
        for i in range(len(df_raw.columns)):
            col_level1_val = filled_level0[i]
            col_level2_val = level1[i]

            if "Unnamed" in str(col_level1_val) and "Unnamed" not in str(col_level2_val):
                new_columns.append(str(col_level2_val).strip())
            elif "Unnamed" in str(col_level2_val) or str(col_level2_val).strip() == '':
                new_columns.append(str(col_level1_val).strip())
            else:
                new_columns.append(f"{str(col_level1_val).strip()}_{str(col_level2_val).strip()}")

        df_raw.columns = new_columns

        # 使用雙層方括號 [[50]]保持 DataFrame 格式
        df_diabetes_data = df_raw.iloc[[50]].copy() # 抓取需要的範圍糖尿病
        df_hypertension_data = df_raw.iloc[[73]].copy() # 抓取需要的範圍高血壓
        # 篩選需要處理的數值欄位
        numeric_cols_to_process = [
            col for col in df_diabetes_data.columns
            if any(keyword in col for keyword in ['總數', '男', '女', '總計', 'Total', 'Male', 'Female', '歲', '人'])
        ]

        for col in numeric_cols_to_process:
            if col in df_diabetes_data.columns:
                df_diabetes_data[col] = pd.to_numeric(
                    df_diabetes_data[col].astype(str).str.replace(',', '', regex=False),
                    errors='coerce'
                ).fillna(0).astype(int)
        
        numeric_cols_to_process2 = [
            col for col in df_hypertension_data.columns
            if any(keyword in col for keyword in ['總數', '男', '女', '總計', 'Total', 'Male', 'Female', '歲', '人'])
        ]        
                
        for col in numeric_cols_to_process2:
            if col in df_hypertension_data.columns:
                df_hypertension_data[col] = pd.to_numeric(
                    df_hypertension_data[col].astype(str).str.replace(',', '', regex=False),
                    errors='coerce'
                ).fillna(0).astype(int)

        # 為這行數據增加一個 '年份' 欄位，方便後續識別
        df_diabetes_data['年份'] = year
        df_hypertension_data['年份'] = year

        # 將處理後的單行 DataFrame 加入列表
        all_diabetes_dataframes.append(df_diabetes_data)
        all_hypertension_dataframes.append(df_hypertension_data) 
        
        
    except FileNotFoundError:
        print(f"錯誤：文件 '{file_path}' 未找到。請檢查文件路徑是否正確。")
    except Exception as e:
        print(f"讀取或處理 Excel 文件 '{file_path}' 時發生錯誤：{e}")
        print("請檢查文件內容和結構是否與預期相符。")

# 處理 112 年檔案
# 這是你獨立處理 112 年的區塊
file_path_112 = r'C:\Users\USER\Desktop\mypy\mohw\112年西醫門診(包括急診)人數統計------按疾病別、性別及年齡別分.xlsx'
year_112 = 112

print(f"\n正在處理檔案：{file_path_112}")

try:
    df_raw_112 = pd.read_excel(
        file_path_112,
        header=[5,6,7,8,9,10,11], #112年是7層表頭
        engine='openpyxl'
    )

    # 清理多級索引的欄位名稱
    level0_112 = [col[0] for col in df_raw_112.columns]
    level1_112 = [col[1] for col in df_raw_112.columns]
    #往前填入空值
    filled_level0_112 = pd.Series(level0_112).ffill().tolist()

    new_columns_112 = []
    for i in range(len(df_raw_112.columns)):
        col_level1_val_112 = filled_level0_112[i]
        col_level2_val_112 = level1_112[i]

        if "Unnamed" in str(col_level1_val_112) and "Unnamed" not in str(col_level2_val_112):
            new_columns_112.append(str(col_level2_val_112).strip())
        elif "Unnamed" in str(col_level2_val_112) or str(col_level2_val_112).strip() == '':
            new_columns_112.append(str(col_level1_val_112).strip())
        else:
            new_columns_112.append(f"{str(col_level1_val_112).strip()}_{str(col_level2_val_112).strip()}")

    df_raw_112.columns = new_columns_112

    # df_raw_112的第49行對應Excel檔案的第62行
    df_diabetes_data_112 = df_raw_112.iloc[[49]].copy()
    df_hypertension_data_112 = df_raw_112.iloc[[72]].copy()
   
    # 篩選需要處理的數值欄位
    numeric_cols_to_process_112 = [
        col for col in df_diabetes_data_112.columns
        if any(keyword in col for keyword in ['總數', '男', '女', '總計', 'Total', 'Male', 'Female', '歲', '人'])
    ]

    for col in numeric_cols_to_process_112:
        if col in df_diabetes_data_112.columns:
            df_diabetes_data_112[col] = pd.to_numeric(
                df_diabetes_data_112[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            ).fillna(0).astype(int)

    for col in numeric_cols_to_process_112:
        if col in df_hypertension_data_112.columns:
            # 修正：直接對 DataFrame 的列進行操作，並移除 [0]
            df_hypertension_data_112[col] = pd.to_numeric(
                df_hypertension_data_112[col].astype(str).str.replace(',', '', regex=False),
                errors='coerce'
            ).fillna(0).astype(int)

    # 增加年份欄位
    df_diabetes_data_112['年份'] = year_112
    df_hypertension_data_112['年份'] = year_112

    #將112年處理好的DataFrame追加到all_dataframes
    all_diabetes_dataframes.append(df_diabetes_data_112)
    all_hypertension_dataframes.append(df_hypertension_data_112)
    print(f"  成功處理 {year_112} 年數據。")

except FileNotFoundError:
    print(f"錯誤：文件 '{file_path_112}' 未找到。請檢查文件路徑是否正確。")
except Exception as e:
    print(f"讀取或處理 Excel 文件 '{file_path_112}' 時發生錯誤：{e}")
    print("請檢查文件內容和結構是否與預期相符。")


# 將所有單獨的糖尿病DataFrame合併成一個大的DataFrame
if all_diabetes_dataframes:
    # 使用 ignore_index=True 重新生成連續的索引，避免不同年份的索引衝突
    final_diabetes_combined_df = pd.concat(all_diabetes_dataframes, ignore_index=True)

    # 將年份欄位設定為index
    # 後面繪圖發生狀況，取消將年份設定為index
    # final_diabetes_combined_df = final_diabetes_combined_df.set_index('年份')

    # 刪除全是 NaN 的列
    df_diabetes_cleaned_by_col = final_diabetes_combined_df.dropna(axis=1, how='all')
    
    # 最終清理
    df_diabetes_cleaned_by_col=df_diabetes_cleaned_by_col.iloc[:,2:]

    print("\n--- 成功合併全部糖尿病數據在 df_diabetes_cleaned_by_col ---")

else:
    print("\n沒有成功載入任何數據。")

# 將所有單獨的高血壓DataFrame合併成一個大的DataFrame
if all_hypertension_dataframes:
    final_hypertension_combined_df = pd.concat(all_hypertension_dataframes, ignore_index=True)
    df_hypertension_cleaned_by_col = final_hypertension_combined_df.dropna(axis=1, how='all')
    
    #最終清理
    df_hypertension_cleaned_by_col=df_hypertension_cleaned_by_col.iloc[:,2:]
   
    print("\n--- 成功合併全部高血壓數據在 df_hypertension_cleaned_by_col ---")

else:
    print("\n沒有成功載入任何數據。請檢查檔案路徑、名稱和錯誤訊息。")
    
#%%
# 年份處理 (將民國年轉換為西元年)
if '年份' in df_diabetes_cleaned_by_col.columns:
    df_diabetes_cleaned_by_col['西元年'] = df_diabetes_cleaned_by_col['年份'] + 1911
    # 不再需要民國年份，刪除 '年份' 欄位
    df_diabetes_cleaned_by_col = df_diabetes_cleaned_by_col.drop(columns=['年份'])
    print("\n糖尿病數據已新增 '西元年' 欄位。")
    print(df_diabetes_cleaned_by_col[['西元年']].head())
else:
    print("\n錯誤：糖尿病數據中找不到 '年份' 欄位")

if '年份' in df_hypertension_cleaned_by_col.columns:
    df_hypertension_cleaned_by_col['西元年'] = df_hypertension_cleaned_by_col['年份'] + 1911
    df_hypertension_cleaned_by_col = df_hypertension_cleaned_by_col.drop(columns=['年份'])
    print("\n高血壓數據已新增 '西元年' 欄位。")
    print(df_hypertension_cleaned_by_col[['西元年']].head())
else:
    print("\n錯誤：高血壓數據中找不到 '年份' 欄位")


# 繪圖時 X 軸所需的西元年列表
if '西元年' in df_diabetes_cleaned_by_col.columns:
    western_years_for_plot = sorted(df_diabetes_cleaned_by_col['西元年'].unique().tolist())
    print("\n用於繪圖X軸的西元年列表:", western_years_for_plot)
else:
    western_years_for_plot = []

#%%# 數據可視化部分開始
#%% 
#門診人數趨勢圖
total_col_name = '總計'

# 1. 糖尿病與高血壓門診總人數趨勢 ---
if total_col_name in df_diabetes_cleaned_by_col.columns and '西元年' in df_diabetes_cleaned_by_col.columns:
    plt.figure(figsize=(12, 6))

    # X 軸現在使用 '西元年' 欄位
    sns.lineplot(x='西元年', y=total_col_name, data=df_diabetes_cleaned_by_col, marker='o', markersize=10,linewidth=3, color='#84C1FF', label='糖尿病總人數')
    sns.lineplot(x='西元年', y=total_col_name, data=df_hypertension_cleaned_by_col, marker='8', markersize=10,linewidth=3,linestyle='--', color='#FF7575', label='高血壓總人數')

    plt.title('2016-2023 年糖尿病與高血壓門診總人數趨勢', fontsize=16)
    plt.xlabel('年份', fontsize=13)
    plt.ylabel('總人數(百萬)', fontsize=13)
    plt.xticks(df_diabetes_cleaned_by_col['西元年'].unique(), fontsize=10) # 確保刻度是整數年份
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.ticklabel_format(style='plain', axis='y') # 關閉Y軸的科學記號顯示數字資訊

    formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
    plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

    plt.show()
    print("\n--- 糖尿病與高血壓門診總人數趨勢圖繪製完成---")
else:
    print(f"錯誤：'{total_col_name}' 或 '西元年' 欄位不存在於您的 DataFrame 中，無法繪製趨勢圖。")
    print("請檢查您的 DataFrame 欄位名稱。")

# # 2. 高血壓門診總人數趨勢 (單獨圖)
# if total_col_name in df_hypertension_cleaned_by_col.columns and '西元年' in df_hypertension_cleaned_by_col.columns:
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(x='西元年', y=total_col_name, data=df_hypertension_cleaned_by_col, marker='o', label='高血壓總人數')

#     plt.title('2016-2023 年高血壓門診總人數趨勢', fontsize=16)
#     plt.xlabel('年份', fontsize=12)
#     plt.ylabel('總人數(百萬)', fontsize=12)

#     plt.xticks(df_hypertension_cleaned_by_col['西元年'].unique(), fontsize=10)
#     plt.yticks(fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.legend(fontsize=11)
#     plt.tight_layout()
#     plt.ticklabel_format(style='plain', axis='y') # 關閉Y軸的科學記號顯示數字資訊

#     formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
#     plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)
#     plt.show()
#     print("\n--- 高血壓總人數趨勢圖已生成 (西元年) ---")
# else:
#     print(f"錯誤：'{total_col_name}' 或 '西元年' 欄位不存在於您的 DataFrame 中，無法繪製趨勢圖。")
#     print("請檢查您的 DataFrame 欄位名稱。")
#%%
# Prophet數據準備（時間序列預測）
# 說明網站：https://facebook.github.io/prophet/docs/quick_start.html
'''
Prophet 需要 ds 和 y 欄位
# ds 來自西元年欄位，y來自總計欄位
# 西元年欄位需要是datetime類型
'''
# 糖尿病 DataFrame 處理
df_diabetes_prophet = df_diabetes_cleaned_by_col[['西元年', '總計']].copy()
df_diabetes_prophet['西元年'] = pd.to_datetime(df_diabetes_prophet['西元年'], format='%Y')
df_diabetes_prophet = df_diabetes_prophet.rename(columns={'西元年': 'ds', '總計': 'y'})

# 高血壓 DataFrame 處理
df_hypertension_prophet = df_hypertension_cleaned_by_col[['西元年', '總計']].copy()
df_hypertension_prophet['西元年'] = pd.to_datetime(df_hypertension_prophet['西元年'], format='%Y')
df_hypertension_prophet = df_hypertension_prophet.rename(columns={'西元年': 'ds', '總計': 'y'})

#%%
# 預測糖尿病趨勢
print("\n--- 開始預測糖尿病趨勢 ---")
m_diabetes = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
m_diabetes.fit(df_diabetes_prophet)

future_diabetes = m_diabetes.make_future_dataframe(periods=10, freq='Y')
forecast_diabetes = m_diabetes.predict(future_diabetes)

fig_diabetes = m_diabetes.plot(forecast_diabetes)


plt.plot(df_diabetes_prophet['ds'], df_diabetes_prophet['y'], 'o', color='red')

# 獲取圖例原始標籤
handles, labels = fig_diabetes.gca().get_legend_handles_labels()

# 修改圖例中「Observed data points」的顏色
for i, label in enumerate(labels):
    if label == 'Observed data points':
        handles[i].set_color('red')
        # 也可以使用這些設定 set_markerfacecolor, set_markeredgecolor, markersize

# 改為中文標籤並更新
new_labels = []
for label in labels:
    if label == 'Observed data points':
        new_labels.append('歷史觀察點')
    elif label == 'Forecast':
        new_labels.append('預測線')
    elif label == 'Uncertainty interval':
        new_labels.append('預測區間')
    else:
        new_labels.append(label)

# 重新設定圖例，使用翻譯後的標籤
fig_diabetes.gca().legend(handles=handles, labels=new_labels, loc='upper left', fontsize=11)

plt.title('糖尿病門診總人數未來10年趨勢預測', fontsize=16)
plt.xlabel('年份', fontsize=13)
plt.ylabel('總人數(百萬)', fontsize=13)

formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.tight_layout()
plt.show()
print("\n--- 糖尿病趨勢預測圖已生成---")


# 預測高血壓趨勢
print("\n--- 開始預測高血壓趨勢 ---")
m_hypertension = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
m_hypertension.fit(df_hypertension_prophet)

future_hypertension = m_hypertension.make_future_dataframe(periods=10, freq='Y')
forecast_hypertension = m_hypertension.predict(future_hypertension)

fig_hypertension = m_hypertension.plot(forecast_hypertension)
plt.plot(df_hypertension_prophet['ds'], df_hypertension_prophet['y'], 'o', color='red')
handles, labels = fig_hypertension.gca().get_legend_handles_labels()

for i, label in enumerate(labels):
    if label == 'Observed data points':
        handles[i].set_color('red')

new_labels = []
for label in labels:
    if label == 'Observed data points':
        new_labels.append('歷史觀察點')
    elif label == 'Forecast':
        new_labels.append('預測線')
    elif label == 'Uncertainty interval':
        new_labels.append('預測區間')
    else:
        new_labels.append(label)

fig_hypertension.gca().legend(handles=handles, labels=new_labels, loc='upper left', fontsize=11)

plt.title('高血壓門診總人數未來10年趨勢預測', fontsize=16)
plt.xlabel('年份', fontsize=13)
plt.ylabel('總人數(百萬)', fontsize=13)

formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.tight_layout()
plt.show()
print("\n--- 高血壓趨勢預測圖已生成---")


# # # 顯示未來10年的預測值
# print("\n--- 糖尿病門診總人數未來10年預測 (百萬人) ---")
# last_historical_year_diabetes = df_diabetes_cleaned_by_col['西元年'].max()
# forecast_diabetes_future = forecast_diabetes[forecast_diabetes['ds'].dt.year > last_historical_year_diabetes][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# forecast_diabetes_future['yhat_million'] = forecast_diabetes_future['yhat'] * 1e-6
# forecast_diabetes_future['yhat_lower_million'] = forecast_diabetes_future['yhat_lower'] * 1e-6
# forecast_diabetes_future['yhat_upper_million'] = forecast_diabetes_future['yhat_upper'] * 1e-6
# print(forecast_diabetes_future[['ds', 'yhat_million', 'yhat_lower_million', 'yhat_upper_million']].to_string(index=False))


# print("\n--- 高血壓門診總人數未來10年預測 (百萬人) ---")
# last_historical_year_hypertension = df_hypertension_cleaned_by_col['西元年'].max()
# forecast_hypertension_future = forecast_hypertension[forecast_hypertension['ds'].dt.year > last_historical_year_hypertension][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# forecast_hypertension_future['yhat_million'] = forecast_hypertension_future['yhat'] * 1e-6
# forecast_hypertension_future['yhat_lower_million'] = forecast_hypertension_future['yhat_lower'] * 1e-6
# forecast_hypertension_future['yhat_upper_million'] = forecast_hypertension_future['yhat_upper'] * 1e-6
# print(forecast_hypertension_future[['ds', 'yhat_million', 'yhat_lower_million', 'yhat_upper_million']].to_string(index=False))
#%%

# 獲取指定年齡性別欄位 (根據新欄位命名規則修改並排除總計及不想要的年齡組)
def get_age_gender_columns_filtered(df, desired_age_prefixes=None):
    """
    根據 DataFrame 欄位名稱模式，獲取特定年齡層和性別的門診人數欄位。
    這些欄位預期以 '_男' 或 '_女' 結尾，並排除包含 '總計' 的欄位。
    desired_age_prefixes: 一個列表，包含要包含的年齡組前綴。如果為 None，則包含所有年齡組 (除了總計)。
    """
    age_gender_cols = []
    for col in df.columns:
        # 判斷是否以 '_男' 或 '_女' 結尾
        if col.endswith('_男') or col.endswith('_女'):
            # 排除掉包含 '總計' 的欄位
            if '總計' not in col:
                # 如果指定了 desired_age_prefixes，則進一步篩選
                if desired_age_prefixes:
                    if any(col.startswith(prefix) for prefix in desired_age_prefixes):
                        age_gender_cols.append(col)
                else: # 如果沒有指定 desired_age_prefixes，則包含所有非總計的年齡性別欄位
                    age_gender_cols.append(col)
    return age_gender_cols

# 全年齡太多了，只顯示60歲以上的資訊
desired_age_group_prefixes = [
    # '0~4歲(Years Old)',
    # '5~9歲(Years Old)',
    # '10~14歲(Years Old)',
    # '15~19歲(Years Old)',
    # '20~24歲(Years Old)',
    # '25~29歲(Years Old)',
    # '30~34歲(Years Old)',
    # '35~39歲(Years Old)',
    # '40~44歲(Years Old)',
    # '45~49歲(Years Old)',
    # '50~54歲(Years Old)',
    # '55~59歲(Years Old)',
    '60~64歲(Years Old)',
    '65~69歲(Years Old)',
    '70~74歲(Years Old)',
    '75~79歲(Years Old)',
    '80~84歲(Years Old)',
    '85歲以上(Above 85 Years Old)'
]


# 糖尿病各年齡層男女性別門診人數趨勢圖
print("\n--- 糖尿病各年齡層男女性別門診人數趨勢圖 ---")

age_gender_cols_diabetes = get_age_gender_columns_filtered(df_diabetes_cleaned_by_col, desired_age_prefixes=desired_age_group_prefixes)

if '西元年' in df_diabetes_cleaned_by_col.columns and age_gender_cols_diabetes:
    df_diabetes_melted = df_diabetes_cleaned_by_col.melt(id_vars=['西元年'],
                                                          value_vars=age_gender_cols_diabetes,
                                                          var_name='性別與年齡層',
                                                          value_name='門診人數')

    plt.figure(figsize=(16, 9))

    sns.lineplot(x='西元年', y='門診人數', hue='性別與年齡層',
                 data=df_diabetes_melted, marker='o', palette='tab20', linewidth=1.5)

    plt.title('糖尿病門診人數 - 各年齡層與性別趨勢', fontsize=20, pad=20)
    plt.xlabel('年份', fontsize=15)
    plt.ylabel('門診人數', fontsize=15)

    plt.xticks(df_diabetes_cleaned_by_col['西元年'].unique(), fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(title='性別與年齡層', bbox_to_anchor=(1.02, 1), loc='upper left',
               borderaxespad=0., fontsize=10, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()
    print("\n--- 糖尿病各年齡層男女性別門診人數趨勢圖已生成---")
else:
    print("錯誤：無法繪製糖尿病各年齡層趨勢圖。請檢查 '西元年' 欄位或年齡性別欄位是否存在。")
    if not age_gender_cols_diabetes:
        print(f"找不到符合 '_男' 或 '_女' 模式的年齡性別欄位 (且不含 '總計')。請檢查欄位名稱：{df_diabetes_cleaned_by_col.columns.tolist()}")


# 高血壓各年齡層男女性別門診人數趨勢圖
print("\n--- 高血壓各年齡層男女性別門診人數趨勢圖 ---")

age_gender_cols_hypertension = get_age_gender_columns_filtered(df_hypertension_cleaned_by_col, desired_age_prefixes=desired_age_group_prefixes)

if '西元年' in df_hypertension_cleaned_by_col.columns and age_gender_cols_hypertension:
    df_hypertension_melted = df_hypertension_cleaned_by_col.melt(id_vars=['西元年'],
                                                                  value_vars=age_gender_cols_hypertension,
                                                                  var_name='性別與年齡層',
                                                                  value_name='門診人數')

    plt.figure(figsize=(16, 9))

    sns.lineplot(x='西元年', y='門診人數', hue='性別與年齡層',
                 data=df_hypertension_melted, marker='o', palette='tab20', linewidth=1.5)

    plt.title('高血壓門診人數 - 各年齡層與性別趨勢', fontsize=20, pad=20)
    plt.xlabel('年份', fontsize=15)
    plt.ylabel('門診人數', fontsize=15)

    plt.xticks(df_hypertension_cleaned_by_col['西元年'].unique(), fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend(title='性別與年齡層', bbox_to_anchor=(1.02, 1), loc='upper left',
               borderaxespad=0., fontsize=10, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()
    print("\n--- 高血壓各年齡層男女性別門診人數趨勢圖已生成---")
else:
    print("錯誤：無法繪製高血壓各年齡層趨勢圖。請檢查 '西元年' 欄位或年齡性別欄位是否存在。")
    if not age_gender_cols_hypertension:
        print(f"找不到符合 '_男' 或 '_女' 模式的年齡性別欄位 (且不含 '總計')。請檢查欄位名稱：{df_hypertension_cleaned_by_col.columns.tolist()}")
