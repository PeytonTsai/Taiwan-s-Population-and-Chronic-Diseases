import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sqlalchemy import create_engine, inspect
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter # 導入動畫相關模組
# from PIL import Image

#######資料庫設定#######
mysqldb_user = "root"
mysqldb_password = "1234"
mysqldb_host = "127.0.0.1"
mysqldb_port = 3306
mysqldb_name = "twdb5y"
engine = create_engine("mysql+pymysql://root:1234@localhost:3306/twdb5y")

#######連線測試#######
with engine.connect() as con:
    print("已連接 MySQL")
#%%
#######匯入CSV檔到mysql#######
# 查詢資料夾裡所有檔案
dirpath = r'C:\Users\USER\Desktop\mypy\data_source_5y'
allfiles = [f for f in os.listdir(dirpath) if f.endswith('.csv')]

# MYSQL寫入CSV檔
for file in allfiles:
    file_path = os.path.join(dirpath, file)
    table_name = os.path.splitext(file)[0]

    try:
       
        try: # 用utf8讀檔
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError: # 失敗用big5
            df = pd.read_csv(file_path, encoding='big5')

        df['source_file'] = file  # 加入來源欄位

        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print(f"成功寫入：{table_name}")

    except Exception as f:
        print(f"寫入 {table_name} 失敗：{f}")

print("寫入完成")
#%%
#######連線到mysql資料庫抓資料#######
inspector = inspect(engine)
tables = inspector.get_table_names()
# 用正則抓名稱
pattern = r"1(0[4-9]|1[0-3])年行政區五歲年齡組性別人口統計_縣市"
target_tables = [t for t in tables if re.match(pattern, t)]

# 全部人口字典
total_population_dict = {}
# 60up人口字典
senior_population_dict = {}
# 男性全人口數字典
male_population_dict = {}
# 女性權人口數字典
female_population_dict = {}
# 60歲以上男性人口數字典
male_senior_population_dict = {}
# 60歲以上女性人口數字典
female_senior_population_dict = {}


for table in sorted(target_tables):
    year = int(table[:3])
    
    # 用mysql搜尋檔案
    df = pd.read_sql(f"SELECT * FROM `{table}`", con=engine)
    
    total_cols = [col for col in df.columns if re.match(r"A\d+A\d+_CNT|A100UP_5_CNT", col)]
    # 用正則在column中搜尋A開頭，\d+ 1個或多個數字_CNT結尾的 or A100UP_CNT的資料都抓取出來
    
    # 抓60歲以上的資料(男女總計資訊)
    senior_cols = [col for col in total_cols if col.startswith(
        ("A60A64", "A65A69", "A70A74", 
         "A75A79", "A80A84", "A85A89", 
         "A90A94", "A95A99", "A100UP_5_CNT")
        )]
    
    df_clean = df.iloc[1:23, 1:65].replace(',', '', regex=True)
    df_clean[total_cols] = df_clean[total_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    # 強制轉int做後續統計計算

    total_population_dict[year] = df_clean[total_cols].sum().sum()
    # 各年齡層的column加總完之後再加總一次，得出當年總人口數
    senior_population_dict[year] = df_clean[senior_cols].sum().sum()
    # 同上，60歲以上的加總
    
    
    
    ######男女分開抓資訊######
    # 所有男性col
    male_cols = [col for col in df.columns 
                 if col.endswith("_M_CNT")]
    # 所有女性col
    female_cols = [col for col in df.columns
                   if col.endswith("_F_CNT")]
    # 60歲以上男性col
    senior_male_cols = [col for col in male_cols
                        if col.startswith(
                                ("A60A64","A65A69", "A70A74", 
                                 "A75A79", "A80A84", "A85A89", 
                                 "A90A94",  "A95A99","A100UP_5")
                                )
                        ]
    # 60歲以上女性col
    senior_female_cols = [col for col in female_cols
                          if col.startswith(
                                  ("A60A64","A65A69", "A70A74", 
                                   "A75A79", "A80A84", "A85A89", 
                                   "A90A94",  "A95A99","A100UP_5")
                                  )
                          ]
    
    # 轉換成int
    df_clean[male_cols + female_cols] = df_clean[male_cols + female_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    # 男性年度人口加總
    male_population_dict[year] = df_clean[male_cols].sum().sum()
    # 女性年度人口加總
    female_population_dict[year] = df_clean[female_cols].sum().sum()
    # 60歲以上男性年度人口加總
    male_senior_population_dict[year] = df_clean[senior_male_cols].sum().sum()
    # 60歲以上女性年度人口加總
    female_senior_population_dict[year] = df_clean[senior_female_cols].sum().sum()
    
#######整理計算DataFrame不分男女#######
sum_df = pd.DataFrame({
    '總人口數': total_population_dict,
    '60歲以上人口數': senior_population_dict
})
sum_df['60歲以上成長率(%)'] = sum_df['60歲以上人口數'].pct_change() * 100
# 104年沒有成長率，沒有103年的資料可以做計算(因為抓取104年~113年的資訊)
sum_df['60歲以上佔比(%)'] = (sum_df['60歲以上人口數'] / sum_df['總人口數']) * 100
print(sum_df)

#######整理計算DataFrame男女分開統計#######
sum_male_df = pd.DataFrame(
    {
    '男性總人口數': male_population_dict,
    '男性60歲以上人口數': male_senior_population_dict
    }
)

sum_male_df['男性60歲以上成長率(%)'] = sum_male_df['男性60歲以上人口數'].pct_change() * 100
sum_male_df['男性60歲以上佔全國男性人口比(%)'] = (sum_male_df['男性60歲以上人口數'] / sum_male_df['男性總人口數']) * 100
print("男性數據：")
print(sum_male_df)

sum_female_df = pd.DataFrame(
    {
    '女性總人口數': female_population_dict,
    '女性60歲以上人口數': female_senior_population_dict
    }
)
sum_female_df['女性60歲以上成長率(%)'] = sum_female_df['女性60歲以上人口數'].pct_change() * 100
sum_female_df['女性60歲以上佔全國女性人口比(%)'] = (sum_female_df['女性60歲以上人口數'] / sum_female_df['女性總人口數']) * 100
print("\n女性數據：")
print(sum_female_df)

#%%
########################
#######畫圖相關開始#######
########################
years = sorted(total_population_dict.keys())

# 顯示年份民國改西元
western_years = [year + 1911 for year in years]
# 每年總人口
total_population = [total_population_dict[y] for y in years]
# 每年60up人口
senior_pops = [senior_population_dict[y] for y in years]
# 成長率 公式:當年-前一年/前一年*100%
growth_rates = [0] + [(senior_pops[i] - senior_pops[i-1]) / senior_pops[i-1] * 100 for i in range(1, len(senior_pops))]
#print(senior_pops[0])
# 60up占比率 公式: 60up人口/全國人口*100%
# 用zip可以帶入兩個
senior_ratios = [s / t * 100 for s, t in zip(senior_pops, total_population)]

#######圖表中文字#######
plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False

#######換算成千萬人單位#######
def millions(x, pos):
    return '%1.3f' % (x * 1e-7)

#######換算成百萬人單位#######
def hundred_thousands(x, pos):
    return '%1.1f' % (x * 1e-6)

age_groups = ['A0A4', 'A5A9', 'A10A14', 'A15A19', 'A20A24', 'A25A29', 'A30A34',
                  'A35A39', 'A40A44', 'A45A49', 'A50A54', 'A55A59', 'A60A64', 'A65A69',
                  'A70A74', 'A75A79', 'A80A84', 'A85A89', 'A90A94', 'A95A99', 'A100UP_5']
age_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
              '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
              '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
       
#%%
#######全台各年份60歲以上資訊圖表#######
# 圖表尺寸先設定
plt.figure(figsize=(14, 10))

# 圖表資訊
# 1. 總人口數
plt.subplot(2, 2, 1)
plt.plot(western_years, total_population, marker='8', color='tab:blue',linewidth=3, markersize=10)
plt.title("2015-2024年 台灣不分年齡人口總數(男性+女性)")
plt.xlabel("年份")
plt.ylabel("人口數(千萬人)")
plt.xticks(western_years)
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')  # 關閉Y軸的科學記號顯示數字資訊
formatter_millions = ticker.FuncFormatter(millions)
plt.gca().yaxis.set_major_formatter(formatter_millions)

# 2. 60歲以上人口數
plt.subplot(2, 2, 2)
plt.plot(western_years, senior_pops, marker='8', color='#99CCCC',linewidth=3, markersize=10)
plt.title("2015-2024年 台灣60歲以上人口數")
plt.xlabel("年份")
plt.ylabel("人口數(百萬人)")
plt.xticks(western_years)
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

# 凸顯2023年的設定
hightlight = 2023
index_2023 = western_years.index(hightlight)
hightlight_2023 = senior_pops[index_2023]
plt.text(
        hightlight, hightlight_2023 - 300000, '2023年\n已突破600萬人',
        {'fontsize': 14,
         'color':'#333333',
         'bbox':
             {'facecolor': '#993333',
              'alpha': 0.2,
              'pad': 4,
              'edgecolor': 'black',
              'linewidth': 0.5,}                    
             },                  
        ha='center',    
        va='center'      
        )

# 凸顯2023年的標記點，做不同記號跟顏色
plt.plot(
    western_years[index_2023], 
    hightlight_2023, 
    marker='v', 
    color='#993333',
    markersize=16
    )

# 3. 成長百分比 
plt.subplot(2, 2, 3)
plt.bar(western_years, growth_rates, color='#add8e6')
plt.title("台灣60歲以上人口成長百分比")
plt.xlabel("年份")
plt.ylabel("成長率 (%)")
plt.xticks(western_years)
plt.grid(False)

# 4. 佔總人口百分比
plt.subplot(2, 2, 4)
plt.plot(western_years, senior_ratios, marker='8', color='#d2b48c',linewidth=3, markersize=10)
plt.title("台灣60歲以上人口佔比")
plt.xlabel("年份")
plt.ylabel("百分比 (%)")
plt.xticks(western_years)
plt.grid(True)

# 凸顯2022年
hightlight = 2022
index_2022 = western_years.index(hightlight)
hightlight_2022 = senior_ratios[index_2022]
plt.text(hightlight, hightlight_2022 - 2, '2022年\n60歲以上人口\n已達全台人口的四分之一',
          {'fontsize': 14,
           'color':'#333333',
           'bbox':
              {'facecolor': '#993333',
                'alpha': 0.2,            
                'pad': 4,
                'edgecolor': 'black',
                'linewidth': 0.5,
          }},               
          ha='center',
          va='center'
          )
# 獨立標記符號
plt.plot(
    western_years[index_2022], 
    hightlight_2022, 
    marker='v', 
    color='#993333', 
    markersize=16
    )
plt.tight_layout()
plt.show()

#%%
####這是全年齡層####
plt.figure(figsize=(12, 8))

for i, age_group in enumerate(age_groups):
    population_by_year = []
    for table in sorted(target_tables):
        df = pd.read_sql(f"SELECT * FROM `{table}`", con=engine)
        df_clean = df.iloc[1:23, :].replace(',', '', regex=True)
        population_col = [col for col in df.columns if col.startswith(age_group) and col.endswith('_CNT')]
        if population_col:
            df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            population_by_year.append(df_clean[population_col].sum().sum())
        elif age_group == 'A100UP_5': # 處理百歲以上
            population_col = [col for col in df.columns if col.startswith(age_group) and col.endswith('_CNT')]
            if population_col:
                df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                population_by_year.append(df_clean[population_col].sum().sum())

    plt.plot(western_years, population_by_year, marker='o', label=age_labels[i])

plt.xlabel("年份")
plt.ylabel("人口數(百萬人)")
plt.title("台灣各年齡組人口數變化 (2015-2024)")
plt.xticks(western_years)
plt.grid(True)
plt.legend(title="年齡組", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.ticklabel_format(style='plain', axis='y')  # 關閉y軸的科學記號
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.show()

#%%
# 對每個年份製作小提琴圖
# 添加這一行，定義 sorted_target_tables
sorted_target_tables = sorted(target_tables, key=lambda x: int(x[:3]))

# 對每個年份製作提琴圖
for table in sorted_target_tables:
    current_year_roc = int(table[:3]) # 民國年
    current_year_western = current_year_roc + 1911 # 西元年

    data_yearly = [] # 儲存單一年份的數據
    df_raw_year = pd.read_sql(f"SELECT * FROM `{table}`", con=engine)
    # df_clean_year = df_raw_year.iloc[1:23, :].replace(',', '', regex=True) # 前22行是縣市數據，第23行是總計

    # 在這裡對整個 DataFrame 進行替換逗號和轉換為數值
    # 將所有可能包含數值的欄位轉換
    data_columns_to_convert = [col for col in df_raw_year.columns if '_CNT' in col]
    
    # 複製一份 DataFrame 以避免 SettingWithCopyWarning
    df_processed_year = df_raw_year.iloc[0:23, :].copy() 
    
    # 替換逗號
    for col in data_columns_to_convert:
        if col in df_processed_year.columns: # 確保欄位存在於處理後的 DataFrame 中
            df_processed_year[col] = df_processed_year[col].replace(',', '', regex=True)
            # 將這些欄位直接轉換為數值，非數字會變成 NaN
            df_processed_year[col] = pd.to_numeric(df_processed_year[col], errors='coerce')

    # 確保 '區域別' 欄位存在且可以作為縣市名稱
    # 假設 '區域別' 是資料框的第一列
    city_names = df_processed_year.iloc[:, 1].tolist() # 獲取所有縣市名稱

    for i, age_group_raw in enumerate(age_groups):
        male_col_name = None
        female_col_name = None

        # 找到正確的男性欄位名稱
        for col in df_raw_year.columns: # 使用 df_raw_year.columns 來檢查所有原始欄位名稱
            if col.startswith(age_group_raw) and col.endswith('_M_CNT'):
                male_col_name = col
                break
        if male_col_name is None and age_group_raw == 'A100UP_5':
            for col in df_raw_year.columns:
                if col == 'A100UP_M_CNT': # 處理舊資料
                    male_col_name = col
                    break
        
        # 找到正確的女性欄位名稱
        for col in df_raw_year.columns:
            if col.startswith(age_group_raw) and col.endswith('_F_CNT'):
                female_col_name = col
                break
        if female_col_name is None and age_group_raw == 'A100UP_5':
            for col in df_raw_year.columns:
                if col == 'A100UP_F_CNT': # 處理舊資料
                    female_col_name = col
                    break

        # 對於每個縣市，獲取其男性和女性人口數
        for row_idx, city in enumerate(city_names):
            
            # 確保索引在 df_processed_year 範圍內
            if row_idx >= len(df_processed_year):
                continue
            
            male_population = 0
            female_population = 0

            if male_col_name and male_col_name in df_processed_year.columns:
                val = df_processed_year.iloc[row_idx][male_col_name]
                male_population = int(val) if not pd.isna(val) else 0
            
            if female_col_name and female_col_name in df_processed_year.columns:
                val = df_processed_year.iloc[row_idx][female_col_name]
                female_population = int(val) if not pd.isna(val) else 0

            # 只有當人口數大於 0 時才加入，且每個縣市作為一個獨立的數據點
            if male_population > 0:
                data_yearly.append({'Year': current_year_western, 'Gender': '男性', 'Age Group': age_labels[i], 'Population': male_population, 'City': city})
            if female_population > 0:
                data_yearly.append({'Year': current_year_western, 'Gender': '女性', 'Age Group': age_labels[i], 'Population': female_population, 'City': city})


    if not data_yearly:
        # print(f"WARN: 年份 {current_year_western} 無法收集到任何人口數據，將不會繪製提琴圖。")
        plt.close() # 關閉空的 figure
        continue

    df_violin_yearly = pd.DataFrame(data_yearly)
    # print(f"Year {current_year_western}: data_yearly 總共 {len(data_yearly)} 筆資料。")
    # print("提琴圖 DataFrame 範例 (前5行):")
    # print(df_violin_yearly.head())
    # print("提琴圖 DataFrame 資訊:")
    # df_violin_yearly.info()
    
    # 計算每個年齡組和性別的總人口數或平均人口數
    # 這裡計算總人口數
    df_pyramid = df_violin_yearly.groupby(['Age Group', 'Gender'])['Population'].sum().reset_index()

    # 為了繪製金字塔，需要將男性人口數變成負值
    df_pyramid['Population_Male'] = df_pyramid.apply(lambda row: -row['Population'] if row['Gender'] == '男性' else 0, axis=1)
    df_pyramid['Population_Female'] = df_pyramid.apply(lambda row: row['Population'] if row['Gender'] == '女性' else 0, axis=1)

    age_order_pyramid = age_labels

    # 創建一個新的 DataFrame，用於繪圖
    df_pyramid_plot = pd.DataFrame({
        'Age Group': age_order_pyramid
    })
    df_pyramid_plot = pd.merge(df_pyramid_plot, df_pyramid[df_pyramid['Gender'] == '男性'][['Age Group', 'Population_Male']], on='Age Group', how='left').fillna(0)
    df_pyramid_plot = pd.merge(df_pyramid_plot, df_pyramid[df_pyramid['Gender'] == '女性'][['Age Group', 'Population_Female']], on='Age Group', how='left').fillna(0)

    # 繪製人口金字塔
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 繪製男性部分 (左側)
    ax.barh(df_pyramid_plot['Age Group'], df_pyramid_plot['Population_Male'], color='skyblue', label='男性')

    # 繪製女性部分 (右側)
    ax.barh(df_pyramid_plot['Age Group'], df_pyramid_plot['Population_Female'], color='salmon', label='女性')
    
    # --- 這裡修改 X 軸的範圍為固定值 150 萬，並手動設定刻度 ---
    fixed_max_population = 1_500_000 
    ax.set_xlim(-fixed_max_population, fixed_max_population)
    
    # 手動設定刻度點，例如從 -150萬 到 150萬，每隔 30萬 一個刻度
    custom_ticks = np.arange(-fixed_max_population, fixed_max_population + 1, 300_000)
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels([f'{abs(int(t)):,}' for t in custom_ticks]) 
    #  # --- 這裡修改 X 軸的範圍為固定值 150 萬 ---
    # fixed_max_population = 1_500_000 
    # ax.set_xlim(-fixed_max_population, fixed_max_population)
    # ticks = ticker.MaxNLocator(nbins=5).tick_values(-fixed_max_population, fixed_max_population) 
    # ax.set_xticks(ticks)
    # ax.set_xticklabels([f'{abs(int(t)):,}' for t in ticks]) 

    ax.set_title(f'{current_year_western}年 五歲年齡組性別人口統計小提琴圖', fontsize=16)
    ax.set_xlabel('人口數', fontsize=12)
    ax.set_ylabel('年齡組', fontsize=12)
    ax.invert_yaxis() # 將年輕的年齡組放在底部
    ax.legend(title='性別')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # plt.show()
    
    # 存檔
    # 定義目標儲存目錄的根路徑
    base_output_directory = r'C:\Users\USER\Downloads\python'
    # 定義子資料夾名稱
    myfolder = 'chart'
    file_name = f'五歲年齡組性別人口統計小提琴圖_{current_year_western}年.png'
    full_save_path = os.path.join(base_output_directory,myfolder)

    # 檢查並創建資料夾 (如果不存在的話，這很重要！)
    if not os.path.exists(myfolder):
        os.makedirs(myfolder)
        print(f"已創建資料夾：{myfolder}")

    # 定義圖片的檔案名稱
    file_name = f'五歲年齡組性別人口統計小提琴圖_{current_year_western}年.png'

    # 組合完整的檔案儲存路徑 (資料夾路徑 + 檔案名稱)
    full_save_path = os.path.join(myfolder, file_name)

    plt.savefig(full_save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() # 關閉圖形以釋放記憶體


print(f"所有 {len(sorted_target_tables)} 張圖表已儲存至：{myfolder}")

#%%
#動態圖檔
# 提前載入所有年份的數據
sorted_target_tables = sorted(target_tables, key=lambda x: int(x[:3]))
all_years_data = {}
for table in sorted_target_tables:
    current_year_roc = int(table[:3])
    current_year_western = current_year_roc + 1911

    data_for_current_year = []
    df_raw_year = pd.read_sql(f"SELECT * FROM `{table}`", con=engine)

    data_columns_to_convert = [col for col in df_raw_year.columns if '_CNT' in col]
    df_processed_year = df_raw_year.iloc[1:23, :].copy() 
    
    for col in data_columns_to_convert:
        if col in df_processed_year.columns:
            df_processed_year[col] = df_processed_year[col].replace(',', '', regex=True)
            df_processed_year[col] = pd.to_numeric(df_processed_year[col], errors='coerce')

    city_names = df_processed_year.iloc[:, 0].tolist()

    for i, age_group_raw in enumerate(age_groups):
        male_col_name = None
        female_col_name = None

        for col in df_raw_year.columns:
            if col.startswith(age_group_raw) and col.endswith('_M_CNT'):
                male_col_name = col
                break
        if male_col_name is None and age_group_raw == 'A100UP_5':
            for col in df_raw_year.columns:
                if col == 'A100UP_M_CNT':
                    male_col_name = col
                    break
        
        for col in df_raw_year.columns:
            if col.startswith(age_group_raw) and col.endswith('_F_CNT'):
                female_col_name = col
                break
        if female_col_name is None and age_group_raw == 'A100UP_5':
            for col in df_raw_year.columns:
                if col == 'A100UP_F_CNT':
                    female_col_name = col
                    break

        for row_idx, city in enumerate(city_names):
            if row_idx >= len(df_processed_year):
                continue
            
            male_population = 0
            female_population = 0

            if male_col_name and male_col_name in df_processed_year.columns:
                val = df_processed_year.iloc[row_idx][male_col_name]
                male_population = int(val) if not pd.isna(val) else 0
            
            if female_col_name and female_col_name in df_processed_year.columns:
                val = df_processed_year.iloc[row_idx][female_col_name]
                female_population = int(val) if not pd.isna(val) else 0

            if male_population > 0:
                data_for_current_year.append({'Year': current_year_western, 'Gender': '男性', 'Age Group': age_labels[i], 'Population': male_population, 'City': city})
            if female_population > 0:
                data_for_current_year.append({'Year': current_year_western, 'Gender': '女性', 'Age Group': age_labels[i], 'Population': female_population, 'City': city})
    
    if data_for_current_year:
        df_yearly = pd.DataFrame(data_for_current_year)
        # 為總人口數，而不是各縣市單獨的數據點
        df_pyramid_data = df_yearly.groupby(['Age Group', 'Gender'])['Population'].sum().reset_index()
        all_years_data[current_year_western] = df_pyramid_data
    else:
        print(f"WARN: 年份 {current_year_western} 無法收集到任何人口數據，將不會包含在動畫中。")

if not all_years_data:
    print("沒有足夠的數據來創建動畫。")
    exit()

# 固定X軸的最大值為120萬人
fixed_max_population = 1_200_000
x_lim_max = fixed_max_population

# 設置動畫的圖形和軸
fig, ax = plt.subplots(figsize=(12, 10))

# 獲取唯一的年齡組列表並確保排序，用於 Y 軸
age_order_pyramid = age_labels

# 初始化 bars 讓動畫更新它們
bars_male = ax.barh(age_order_pyramid, np.zeros(len(age_order_pyramid)), color='skyblue', label='男性')
bars_female = ax.barh(age_order_pyramid, np.zeros(len(age_order_pyramid)), color='salmon', label='女性')

ax.set_xlabel('人口數', fontsize=12)
ax.set_ylabel('年齡組', fontsize=12)

# --- 這裡修改初始標題，並將 Text 對象儲存起來 ---
ax.set_title(f'{list(all_years_data.keys())[0]}年 五歲年齡組性別人口統計小提琴圖', fontsize=16) # 初始標題

ax.invert_yaxis() # 將年輕的年齡組放在底部
ax.legend(title='性別')
ax.grid(True, linestyle='--', alpha=0.7)

# 設定固定的 X 軸範圍和刻度
ax.set_xlim(-x_lim_max, x_lim_max)
ticks = ticker.MaxNLocator(nbins=5).tick_values(-x_lim_max, x_lim_max)
ax.set_xticks(ticks)
ax.set_xticklabels([f'{abs(int(t)):,}' for t in ticks])

# 動畫更新函數
def update(frame):
    year_western = list(all_years_data.keys())[frame]
    
    df_pyramid = all_years_data[year_western].copy()
    
    df_pyramid['Population_Male'] = df_pyramid.apply(lambda row: -row['Population'] if row['Gender'] == '男性' else 0, axis=1)
    df_pyramid['Population_Female'] = df_pyramid.apply(lambda row: row['Population'] if row['Gender'] == '女性' else 0, axis=1)

    df_pyramid_plot = pd.DataFrame({
        'Age Group': age_order_pyramid
    })
    df_pyramid_plot = pd.merge(df_pyramid_plot, df_pyramid[df_pyramid['Gender'] == '男性'][['Age Group', 'Population_Male']], on='Age Group', how='left').fillna(0)
    df_pyramid_plot = pd.merge(df_pyramid_plot, df_pyramid[df_pyramid['Gender'] == '女性'][['Age Group', 'Population_Female']], on='Age Group', how='left').fillna(0)

    for i, age_label in enumerate(age_order_pyramid):
        male_data_row = df_pyramid_plot[(df_pyramid_plot['Age Group'] == age_label)]
        
        male_pop = 0
        female_pop = 0
        
        if not male_data_row.empty:
            male_pop = male_data_row['Population_Male'].iloc[0] if 'Population_Male' in male_data_row.columns and not male_data_row['Population_Male'].isnull().all() else 0
            female_pop = male_data_row['Population_Female'].iloc[0] if 'Population_Female' in male_data_row.columns and not male_data_row['Population_Female'].isnull().all() else 0
        
        bars_male[i].set_width(male_pop)
        bars_female[i].set_width(female_pop)

    # --- 這裡修改 update 函數中的標題格式，使其與初始標題一致 ---
    ax.set_title(f'{year_western}年 五歲年齡組性別人口統計小提琴圖', fontsize=16)

    # 將所有被修改的 Artist 對象返回
    return (*bars_male.patches, *bars_female.patches, ax.title)


# 創建動畫
ani = FuncAnimation(fig, update, frames=len(all_years_data), interval=500, blit=True)

# 保存動畫
print("正在生成動畫，這可能需要一些時間...")
try:
    ani.save('population_pyramid.gif', writer='pillow', dpi=150)
    print("動畫已保存為 population_pyramid.gif")
except Exception as e:
    print(f"保存動畫時發生錯誤: {e}")
    print("錯誤類型:", type(e).__name__)
    print("錯誤訊息:", e)

plt.show()
#%%
####60歲以上####柱狀圖
num_years = len(western_years)
age_labels_senior = ['60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
age_groups_raw_senior = ['A60A64', 'A65A69', 'A70A74', 'A75A79', 'A80A84', 'A85A89', 'A90A94', 'A95A99', 'A100UP_5']
num_age_groups_senior = len(age_labels_senior)
bar_width = 1 / num_age_groups_senior
x = np.arange(num_years)
colors = plt.cm.get_cmap('tab20c', num_age_groups_senior) # Set3 tab20c

plt.figure(figsize=(15, 6)) # 調整圖表大小以適應較少的年齡組

for i, age_group_label in enumerate(age_labels_senior):
    population_by_year = []
    age_group_raw = age_groups_raw_senior[i]
    for year_raw in sorted(target_tables):
        df = pd.read_sql(f"SELECT * FROM `{year_raw}`", con=engine)
        df_clean = df.iloc[1:23, :].replace(',', '', regex=True)
        population_col = [col for col in df.columns if col.startswith(age_group_raw) and col.endswith('_CNT')]
        population = 0
        if population_col:
            df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            population = df_clean[population_col].sum().sum()
        elif age_group_raw == 'A100UP_5':
            population_col = [col for col in df.columns if col.startswith(age_group_raw) and col.endswith('_CNT')]
            if population_col:
                df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                population = df_clean[population_col].sum().sum()
            else:
                population_col_alt = [col for col in df.columns if col == 'A100UP_CNT']
                if population_col_alt:
                    df_clean[population_col_alt] = df_clean[population_col_alt].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                    population = df_clean[population_col_alt].sum().sum()
        population_by_year.append(population)
    plt.bar(x + i * bar_width, population_by_year, bar_width, label=age_group_label,color=colors(i))

plt.xlabel("年份")
plt.ylabel("人口數(百萬)")
plt.title("台灣 60 歲以上各年齡組人口數 (2015-2024)")
plt.xticks(x + (num_age_groups_senior * bar_width) / 2 - bar_width / 2, western_years, rotation=45, ha="right")
plt.legend(title="年齡組", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(style='plain', axis='y')
plt.grid(True)
plt.tight_layout()
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.show()

#%%
####60歲曲線圖

plt.figure(figsize=(12, 6))

for i, age_group_label in enumerate(age_labels_senior):
    population_by_year = []
    age_group_raw = age_groups_raw_senior[i]
    for year_raw in sorted(target_tables):
        df = pd.read_sql(f"SELECT * FROM `{year_raw}`", con=engine)
        df_clean = df.iloc[1:23, :].replace(',', '', regex=True)
        population_col = [col for col in df.columns if col.startswith(age_group_raw) and col.endswith('_CNT')]
        population = 0
        if population_col:
            df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            population = df_clean[population_col].sum().sum()
        elif age_group_raw == 'A100UP_5':
            population_col = [col for col in df.columns if col.startswith(age_group_raw) and col.endswith('_CNT')]
            if population_col:
                df_clean[population_col] = df_clean[population_col].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                population = df_clean[population_col].sum().sum()
            else:
                population_col_alt = [col for col in df.columns if col == 'A100UP_CNT']
                if population_col_alt:
                    df_clean[population_col_alt] = df_clean[population_col_alt].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
                    population = df_clean[population_col_alt].sum().sum()
        population_by_year.append(population)
    plt.plot(western_years, population_by_year, marker='o', label=age_group_label, color=colors(i)) # 應用顏色

plt.xlabel("年份")
plt.ylabel("人口數(百萬)")
plt.title("台灣 60 歲以上各年齡組人口數變化 (2015-2024)")
plt.xticks(western_years)
plt.grid(True)
plt.legend(title="年齡組", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.show()

#%% 暫時不需要
#######全台各年份60歲以上男性跟女性各別資訊的圖表#######
male_total_pops = sum_male_df['男性總人口數'].tolist()
female_total_pops = sum_female_df['女性總人口數'].tolist()
male_senior_pops = sum_male_df['男性60歲以上人口數'].tolist()
female_senior_pops = sum_female_df['女性60歲以上人口數'].tolist()
male_growth_rates = [0] + sum_male_df['男性60歲以上成長率(%)'].fillna(0).tolist()[1:]
female_growth_rates = [0] + sum_female_df['女性60歲以上成長率(%)'].fillna(0).tolist()[1:]
male_senior_ratios = sum_male_df['男性60歲以上佔全國男性人口比(%)'].tolist()
female_senior_ratios = sum_female_df['女性60歲以上佔全國女性人口比(%)'].tolist()

# 男性人口相關圖表
# 1. 總人口數
plt.figure(figsize=(14, 10))  # 設定男性圖表的尺寸
plt.subplot(2, 2, 1)
plt.plot(western_years, male_total_pops, marker='o', color='#003060', label='男性')
plt.title("各年份男性總人口數")
plt.xlabel("年份")
plt.ylabel("")
plt.xticks(western_years)
plt.grid(True)
formatter_millions_male = ticker.FuncFormatter(millions)
plt.gca().yaxis.set_major_formatter(formatter_millions_male)
# 移除 plt.ticklabel_format(style='plain', axis='y')

# 2. 60歲以上人口數
plt.subplot(2, 2, 2)
plt.plot(western_years, male_senior_pops, marker='o', color='#005AB5', label='男性')
plt.title("各年份男性60歲以上人口數")
plt.xlabel("年份")
plt.ylabel("")
plt.xticks(western_years)
plt.grid(True)
formatter_hundred_thousands_male = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands_male)
# 移除 plt.ticklabel_format(style='plain', axis='y')

# 3. 成長百分比
plt.subplot(2, 2, 3)
plt.bar(western_years, male_growth_rates, color='#6699CC', label='男性')
plt.title("男性60歲以上人口成長百分比")
plt.xlabel("年份")
plt.ylabel("成長率 (%)")
plt.xticks(western_years)
plt.grid(True)

# 4. 佔總人口百分比
plt.subplot(2, 2, 4)
plt.plot(western_years, male_senior_ratios, marker='o', color='#005757', label='男性')
plt.title("男性60歲以上人口佔男性總人口比例")
plt.xlabel("年份")
plt.ylabel("百分比 (%)")
plt.xticks(western_years)
plt.grid(True)
plt.tight_layout()


# 女性人口相關圖表
# 1. 總人口數
plt.figure(figsize=(14, 10))  # 設定女性圖表的尺寸
plt.subplot(2, 2, 1)
plt.plot(western_years, female_total_pops, marker='o', color='salmon', label='女性')
plt.title("各年份女性總人口數")
plt.xlabel("年份")
plt.ylabel("")
plt.xticks(western_years)
plt.grid(True)
formatter_millions_female = ticker.FuncFormatter(millions)
plt.gca().yaxis.set_major_formatter(formatter_millions_female)
# 移除 plt.ticklabel_format(style='plain', axis='y')


# 2. 60歲以上人口數
plt.subplot(2, 2, 2)
plt.plot(western_years, female_senior_pops, marker='o', color='mediumseagreen', label='女性')
plt.title("各年份女性60歲以上人口數")
plt.xlabel("年份")
plt.ylabel("")
plt.xticks(western_years)
plt.grid(True)
formatter_hundred_thousands_female = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands_female)
# 移除 plt.ticklabel_format(style='plain', axis='y')


# 3. 成長百分比
plt.subplot(2, 2, 3)
plt.bar(western_years, female_growth_rates, color='tomato', label='女性')
plt.title("女性60歲以上人口成長百分比")
plt.xlabel("年份")
plt.ylabel("成長率 (%)")
plt.xticks(western_years)
plt.grid(True)

# 4. 佔總人口百分比
plt.subplot(2, 2, 4)
plt.plot(western_years, female_senior_ratios, marker='o', color='navajowhite', label='女性')
plt.title("女性60歲以上人口佔女性總人口比例")
plt.xlabel("年份")
plt.ylabel("百分比 (%)")
plt.xticks(western_years)
plt.grid(True)

plt.tight_layout()
plt.show()

#%% 
#######男性VS女性對照圖#######
#######全台各年份60歲以上男性跟女性各別資訊的圖表#######
male_total_pops = sum_male_df['男性總人口數'].tolist()
female_total_pops = sum_female_df['女性總人口數'].tolist()
male_senior_pops = sum_male_df['男性60歲以上人口數'].tolist()
female_senior_pops = sum_female_df['女性60歲以上人口數'].tolist()
male_growth_rates = [0] + sum_male_df['男性60歲以上成長率(%)'].fillna(0).tolist()[1:]
female_growth_rates = [0] + sum_female_df['女性60歲以上成長率(%)'].fillna(0).tolist()[1:]
male_senior_ratios = sum_male_df['男性60歲以上佔全國男性人口比(%)'].tolist()
female_senior_ratios = sum_female_df['女性60歲以上佔全國女性人口比(%)'].tolist()

plt.rcParams["font.family"] = "Microsoft JhengHei"
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(14, 10))

# 1. 總人口數
plt.subplot(2, 2, 1)
plt.plot(western_years, male_total_pops, marker='o', color='#84C1FF', label='男性')
plt.plot(western_years, female_total_pops, marker='o', color='#FF7575', label='女性')

plt.title("各年份總人口數 (男 vs 女)")
plt.xlabel("年份")
plt.ylabel("人口數(千萬)")
plt.xticks(western_years)
plt.grid(True)
plt.legend()
formatter_millions = ticker.FuncFormatter(millions)
plt.gca().yaxis.set_major_formatter(formatter_millions)

# 2. 60歲以上人口數
plt.subplot(2, 2, 2)
plt.plot(western_years, male_senior_pops, marker='o', color='#84C1FF', label='男性')
plt.plot(western_years, female_senior_pops, marker='o', color='#FF7575', label='女性')
plt.title("各年份60歲以上人口數 (男 vs 女)")
plt.xlabel("年份")
plt.ylabel("人口數(百萬)")
plt.xticks(western_years)
plt.grid(True)
plt.legend()
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

# 3. 成長百分比
plt.subplot(2, 2, 3)
plt.bar(western_years, male_growth_rates, color='#84C1FF', label='男性', width=0.35)
plt.bar([y + 0.35 for y in western_years], female_growth_rates, color='#FF7575', label='女性', width=0.35)
plt.title("60歲以上人口成長百分比 (男 vs 女)")
plt.xlabel("年份")
plt.ylabel("成長率 (%)")
plt.xticks(western_years)
plt.grid(True)
plt.legend()

# 4. 佔總人口百分比
plt.subplot(2, 2, 4)
plt.plot(western_years, male_senior_ratios, marker='o', color='#84C1FF', label='男性')
plt.plot(western_years, female_senior_ratios, marker='o', color='#FF7575', label='女性')
plt.title("60歲以上人口佔總人口比例 (男 vs 女)")
plt.xlabel("年份")
plt.ylabel("百分比 (%)")
plt.xticks(western_years)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%%
#######推算20年後的60歲以上人口#######
# --- 推算20年後的60歲以上人口使用線性迴歸預測成長率 ---
years_for_regression = np.array(western_years[1:]).reshape(-1, 1) # 轉成二維數組
growth_rates_for_regression = np.array(growth_rates[1:])

# 創建並訓練模型
model_growth = LinearRegression()
model_growth.fit(years_for_regression, growth_rates_for_regression)

# 預測未來20年的成長率
future_years_20 = np.array([western_years[-1] + i for i in range(1, 21)]).reshape(-1, 1)
predicted_growth_rates_20 = model_growth.predict(future_years_20)

# 使用預測的成長率推算未來人口
predicted_senior_pops_lr_20_years = [senior_pops[-1]]
for rate in predicted_growth_rates_20:
    next_pop = predicted_senior_pops_lr_20_years[-1] * (1 + rate / 100)
    predicted_senior_pops_lr_20_years.append(next_pop)

print("\n--- 未來20年60歲以上人口預測 (基於線性迴歸成長率) ---")
for i, year in enumerate(future_years_20.flatten()):
    print(f"西元 {int(year)} 年: {int(predicted_senior_pops_lr_20_years[i+1])} 人 (預測成長率: {predicted_growth_rates_20[i]:.2f}%)")


# 預測結果
plt.figure(figsize=(12, 6))
plt.plot(western_years, senior_pops, marker='o', label='歷史紀錄的人口')
plt.plot(future_years_20, predicted_senior_pops_lr_20_years[1:], marker='x', linestyle='--', color='red', label='未來20年預測趨勢')
plt.title('60歲以上人口預測')
plt.xlabel('年份 (西元)')
plt.ylabel('人口數(百萬)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='y')  # 關閉y軸的科學記號
formatter_hundred_thousands = ticker.FuncFormatter(hundred_thousands)
plt.gca().yaxis.set_major_formatter(formatter_hundred_thousands)

plt.show()

