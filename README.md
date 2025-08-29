
# Haikus for Codespaces

This is a quick node project template for demoing Codespaces. It is based on the [Azure node sample](https://github.com/Azure-Samples/nodejs-docs-hello-world). It's great!!!

Point your browser to [Quickstart for GitHub Codespaces](https://docs.github.com/en/codespaces/getting-started/quickstart) for a tour of using Codespaces with this repo.

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
import streamlit as st
# 1. 加载数据并预处理
df = pd.read_csv('shanghai_ev_data_vehicle_to_charger_ratio.csv')

# 清理 'Ev Stock' 列，去掉逗号并转换为数字
df['Ev Stock'] = df['Ev Stock'].str.replace(',', '').astype(float) 

# 将日期列转换为日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')

# 2. 准备 Prophet 模型所需的格式
df_prophet = df.rename(columns={'Date': 'ds', 'Ev Stock': 'y'})

# 3. 训练 Prophet 模型
model = Prophet(seasonality_mode='multiplicative')
model.fit(df_prophet)

# 4. 创建未来36个月的日期框架 (多预测3年)
future = model.make_future_dataframe(periods=36, freq='M') 

# 5. 进行预测
forecast = model.predict(future)

# --- 绘图部分 ---

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] #font.sans-serif参数来指定"SimHei"字体
matplotlib.rcParams['axes.unicode_minus'] = False	#axes.unicode_minus参数用于显示负号

# 6. 绘制第一张图：整体趋势图 (2022-2028年)
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制历史数据（移除点标记）
ax.plot(df_prophet['ds'], df_prophet['y'], 'k-', label='历史新能源车保有量')

# 绘制 Prophet 的预测趋势 (橙色虚线)
ax.plot(forecast['ds'], forecast['yhat'], color='darkorange', linestyle='--', label='预测新能源车保有量')

# 绘制 Prophet 的置信区间 (蓝色区域)
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.4, label='预测置信区间')

# 设置纵坐标格式：使用普通数字格式，不带千位分隔符，避免字体问题
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.get_major_formatter().set_scientific(False)

# 设置横坐标格式：以月为单位显示
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12)) # 调整x轴刻度数量
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.title('上海市新能源汽车保有量预测 (2022-2028)', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('新能源汽车保有量 (辆)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='-', alpha=0.6)
plt.tight_layout()
plt.show()

# 7. 绘制第二张图：预测细节图 (2025-2028年，每半年标注数据)
fig, ax = plt.subplots(figsize=(12, 8))

# 筛选出预测数据（从历史数据最后一个点开始）
forecast_future = forecast[forecast['ds'] >= df_prophet['ds'].max()]

# 绘制历史数据（最后一个点）
ax.plot(df_prophet['ds'].iloc[-1:], df_prophet['y'].iloc[-1:], 'k.', label='历史数据末尾')

# 绘制预测趋势线和置信区间
ax.plot(forecast_future['ds'], forecast_future['yhat'], color='darkorange', linestyle='--', label='预测新能源车保有量')
ax.fill_between(forecast_future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], color='skyblue', alpha=0.4)

# 在预测数据上每隔6个月添加数据标签
for i in range(0, len(forecast_future), 6):
    row = forecast_future.iloc[i]
    ax.text(row['ds'], row['yhat'], f'{int(row["yhat"]):,}', ha='center', va='bottom', fontsize=9, color='darkorange')

# 设置纵坐标格式
ax.ticklabel_format(style='plain', axis='y')
ax.yaxis.get_major_formatter().set_scientific(False)

# 设置横坐标格式：以月为单位显示
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=12))
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# 手动设置x轴范围，只显示2025-2028年
ax.set_xlim(pd.to_datetime('2025-01-01'), pd.to_datetime('2029-01-01'))

plt.title('上海市新能源汽车保有量预测 (2025-2028)', fontsize=16)
plt.xlabel('日期', fontsize=12)
plt.ylabel('新能源汽车保有量 (辆)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='-', alpha=0.6)
plt.tight_layout()
plt.show()

# 8. 打印预测结果
print("预测的未来新能源汽车保有量：")
for index, row in forecast.tail(36).iterrows():
    print(f"{row['ds'].strftime('%Y-%m')}: {int(row['yhat'])} 辆")
