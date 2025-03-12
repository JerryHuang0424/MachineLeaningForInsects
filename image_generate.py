import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel('气象地点+虫情地点数据/Step9_盐都-奉贤/Step9_盐都-奉贤.xlsx')
df_5 = pd.read_excel('气象地点+虫情地点数据/Step9_盐都-奉贤/Step9_盐都-奉贤(气象数据前移5天).xlsx')
df_10 = pd.read_excel('气象地点+虫情地点数据/Step9_盐都-奉贤/Step9_盐都-奉贤(气象数据前移10天).xlsx')
df_15 = pd.read_excel('气象地点+虫情地点数据/Step9_盐都-奉贤/Step9_盐都-奉贤(气象数据前移15天).xlsx')
# df_pred = pd.read_csv('cmip_Nor _dealed\cmip_Nor _dealed\SSP4\江淮奉贤ssp585.csv')


missing_dates = df[df['date'].isna()]
print(missing_dates)
df['date'] = df['date'].apply(lambda x: x.timestamp())
# 查找 NaT 值

missing_dates = df_5[df_5['date'].isna()]
print(missing_dates)
df_5['date'] = df_5['date'].apply(lambda x: x.timestamp())

df_10['date'] = df_10['date'].apply(lambda x: x.timestamp())
missing_dates = df_10[df_10['date'].isna()]
print(missing_dates)
df_15['date'] = df_15['date'].apply(lambda x: x.timestamp())
missing_dates = df_15[df_15['date'].isna()]
print(missing_dates)

df['白背飞虱'] = np.log(df['白背飞虱'] + 1)
df_5['白背飞虱'] = np.log(df_5['白背飞虱'] + 1)
df_10['白背飞虱'] = np.log(df_10['白背飞虱'] + 1)
df_15['白背飞虱'] = np.log(df_15['白背飞虱'] + 1)

df['褐飞虱'] = np.log(df['褐飞虱'] + 1)
df_5['褐飞虱'] = np.log(df_5['褐飞虱'] + 1)
df_10['褐飞虱'] = np.log(df_10['褐飞虱'] + 1)
df_15['褐飞虱'] = np.log(df_15['褐飞虱'] + 1)


print(df.columns)


# 相关性矩阵
correlation_matrix = df.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
# 指定保存图像的文件夹和文件名
# save_folder = 'D:\JerryHuang\活动\创新创业\利用机器学习解决稻飞虱问题\project\新方向\气象地点+虫情地点数据\Step9_盐都-奉贤\SSP585'  # 替换为你的文件夹路径
# save_filename = 'SSP585-盐都-奉贤相关性热力图.png'  # 替换为你想要的文件名
#
# # 保存图像
# plt.savefig(f'{save_folder}/{save_filename}', bbox_inches='tight')  # bbox_inches='tight' 用于去除边缘空白
#
plt.show()

correlation_matrix1 = df_5.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix1.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
# 指定保存图像的文件夹和文件名
# save_folder = 'D:\JerryHuang\活动\创新创业\利用机器学习解决稻飞虱问题\project\新方向\气象地点+虫情地点数据\Step9_盐都-奉贤\SSP585'  # 替换为你的文件夹路径
# save_filename = 'SSP585-盐都-奉贤相关性热力图(气象数据前5天).png'  # 替换为你想要的文件名
#
# # 保存图像
# plt.savefig(f'{save_folder}/{save_filename}', bbox_inches='tight')  # bbox_inches='tight' 用于去除边缘空白
#
plt.show()


correlation_matrix2 = df_10.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix2.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
# # 指定保存图像的文件夹和文件名
# save_folder = 'D:\JerryHuang\活动\创新创业\利用机器学习解决稻飞虱问题\project\新方向\气象地点+虫情地点数据\Step9_盐都-奉贤\SSP585'  # 替换为你的文件夹路径
# save_filename = 'SSP585-盐都-奉贤相关性热力图(气象数据前10天).png'  # 替换为你想要的文件名
#
# # 保存图像
# plt.savefig(f'{save_folder}/{save_filename}', bbox_inches='tight')  # bbox_inches='tight' 用于去除边缘空白
#
plt.show()


correlation_matrix3 = df_15.corr()
# 绘制热力图，查看各特征与目标变量的相关性
selected_rows_cols = correlation_matrix3.loc[['date', 'year', 'month', 'hou', '1000hPa_air', '850hPa_air', '1000hPa_rhum', '850hPa_rhum', '1000hPa_wind', '850hPa_wind', '1000hPa_azimuth', '850hPa_azimuth', '1000hPa_number', '850hPa_number', '850hPa_omega'], ['白背飞虱', '褐飞虱']]
plt.figure(figsize=(20, 12))
sns.heatmap(selected_rows_cols, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
# 指定保存图像的文件夹和文件名
# save_folder = 'D:\JerryHuang\活动\创新创业\利用机器学习解决稻飞虱问题\project\新方向\气象地点+虫情地点数据\Step9_盐都-奉贤\SSP585'  # 替换为你的文件夹路径
# save_filename = 'SSP585-盐都-奉贤相关性热力图(气象数据前15天).png'  # 替换为你想要的文件名
#
# # 保存图像
# plt.savefig(f'{save_folder}/{save_filename}', bbox_inches='tight')  # bbox_inches='tight' 用于去除边缘空白
#
plt.show()