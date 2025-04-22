import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def clean_data(df_pred):
    missing_values = df_pred.isnull().sum()
    print("loss value: \n", missing_values)
    df_pred.fillna(df_pred.mean(), inplace=True)  # 用平均值来替代缺失值
    duplicates = df_pred.duplicated().sum()
    print(f"重复数据行数: {duplicates}")
    # 如果有重复行，删除它们
    df_pred.drop_duplicates(inplace=True)

    return df_pred

def deal_data(df):
    df['date'] = df['date'].apply(lambda x: x.timestamp())
    missing_dates = df[df['date'].isna()]
    print(missing_dates)

    df['白背飞虱'] = np.log(df['白背飞虱'] + 1)
    df['褐飞虱'] = np.log(df['褐飞虱'] + 1)

    return df

def choose_feature(df_deal, target):

    #绘制热力图，计算相关性
    correlation_matrix = df_deal.corr()

    # 选择和目标变量"白背飞虱"和"褐飞虱"相关性较强的特征
    cor_target = correlation_matrix[target].sort_values(ascending=False)
    print(cor_target)

    # 选择和目标变量相关性较高的特征
    # 可以选择相关性高于某个阈值的特征，假设阈值为0.1
    # 直接在筛选相关性绝对值大于0.1的特征索引时，排除指定的特征
    selected_features = cor_target[(cor_target > 0.1) & (~cor_target.index.isin(['白背飞虱', '褐飞虱']))].index.tolist()
    print("Selected features:", selected_features)


    return selected_features

def train_model(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    shape = X_train.shape[1]
    print(shape)
    mes = []

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    #使用SVM模型进行预测
    from sklearn.svm import SVR  # 支持向量回归
    svm = SVR()
    svm.fit(X_train, y_train)

    y_pred_svm = svm.predict(X_test)
    mes.append(mean_squared_error(y_test, y_pred_svm))
    print("SVM R2:", r2_score(y_test, y_pred_svm))
    print("SVM MSE:", mean_squared_error(y_test, y_pred_svm))
    print('\n')


    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline

    # 创建多项式回归的典型方式
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),  # n为多项式阶数
        ('linear', LinearRegression())
    ])
    poly_reg.fit(X_train, y_train)

    # 预测并评估
    y_pred_poly = poly_reg.predict(X_test)
    mes.append(mean_squared_error(y_test, y_pred_poly))
    print("poly R2:", r2_score(y_test, y_pred_poly))
    print("poly MSE:", mean_squared_error(y_test, y_pred_poly))
    print('\n')

    #使用随机森林模型
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    # 预测并评估
    y_pred_rf = rf.predict(X_test)
    mes.append(mean_squared_error(y_test, y_pred_rf))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))
    print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
    print('\n')

    if mes[0] < mes[1] and mes[0] < mes[2]:

        print("Use random forest model")
        return svm
    elif mes[1] < mes[2] and mes[1] < mes[0]:
        print("Use poly model")
        return poly_reg
    elif mes[2] < mes[0] and mes[2] < mes[1]:
        print("Use linear return model")
        return rf


def stringDeal(train_file_name, pred_file_name):

    if len(train_file_name) == 56:
        name1 = train_file_name[-32:-5]
    elif len(train_file_name) == 57:
        name1 = train_file_name[-33:-5]
    else:
        name1 = train_file_name[-22:-5]

    name2 = pred_file_name[-10:-4]

    name = "pred_" + name1 + "_" + name2

    return name

def convert_to_level(predicted_values):
    # 使用 np.select 实现条件判断
    conditions = [
        (predicted_values >= 0) & (predicted_values < 1),
        (predicted_values >= 1) & (predicted_values < 2),
        (predicted_values >= 2) & (predicted_values < 3),
        (predicted_values >= 3) & (predicted_values < 4),
        (predicted_values >= 4)
    ]
    choices = ["Level1", "Level2", "Level3", "Level4", "Level5"]
    return np.select(conditions, choices, default="Unknown")
def prediction(train_file_name, pred_file_name):
    # 这个方法的参数是传入的文件的名称，文件为历史数据和虫情数据的集合
    # 传入的文件会进行机器学习分析，分别使用随机森林（RF），KNN方法和线性回归
    # 训练好的模型会保存到本地

    df = pd.read_excel(train_file_name, engine='openpyxl')
    df_pred = pd.read_csv(pred_file_name)

    df_pred_clean = clean_data(df_pred)

    df_deal = deal_data(df)

    print("df_deal:\n", df_deal.columns)


    selected_features_white = choose_feature(df_deal,"白背飞虱")
    x_white = df_deal[selected_features_white]
    y_white = df_deal["白背飞虱"]

    selected_features_brown = choose_feature(df_deal,"褐飞虱")
    x_brown = df_deal[selected_features_brown]
    y_brown = df_deal["褐飞虱"]

    name = stringDeal(train_file_name, pred_file_name)

    model_white = train_model(x_white, y_white)
    model_brown = train_model(x_brown, y_brown)


    model_path = train_file_name[:-5]
    dump(model_white, f'modle/{name}_white_model.joblib')
    dump(model_brown, f'modle/{name}_model_brown.joblib')

    x_white_pred = df_pred_clean[selected_features_white]
    x_brown_pred = df_pred_clean[selected_features_brown]

    white_num = model_white.predict(x_white_pred)
    brown_num = model_brown.predict(x_brown_pred)

    white_level = convert_to_level(white_num)
    brown_level = convert_to_level(brown_num)

    df_pred['白背飞虱'] = white_level
    df_pred['褐飞虱'] = brown_level



    name += ".xlsx"

    df_pred.to_excel(name, index=False)


