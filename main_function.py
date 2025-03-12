import prediction_function

if __name__ == '__main__':
    #只需要调用prediction_function文件包里面的prediction方法就行了，第一个变量是训练数据集，包含了气象数据和虫情数据，第二个变量为预测集，只有cimp6的未来气象数据

    #最基础的应用
    prediction_function.prediction("line/line 1/step1 丘北-锦屏/line1_step1_丘北-锦屏.xlsx", "cmip_Nor _dealed\cmip_Nor _dealed\SSP2\西南丘北ssp245.csv")

    # # 改变第一个变量的名字，查看输出文件的名字
    #prediction_function.prediction("line/line 1/step1 丘北-锦屏/line1_step1_丘北-锦屏(气象数据前移5天).xlsx", "cmip_Nor _dealed\cmip_Nor _dealed\SSP2\西南丘北ssp245.csv")
    #
    # #改变了第一个变量和第二个变量(ssp245变为ssp126)，查看输出文件的名字
    # prediction_function.prediction("line/line 1/step1 丘北-锦屏/line1_step1_丘北-锦屏(气象数据前移10天).xlsx", "cmip_Nor _dealed\cmip_Nor _dealed\SSP1\西南丘北ssp126.csv")