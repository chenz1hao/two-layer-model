import pandas as pd
from xml.dom.minidom import parse
import pprint
import riskslim_in_use
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import original_two_layer_model
import two_layer_riskslim
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import time



############################################
###双层模型+RISKSLIM+逻辑回归在HELOC数据集上对比##
################CHENZIHAO###################
############################################


# AUC图像绘制
def AUC_plot(algorithmName, test_y, pred_y_prob):
    # print(algorithmName, "AUC图像绘制：")
    fpr, tpr, thresholds = roc_curve(test_y, pred_y_prob)
    auc = roc_auc_score(test_y, pred_y_prob)
    plt.plot(fpr, tpr)
    plt.title(algorithmName+" AUC=%.4f" % (auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.fill_between(fpr, tpr, where=(tpr > 0), color='green', alpha=0.5)
    plt.show()


# 输出打印算法性能
def printPerformance(algorithm_name, test_y, pred_y, pred_y_prob):
    # TP(True Positive) 预测正确的1
    # FN(False Negative) 预测为-1，真实为1
    # FP(False Positive) 预测为1，真实为-1
    # TN（True Negative) 预测为-1，真实为-1

    TP = []
    FN = []
    FP = []
    TN = []

    for i in range(len(pred_y)):
        if pred_y[i] == 1 and test_y[i] == 1:
            TP.append(i)
        elif pred_y[i] == -1 and test_y[i] == 1:
            FN.append(i)
        elif pred_y[i] == 1 and test_y[i] == -1:
            FP.append(i)
        elif pred_y[i] == -1 and test_y[i] == -1:
            TN.append(i)

    accuracy = (len(TP)+len(TN))/(len(TP)+len(FP)+len(TN)+len(FN))
    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))
    F1_score = 2 * ((precision*recall)/(precision+recall))
    print(algorithm_name, '：')
    print('Accuracy：', accuracy)
    print('Precision：', precision)
    print('Recall：', recall)
    print('F1-SCORE：', F1_score)
    AUC_plot(algorithm_name, test_y, pred_y_prob)
    print('\n')


# 生成表头 例如var=xx split_num=3 则返回 ['xx_1', 'xx_2', 'xx_3']，在生成one-hot表时用
def generateColNames(var, split_num):
    res = []
    for i in range(split_num):
        res.append(var + '_' + str(i+1))
    return res


# 判断当前值处于第几个区间中,在生成one-hot表时用
def checkWhichDivision(split_list, var):
    for i in range(len(split_list)):
        begin_index = split_list[i].index('(')
        mid_index = split_list[i].index(',')
        if('-INF' in split_list[i]): # 含有-INF说明是第一个区间，则直接取'逗号'开始至']'结束
            end_index = split_list[i].index(']')
            # print(split_list[i][mid_index+1:end_index])
            if (var <= float(split_list[i][mid_index+1 : end_index])):
                return i+1
        elif('+INF' in split_list[i]): # 含有+INF说明是最后一个区间，则直接取第一个括号开始至','结束
            # print(split_list[i][begin_index+1:mid_index])
            if (var > float(split_list[i][begin_index+1:mid_index])):
                return i+1
        else:   # 处于中间区间的，取两个数，第一个数是'('开始','结束，第二个数是','开始']'结束
            end_index = split_list[i].index(']')
            if float(split_list[i][mid_index+1:end_index]) >= var > float(split_list[i][begin_index+1: mid_index]):
                return i+1


# 生成one-hot某一行的字典格式数据，用于追加到dataframe中，在生成one-hot表时用
def generateCurRowInDict(col_names, whichDivision):
    res = {}
    hasSetOne = False

    for col_name in col_names:
        if hasSetOne:
            res[col_name] = 0
        else:
            if str(whichDivision) in col_name:
                res[col_name] = 1
                hasSetOne = True
            else:
                res[col_name] = 0

    return res


# 根据传入的subscale的变量list来生成one-hot文件
def generateOneHotByList(list, var_split_list, generatePath):
    #  print(list, generatePath)
    # 取出原数据集中对应列
    data = pd.read_csv('data/heloc_dataset_v2.csv')
    # print(data[list])
    try:
        partial_data = data[list] # 只包含在了该subscale中的变量的部分data视图
    except KeyError:
        raise Exception('配置文件中的变量不能在原数据集中找到，检查subscale.xml中的变量')


    dataframe_final = pd.DataFrame()

    for var in list:
        if var in var_split_list.keys(): # 确定是否要分箱
            cur_split_list = var_split_list[var]
            split_num = len(cur_split_list)
            col_names = generateColNames(var, split_num)
            dataframe_temp = pd.DataFrame(columns = col_names) # 有几个分段就要创建一个几列的dataframe
            cur_col = partial_data[var]
            # 这个for循环一行一行插入dataframe_temp
            for index in cur_col.index:
                whichDivision = checkWhichDivision(cur_split_list, cur_col[index]) # 判断当前这个值在哪个区间里面
                temp_row = generateCurRowInDict(col_names, whichDivision) # 根据所处区间生成一个字典数据，如："{'ExternalRiskEstimate_1': 0, 'ExternalRiskEstimate_2': 0, 'ExternalRiskEstimate_3': 1}"用于之后的追加
                # print(temp_row)
                dataframe_temp = dataframe_temp.append(temp_row, ignore_index = True)
                # print(cur_split_list, cur_col[index], whichDivision, temp_row)
            dataframe_temp.columns = col_names

            # 合并在dataframe_final中
            if dataframe_final.empty:
                dataframe_final = dataframe_temp
            else:
                dataframe_final = pd.concat([dataframe_final, dataframe_temp], axis = 1, ignore_index = False)

    # print(dataframe_final)
    dataframe_final.to_csv(generatePath, index=0)
    print('生成', generatePath, '成功')


if __name__ == "__main__":

    p0 = time.process_time()

    ### 数据预处理 BEGIN

    split_file = pd.read_csv('data/features_detail.csv') # 这个文件的split_list就是分箱的结果
    original_data = pd.read_csv('data/heloc_dataset_v2.csv') # v2在heloc_dataset_v1.csv的基础上将预测目标列名改为了target，且Bad->-1，Good->1
    original_data_onehot = pd.read_csv('onehot/ALL_IN_ONE.csv')
    # 测试修改

    var_start_with = {}                 # 每个变量所在区间的起始下标
    var_all = []                        # 所有变量 = var_to_be_bin + var_not_to_be_bin
    var_to_be_bin = []                  # 分了箱的变量数组
    var_not_to_be_bin = []              # 未分箱的变量数组
    var_split_list = {}                 # 分箱变量的分箱结果，从features_detail.csv中读取的
    subscales = {}                      # 从配置文件中读取的subscale分组情况，以分组名为key，组中成员变量名为values
    subscales_name = []                 # subscale分组名的记录数组
    NEED_GENERATE_ONE_HOT_CSV = False   # 是否生成one-hot文件，只需要生成一次，不用每次运行反复生成，设置为False前需要确认是否真的存在了对应subscale个csv
    RUN_RISKSLIM = False                # 本次测试是否运行RiskSlim
    RUN_TWOLAYER = False                # 本次测试是否运行Twolayer
    RUN_LOGISTICREG = False             # 本次测试是否运行逻辑回归
    RUN_RISKSLIM_WITH_TWOLAYER = False  # 本次测试是否运行双层RISKSLIM模型


    # 找出每个变量所在文件中区间的起始下标，顺便记录分了箱的那些变量数组
    for i in range(split_file.shape[0]):
        if (i == 0)  or ((i != 0) and (split_file.var_name[i] != split_file.var_name[i-1])) :
            var_start_with[split_file.var_name[i]] = i
            var_to_be_bin.append(split_file.var_name[i])


    # 记录分箱变量的分箱结果
    temp = []
    for i in range(split_file.shape[0]):
        if i == split_file.shape[0] - 1:   # 最后一个变量时
            temp.append(split_file.split_list[i])
            var_split_list[split_file.var_name[i]] = temp
        elif i == 0 or split_file.var_name[i] == split_file.var_name[i - 1]:
            temp.append(split_file.split_list[i])
        elif split_file.var_name[i] != split_file.var_name[i - 1]:
            var_split_list[split_file.var_name[i-1]] = temp
            temp = []
            temp.append(split_file.split_list[i])


    # 从配置文件中读取subscale信息
    config_path = "config/subscale.xml"
    rootNode = parse(config_path).documentElement
    count = 0
    if rootNode.nodeName == "root":
        subs = rootNode.getElementsByTagName("subscale")
        for sub in subs:
            temp_sub = []
            vars = sub.getElementsByTagName("var")
            for var in vars:
                temp_sub.append(var.childNodes[0].data)
                var_all.append(var.childNodes[0].data)
            subscales[sub.getAttribute("name")] = temp_sub

    var_not_to_be_bin = list(set(var_all).difference(var_to_be_bin))

    ### 数据预处理 END

    # 打印一些summary
    print("============================================================================================================")
    print('共有', len(var_all), '个变量')
    print('分箱变量有', len(var_to_be_bin), '个，不分箱的有', len(var_not_to_be_bin), '个')
    print('其中分箱变量的分箱信息：')
    pprint.pprint(var_split_list)
    print('从配置文件:', config_path, '中读取的subscale信息如下：')
    pprint.pprint(subscales)
    # for key in subscales:
    #     pprint_pprint(key, " ", subscales[key])
    #     subscales_name.append(key)
    print("============================================================================================================")

    # 生成了一次one-hot后不用每次运行都生成一次。
    if NEED_GENERATE_ONE_HOT_CSV:
        # 按照subscale生成one-hot的csv文件，有多少个subscale就需要生成多少个文件
        for key in subscales:
            generateOneHotByList(subscales[key], var_split_list, 'onehot/'+key +'.csv')  # 根据subscale生成one-hot编码文件

        # 给每个one-hot首列加入Labels列：RiskPerformance
        for key in subscales:
            temp_csv = pd.read_csv('onehot/' + key + '.csv')
            # print(temp_csv)
            temp_csv.insert(0, 'RiskPerformance', original_data['target'])
            temp_csv.to_csv('onehot/' + key + '.csv', index=0)
        print('====================================所有subscale的one-hot编码文件生成成功===============================')
    else:
        print('---直接使用了onehot文件夹中的数据来运行算法')


    ### 算法运行 BEGIN

    if RUN_RISKSLIM:
        pred_y_riskslim, pred_y_prob_riskslim, test_y_riskslim = riskslim_in_use.run('onehot/ALL_IN_ONE.csv')
        printPerformance('RiskSlim', test_y_riskslim, pred_y_riskslim, pred_y_prob_riskslim)
    else:
        print('---不运行RiskSLIM')


    if RUN_LOGISTICREG:
        # 保证算法比较时用的训练集和数据集相同
        train_X_lr,test_X_lr,train_y_lr,test_y_lr = train_test_split(original_data_onehot.drop(['RiskPerformance'], axis=1), original_data_onehot['RiskPerformance'], test_size=0.3, random_state=666)
        test_y_lr = np.array(test_y_lr)
        lr = LogisticRegression(max_iter=1000000)
        lr.fit(train_X_lr, train_y_lr)
        pred_y_lr = lr.predict(test_X_lr)
        pred_y_prob_lr = lr.predict_proba(test_X_lr)[:, 1]
        printPerformance('LogisticReg', np.array(test_y_lr), pred_y_lr, pred_y_prob_lr)
    else:
        print('---不运行逻辑回归')


    if RUN_TWOLAYER:
        test_y_twolayer, pred_y_twolayer, pred_y_prob_twolayer = original_two_layer_model.run(subscales, var_split_list)
        printPerformance('Two-layer-model', np.array(test_y_twolayer), pred_y_twolayer, pred_y_prob_twolayer)
    else:
        print('---不运行双层模型')


    if RUN_RISKSLIM_WITH_TWOLAYER:
        test_y_twolayer_riskslim, pred_y_twolayer_riskslim, pred_y_prob_twolayer_riskslim = two_layer_riskslim.run(subscales, var_split_list)
        printPerformance('RiskSlim & Two-layer', test_y_twolayer_riskslim, pred_y_twolayer_riskslim, pred_y_prob_twolayer_riskslim)
    else:
        print('---不运行riskslim&twolayer')

    ### 算法运行 END

    p1 = time.process_time()
    print('运行时间: %s 秒' % (p1-p0))



