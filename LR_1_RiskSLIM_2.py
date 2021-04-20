import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import riskslim_in_use

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

# 第一层LR 第二层RiskSLIM
def run(subscales, var_split_list):
    sub_lrs = {}
    sub_test_data = {}
    sub_train_data = {}
    subscale_prob_train = pd.DataFrame()
    subscale_prob_test = pd.DataFrame()

    # 训练阶段
    # 第一层 ，十个LR
    for sub_name in subscales:
        df = pd.read_csv('onehot/' + sub_name + '.csv')
        X = df.drop(['RiskPerformance'], axis = 1)
        y = df['RiskPerformance']
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=666)

        lr = LogisticRegression()
        lr.fit(train_X, train_y)
        sub_lrs[sub_name] = lr

        # lr.predict_proba(train_X)
        if subscale_prob_train.empty:
            subscale_prob_train = pd.DataFrame(data=lr.predict_proba(train_X)[:, 0].flatten(), columns=[sub_name])
        else:
            temp_dataframe = pd.DataFrame(data=lr.predict_proba(train_X)[:, 0].flatten(), columns=[sub_name])
            subscale_prob_train = pd.concat([subscale_prob_train, temp_dataframe], axis=1, ignore_index=False)


        sub_train_data[sub_name] = [train_X, train_y]
        sub_test_data[sub_name] = [test_X, test_y]

    subscale_prob_train.insert(0, 'RiskPerformance', train_y.values)
    subscale_prob_train.to_csv('generate/lr1&riskslim2_prob_train.csv', index = False)

    # 第二层 一个RiskSLIM
    solution = riskslim_in_use.run(file_path = 'generate/lr1&riskslim2_prob_train.csv', returnSolution = True)
    riskslim_intercept = solution[0]
    riskslim_coef = solution[1:]



    # 测试阶段
    # 第一层
    for sub_name in subscales:
        df = pd.read_csv('onehot/' + sub_name + '.csv')
        X = df.drop(['RiskPerformance'], axis=1)
        y = df['RiskPerformance']
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=666)

        if subscale_prob_test.empty:
            subscale_prob_test = pd.DataFrame(data=sub_lrs[sub_name].predict_proba(test_X)[:, 0].flatten(), columns=[sub_name])
        else:
            temp_dataframe = pd.DataFrame(data=sub_lrs[sub_name].predict_proba(test_X)[:, 0].flatten(), columns=[sub_name])
            subscale_prob_test = pd.concat([subscale_prob_test, temp_dataframe], axis=1, ignore_index=False)

    # subscale_prob_test.insert(0, 'RiskPerformance', test_y.values)
    subscale_prob_test.to_csv('generate/lr1&riskslim2_prob_test.csv', index = False)

    subscale_prob_test = np.array(subscale_prob_test)

    # 第二层
    pred_y_prob = sigmoid(np.dot(subscale_prob_test, riskslim_coef) + riskslim_intercept)
    pred_y = np.zeros(pred_y_prob.shape, dtype = np.int)
    pred_y[pred_y_prob < 0.5] = -1
    pred_y[pred_y_prob >= 0.5] = 1


    return test_y.values, pred_y, pred_y_prob



