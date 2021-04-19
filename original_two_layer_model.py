import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pprint



def generateColNames(var, split_num):
    res = []
    for i in range(split_num):
        res.append(var + '_' + str(i+1))
    return res

def run(subscales, var_split_list):
    original_data = pd.read_csv('onehot/ALL_IN_ONE.csv')
    train_X, test_X, train_y, test_y = train_test_split(original_data.drop(['RiskPerformance'], axis=1),
                                                        original_data['RiskPerformance'],
                                                        test_size=0.3,
                                                        random_state=666)
    coef = {}
    subscale_prob_train = pd.DataFrame()
    subscales_lr = {} # key:subscale_name  value:该subscale的逻辑回归
    # 对每个subscale当做一个独立的分类器
    for subscale in subscales:
        col = []
        for var in subscales[subscale]:
            col.extend(generateColNames(var, len(var_split_list[var])))

        train_X_temp = train_X[col]
        train_y_temp = train_y
        lr = LogisticRegression(max_iter=10000)
        lr.fit(train_X_temp, train_y_temp)

        # print(subscale, ':', train_X_temp.shape[1])
        subscales_lr[subscale] = lr

        # print(lr.predict_proba(train_X_temp)[:, [0]])

        if subscale_prob_train.empty:
            subscale_prob_train = pd.DataFrame(data = lr.predict_proba(train_X_temp)[:, [0]].flatten(), columns=[subscale])
        else:
            temp_dataframe = pd.DataFrame(data = lr.predict_proba(train_X_temp)[:, [0]].flatten(), columns=[subscale])

            subscale_prob_train = pd.concat([subscale_prob_train, temp_dataframe], axis=1, ignore_index=False)


    subscale_prob_train.insert(0, 'target', train_y.values)


    second_layer_lr = LogisticRegression()
    second_layer_lr.fit(subscale_prob_train.drop(['target'], axis = 1), subscale_prob_train.target)
    subscale_prob_train.to_csv('generate/ori2layer_subscale_prob_train.csv', index=False)
    print('十个subscale的权重与截距', second_layer_lr.coef_, second_layer_lr.intercept_)



    # 测试阶段
    # print(test_X)
    # subscales_lr
    subscale_prob_test = pd.DataFrame()
    for subscale in subscales:
        col = []
        for var in subscales[subscale]:
            col.extend(generateColNames(var, len(var_split_list[var])))

        test_X_temp = test_X[col]

        #print(col)

        #print(subscales_lr[subscale].predict_proba(test_X_temp))
        #print(len(subscales_lr[subscale].coef_[0]), test_X_temp.shape[1])
        #print(subscales_lr[subscale].predict_proba(test_X_temp).shape)
        if subscale_prob_test.empty:
            subscale_prob_test = pd.DataFrame(data=subscales_lr[subscale].predict_proba(test_X_temp)[:, [0]].flatten(),
                                               columns=[subscale])
        else:
            temp_dataframe = pd.DataFrame(data=subscales_lr[subscale].predict_proba(test_X_temp)[:, [0]].flatten(), columns=[subscale])
            subscale_prob_test = pd.concat([subscale_prob_test, temp_dataframe], axis=1, ignore_index=False)

    subscale_prob_test.to_csv('generate/ori2layer_subscale_prob_test.csv', index = False)

    # print(subscale_prob_test)
    pred_y = second_layer_lr.predict(subscale_prob_test)
    pred_y_prob = second_layer_lr.predict_proba(subscale_prob_test)[:, 1]

    return test_y, pred_y, pred_y_prob



