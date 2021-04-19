import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 第一层LR 第二层RiskSLIM

def run(subscales, var_split_list):
    # print(subscales, var_split_list)
    # 先对分组后的10个one-hot训练LR 用字典保存训练的分类器
    sub_lrs = {}
    sub_test_data = {}
    sub_train_data = {}
    for sub_name in subscales:
        df = pd.read_csv('onehot/' + sub_name + '.csv')
        X = df.drop(['RiskPerformance'], axis = 1)
        y = df['RiskPerformance']
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=666)

        lr = LogisticRegression()
        lr.fit(train_X, train_y)

        ### 接续： 将这个的输出拼接成一个 xxx * 10 的panda.dataframe 2021/4/19 22:50
        lr.predict_proba(train_X)

        sub_train_data[sub_name] = [train_X, train_y]
        sub_test_data[sub_name] = [test_X, test_y]
        break

    # for key in sub_test_data:
    #     print(key, ' ', sub_test_data[key], '\n')


