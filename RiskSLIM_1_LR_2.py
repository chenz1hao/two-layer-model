import pandas as pd
from sklearn.model_selection import train_test_split
import riskslim_in_use
import numpy as np
from sklearn.linear_model import LogisticRegression

# 第一层RiskSLIM 第二层逻辑回归

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def generateColNames(var, split_num):
    res = []
    for i in range(split_num):
        res.append(var + '_' + str(i+1))
    return res


# 双层模型 + RiskSlim
def run(subscales, var_split_list):
    NEED_GENERATE_CSV = False # 设置为False不用每次运行都要生成一遍
    original_data_onehot = pd.read_csv('onehot/ALL_IN_ONE.csv')
    train_X, test_X, train_y, test_y = train_test_split(original_data_onehot.drop(['RiskPerformance'], axis=1),
                                                        original_data_onehot['RiskPerformance'], test_size=0.3,
                                                        random_state=666)
    if NEED_GENERATE_CSV:
        sub_solutions = pd.DataFrame()  # 所有subscale的solutions
        sub_prob = pd.DataFrame()
        count = 0

        for sub in subscales:
            sub_data = pd.read_csv('onehot/' + sub + '.csv')
            train_X_sub, test_X_sub, train_y_sub, test_y_sub = train_test_split(sub_data.drop(['RiskPerformance'], axis=1),
                             sub_data['RiskPerformance'],
                             test_size=0.3,
                             random_state=666)

            solution = riskslim_in_use.run('onehot/' + sub + '.csv', returnSolution=True)

            part_solution = np.array(solution)[1:]

            # print(part_solution.shape, train_X_sub.shape)


            temp_result = np.dot(train_X_sub, part_solution)
            temp_result = sigmoid(temp_result)
            temp_result = pd.DataFrame(temp_result, columns=[sub])

            temp_solutions = pd.DataFrame(solution, columns=[sub])

            if sub_solutions.empty:
                sub_solutions = temp_solutions
            else:
                sub_solutions = pd.concat([sub_solutions, temp_solutions], axis = 1, ignore_index = False)


            if sub_prob.empty:
                sub_prob = temp_result
            else:
                sub_prob = pd.concat([sub_prob, temp_result], axis = 1, ignore_index = False)

            print('SOLUTION:', solution)
            count = count + 1

        print(count, '次RiskSlim算法调用')
        sub_prob.insert(0, 'target', train_y.values)
        sub_prob.to_csv('generate/riskslim&2layer_subscale_prob_train.csv', index=False)
        sub_solutions.to_csv('generate/riskslim&2layer_subscale_solution.csv', index=False)



    two_layer_lr = LogisticRegression()

    if not NEED_GENERATE_CSV:
        sub_prob = pd.read_csv('generate/riskslim&2layer_subscale_prob_train.csv')
        sub_solutions = pd.read_csv('generate/riskslim&2layer_subscale_solution.csv')

    two_layer_lr.fit(sub_prob.drop(['target'], axis=1), sub_prob['target'])
    # print(lr.coef_) 第二层逻辑回归的系数
    # print(sub_solutions) 第一层每个subscale的变量得分score



    # 训练完毕，测试阶段

    subscale_score_test = pd.DataFrame()
    for subscale in subscales:
        col = []
        for var in subscales[subscale]:
            col.extend(generateColNames(var, len(var_split_list[var])))

        test_X_temp = test_X[col]

        # print('TEST_X_TEMP.SHAPE:', test_X_temp.shape)
        temp_solution = np.array(sub_solutions[subscale][:test_X_temp.shape[1]+1])
        intercept = temp_solution[0]
        coef = temp_solution[1:]
        # print(intercept, coef)
        cur_sub_data = pd.read_csv('onehot/' + subscale + '.csv')
        train_cur_sub_X, test_cur_sub_X, train_cur_sub_y, test_cur_sub_y = train_test_split(cur_sub_data.drop(['RiskPerformance'], axis=1),
                                                                            cur_sub_data['RiskPerformance'],
                                                                            test_size=0.3,
                                                                            random_state=666)


        # print(cur_sub_data)
        # print(coef.shape)
        # print(cur_sub_data.shape)
        temp_result = np.dot(test_cur_sub_X, coef) + intercept
        # print(temp_result, intercept)


        if subscale_score_test.empty:
            subscale_score_test = pd.DataFrame(data=temp_result,
                                              columns=[subscale])
        else:
            temp_dataframe = pd.DataFrame(data=temp_result,
                                          columns=[subscale])
            subscale_score_test = pd.concat([subscale_score_test, temp_dataframe], axis=1, ignore_index=False)



    subsclae_prob_test = sigmoid(subscale_score_test)
    subsclae_prob_test = pd.DataFrame(subsclae_prob_test)
    # subsclae_prob_test.insert(0, 'RiskPerformance', test_y.values)
    subsclae_prob_test.to_csv('generate/riskslim&2layer_subscale_prob_test.csv', index=False)

    pred_y = two_layer_lr.predict(subsclae_prob_test)
    pred_y_prob = two_layer_lr.predict_proba(subsclae_prob_test)[:, 1]
    #print(pred_y, pred_y_prob)
    #print(two_layer_lr.classes_)

    return test_y.values, pred_y, pred_y_prob





