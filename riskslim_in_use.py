import os
import pprint
import numpy as np
import riskslim
from sklearn.model_selection import train_test_split
import pandas as pd


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

def run(file_path = None, returnSolution = False): # returnSolution为False时直接返回pred_y, pred_y_prob, test_y；为True时返回Solution
    print("===========================RISKSLIM开始============================")
    if file_path == None:
        raise Exception('文件名不能为空')
    # data
    data_name = "breastcancer"                                  # name of the data
    data_dir = os.getcwd() + '/onehot/'                         # directory where datasets are stored
    #data_csv_file = data_dir + data_name + '_data.csv'         # csv file for the dataset
    data_csv_file = file_path
    sample_weights_csv_file = None                              # csv file of sample weights for the dataset (optional)



    # problem parameters
    max_coefficient = 10                                       # value of largest/smallest coefficient
    max_L0_value = len(pd.read_csv(file_path).columns)-1       # 有几个特征就设置为几 maximum model size (set as float(inf))
    max_offset =  50                                           # maximum value of offset parameter (optional)
    c0_value = 1e-6                                            # L0-penalty parameter such that c0_value > 0; larger values -> sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms


    # load data from disk
    data = riskslim.load_data_from_csv(dataset_csv_file = data_csv_file, sample_weights_csv_file = sample_weights_csv_file)
    # 随机划分训练集与训练集 70%训练集 30%数据集
    train_X,test_X,train_y,test_y = train_test_split(data['X'], data['Y'], test_size=0.3, random_state=666)


    # create coefficient set and set the value of the offset parameter
    # 设置变量的上下界，传入customSingleBoundary false代表使用max_coefficient统一上下界，为True则从配置文件中读取，此处传入的lb ub可忽略
    coef_set = riskslim.CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0, customSingleBoundary = False, configPath = "config/ExternalRiskEstimate_lbub.csv")
    # 设置intercept的上下界
    coef_set.update_intercept_bounds(X = train_X, y = train_y, max_offset = max_offset)

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set': coef_set,
    }

    # major settings (see riskslim_ex_02_complete for full set of options)
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        #
        # LCPA Settings
        'max_runtime': 30.0,                               # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'normal',                         # how to compute the loss function ('normal','fast','lookup')
        #
        # LCPA Improvements
        'round_flag': True,                                # round continuous solutions with SeqRd
        'polish_flag': True,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': True,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        #
        # Initialization
        'initialization_flag': True,                       # use initialization procedure
        'init_max_runtime': 120.0,                         # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,
        #
        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = riskslim.run_lattice_cpa(data, constraints, settings)

    #print model contains model
    riskslim.print_model(model_info['solution'], data, show_omitted_variables=True)

    #model info contains key results
    pprint.pprint(model_info)

    if returnSolution:
        return model_info['solution']

    # 训练完毕，测试数据集
    # print(test_X.shape, model_info['solution'].shape)
    # print('test_X：', test_X)
    # print('model_info["solution"]：', model_info['solution'])
    pred_score = np.dot(test_X, model_info['solution'])

    pred_y_prob = sigmoid(pred_score)
    pred_y = np.zeros(pred_y_prob.shape, dtype = np.int)
    pred_y[pred_y_prob < 0.5] = -1
    pred_y[pred_y_prob >= 0.5] = 1

    # test_y = np.array(test_y).reshape(test_y.shape[])

    print("===========================RiskSlim结束============================")


    if not returnSolution:
        return pred_y, pred_y_prob, test_y

