# # Useful starting lines
# %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

from proj1_helpers import *
from implementations import  *
from data_cleaning import *
import numpy as np

#you might need to create a directory called data to store the train and test data

#load train data
DATA_TRAIN_PATH = 'data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


# Clean outliers, augment features, normalize data
grp_1_tx_poly,grp_2_tx_poly,grp_3_tx_poly,grp_1_y, grp_2_y, grp_3_y,def_1, def_2, def_3 = process_traindata(tX,y)

# Use ridge rigression for data group 1 and data group 2
lambda_1 = 0.0000000000001
w1,loss1 = ridge_regression(grp_1_y, grp_1_tx_poly, lambda_1)
w2,loss2 = ridge_regression(grp_2_y, grp_2_tx_poly, lambda_1)

# Use regularized logistic regression for data group 3
lambda_2 = 0.000000001
max_iters = 8000
gamma = 0.015
grp_3_y[grp_3_y==-1] = 0 # Only for logistic regression we need change the value of labels
w3,loss3 = reg_logistic_regression(grp_3_y, grp_3_tx_poly, lambda_2 ,np.zeros(grp_3_tx_poly.shape[1]), max_iters, gamma)

print('training done!')



#load test data

DATA_TEST_PATH = 'data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Process first data column: replace -999 with median of other data in the column
tX_test[tX_test[:,0]==-999] = np.median(tX_test[tX_test[:,0]!=-999,0])

# Process test data in the same way as training data
grp_1_tX_test_poly,grp_2_tX_test_poly,grp_3_tX_test_poly,ind_1,ind_2,ind_3 = process_testdata(tX_test,def_1,def_2,def_3)

# predict the label for each test group
pred_1 = predict_labels(w1, grp_1_tX_test_poly)
pred_2 = predict_labels(w2, grp_2_tX_test_poly)
pred_3 = predict_labels(w3, grp_3_tX_test_poly)

#collect the prediction
y_pred = np.zeros((tX_test.shape[0]))
y_pred[ind_1] = pred_1
y_pred[ind_2] = pred_2
y_pred[ind_3] = pred_3


OUTPUT_PATH = 'prediction/pred01.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

print('all done!')
