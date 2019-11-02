import numpy as np


# divide data into three groups
def divide_data(tx, y):
    """divide data according to jet num"""

    grp_1_tx = tx[tx[:, 22] >= 2]
    def_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29]
    grp_1_tx = grp_1_tx[:, def_1]

    grp_2_tx = tx[tx[:, 22] == 1]
    def_2 = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29]
    grp_2_tx = grp_2_tx[:, def_2]

    grp_3_tx = tx[tx[:, 22] == 0]
    def_3 = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    grp_3_tx = grp_3_tx[:, def_3]

    grp_1_y = y[tx[:, 22] >= 2]
    grp_2_y = y[tx[:, 22] == 1]
    grp_3_y = y[tx[:, 22] == 0]

    return grp_1_tx, grp_2_tx, grp_3_tx, grp_1_y, grp_2_y, grp_3_y, def_1, def_2, def_3

# normalize data
def normalize(tX):
    """normalize data"""
    for i in range(tX.shape[1]):
        if (tX[:,i] == 1).all(): #it's not necessary to normalize the bias term
            continue
        tX[:, i] = (tX[:, i] - min(tX[:, i])) / (max(tX[:, i] - min(tX[:, i])))

    return tX

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_tx_poly(grp_tX, degree):
    """Polynomial for each data group, degree(max polynomial number)"""
    if grp_tX.shape[1] == 29:
        grp_tX_poly_der = build_poly(grp_tX[:,:13], degree)
        grp_tX_poly_pri = grp_tX[:,13:]
        grp_tX_poly = np.c_[grp_tX_poly_der, grp_tX_poly_pri]
    else:
        grp_tX_poly_der = build_poly(grp_tX[:,:10], degree)
        grp_tX_poly_pri = grp_tX[:,10:]
        grp_tX_poly = np.c_[grp_tX_poly_der, grp_tX_poly_pri]
    return grp_tX_poly


# Outlier Cleaning
def clean_outlier(tX):
    '''apply standard deviation method to process outliers  '''

    for i in range(tX.shape[1]):
        # calculate summary statistics
        data = tX[:,i]
        data_mean, data_std = np.mean(data), np.std(data)
        # identify outliers
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off
        outliers_removed = [x for x in data if x >= lower and x <= upper]
        value_rep_lo = np.min(outliers_removed)
        value_rep_up = np.max(outliers_removed)
        idx_outlier_lo = np.where((data < lower))
        idx_outlier_up = np.where((data > upper))
        data[idx_outlier_lo] = value_rep_lo
        data[idx_outlier_up] = value_rep_up
        tX[:,i] = data
    return tX

def process_traindata(tX,y):
    '''Process outliers, augment features, normalize training data'''

    # Max polynomial degree of three groups
    degree1 = 9
    degree2 = 9
    degree3 = 9

    # Replace -999 with median of other data in the column
    tX[tX[:, 0] == -999] = np.median(tX[tX[:, 0] != -999, 0])

    # Divide data in three groups
    grp_1_tx, grp_2_tx, grp_3_tx, grp_1_y, grp_2_y, grp_3_y, def_1, def_2, def_3 = divide_data(tX, y)

    #Outliers cleaning
    grp_1_tx = clean_outlier(grp_1_tx)
    grp_2_tx = clean_outlier(grp_2_tx)
    grp_3_tx = clean_outlier(grp_3_tx)

    #Features augmentation
    grp_1_tx_poly = build_tx_poly(grp_1_tx, degree1)
    grp_2_tx_poly = build_tx_poly(grp_2_tx, degree2)
    grp_3_tx_poly = build_tx_poly(grp_3_tx, degree3)

    #Normalization
    grp_1_tx_poly = normalize(grp_1_tx_poly)
    grp_2_tx_poly = normalize(grp_2_tx_poly)
    grp_3_tx_poly = normalize(grp_3_tx_poly)


    return grp_1_tx_poly,grp_2_tx_poly,grp_3_tx_poly,grp_1_y, grp_2_y, grp_3_y,def_1, def_2, def_3

def process_testdata(tX_test,def_1,def_2,def_3):

    ''' Process test data in the same way as training data '''
    # here the degree must be same as training group
    degree1 = 9
    degree2 = 9
    degree3 = 9

    ind_1 = np.where((tX_test[:, 22] >= 2) == True)
    ind_2 = np.where((tX_test[:, 22] == 1) == True)
    ind_3 = np.where((tX_test[:, 22] == 0) == True)

    # divide test data into three groups
    grp_1_tX_test = tX_test[(tX_test[:, 22] >= 2)]
    grp_1_tX_test = grp_1_tX_test[:, def_1]

    grp_2_tX_test = tX_test[(tX_test[:, 22] == 1)]
    grp_2_tX_test = grp_2_tX_test[:, def_2]

    grp_3_tX_test = tX_test[(tX_test[:, 22] == 0)]
    grp_3_tX_test = grp_3_tX_test[:, def_3]

    #Outliers cleaning
    grp_1_tX_test = clean_outlier(grp_1_tX_test)
    grp_2_tX_test = clean_outlier(grp_2_tX_test)
    grp_3_tX_test = clean_outlier(grp_3_tX_test)

    #Features augmentation
    grp_1_tX_test_poly = build_tx_poly(grp_1_tX_test, degree1)
    grp_2_tX_test_poly = build_tx_poly(grp_2_tX_test, degree2)
    grp_3_tX_test_poly = build_tx_poly(grp_3_tX_test, degree3)

    #Normalization
    grp_1_tX_test_poly = normalize(grp_1_tX_test_poly)
    grp_2_tX_test_poly = normalize(grp_2_tX_test_poly)
    grp_3_tX_test_poly = normalize(grp_3_tX_test_poly)

    return grp_1_tX_test_poly,grp_2_tX_test_poly,grp_3_tX_test_poly,ind_1,ind_2,ind_3
