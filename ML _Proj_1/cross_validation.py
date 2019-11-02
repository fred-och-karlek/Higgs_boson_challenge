
from proj1_helpers import *


def split_data(x, y, k):
    """split the dataset based on the split ratio."""
    # generate random indices
    data_size = len(y)
    indices = np.random.permutation(data_size)
    cros_val_set = []
    label_set = []

    for i in range(k):
        # ratio = i/k
        fold = x[indices[int(i / k * data_size):int((i + 1) / k * data_size)]]
        label = y[indices[int(i / k * data_size):int((i + 1) / k * data_size)]]
        cros_val_set.append(fold)
        label_set.append(label)

    return cros_val_set, label_set

def build_combinations(k):
    """ build k groups data """

    folds_id = set()
    leave_one_out = set()
    combinations = []
    for i in range(k):
        folds_id.add(i)
    for i in range(k):
        leave_one_out.add(i)
        combinations.append(folds_id.difference(leave_one_out))
        leave_one_out = set()

    return combinations,folds_id



def validate(cros_val_set, label_set, combinations, folds_id):
    """
        this function is used to test different parameters and different algorithm

        return

        loss:  the number of wrong prediction
        acc:  accuracy
        los:  this los is used to check the convergence especially for gradient descent method

    """
    loss = 0;
    count = 0;
    num = 0;
    for combination in combinations:
        tr = []
        tr_l = []
        te_id = list(folds_id.difference(combination))[0]
        te = cros_val_set[te_id]
        te_l = label_set[te_id]
        for fold_id in combination:
            tr.append(cros_val_set[fold_id])
            tr_l.append(label_set[fold_id])
        tr = np.vstack(tr)
        tr_l = np.hstack(tr_l)
        # w, los = least_squares_SGD(tr_l, tr, initial_w, 1800, 0.03)
        # w, los = least_squares_GD(tr_l, tr, initial_w, 500, 0.03)
        # w, los = least_squares(tr_l, tr)
        # w, los = ridge_regression(tr_l, tr, 0.00000000001)
        # w, los = logistic_regression(tr_l, tr, np.zeros(grp_2_tx_poly.shape[1]), 10000, 0.1)
        w, los = reg_logistic_regression(tr_l, tr, 0.000000001, np.zeros(grp_3_tx_poly.shape[1]), 10000, 0.02)

        pred_l = predict_labels(w, te)
        # only for logistic regression
        pred_l[pred_l == -1] = 0

        loss += len(te_l) - sum(te_l == pred_l)
        count += sum(te_l == pred_l)

        num += len(te_l)
    loss = loss / len(folds_id)
    acc = count / num
    return loss, acc, los