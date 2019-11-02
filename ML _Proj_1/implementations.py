import numpy as np

def compute_mse_loss(y, tx, w):
    """ return mse loss """
    e = y - np.dot(tx,w)
    return 1/2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = len(y)
    e = y - np.dot(tx,w)
    grad = -(np.dot(tx.T,e))/N
    return grad


def sigmoid(x):
    """sigmoid function """
    return 1.0 / (1 + np.exp(-x))


def compute_lg_loss(y, x, w):
    '''return loss for logistic regression '''
    e = 1e-11
    p = sigmoid(np.dot(x, w))
    loss = -np.dot(y, np.log(p + e)) - np.dot((1 - y).T, np.log(1 - p + e))
    return loss


def compute_lg_grad(y, x, w):
    """Compute the gradient for logistic regression"""
    p = sigmoid(np.dot(x, w))
    grad = np.dot(x.T, (p - y))
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient decent method using least square loss function"""
    losses = []
    w = initial_w
    threshold = 1e-8
    for n_iter in range(max_iters):

        grad = compute_gradient(y, tx, w)
        grad = grad/np.linalg.norm(grad)

        loss = compute_mse_loss(y, tx, w)

        w = w - gamma*grad
        # store loss in order to check the convergence
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1]-losses[-2]) < threshold:
            break

    return w,losses[-1]



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Stochastic gradient decent method using least square loss function"""
    losses = []
    w = initial_w
    threshold = 1e-8
    batch_size = 1

    for n_iter in range(max_iters):
        # Randomly select sample and compute stochastic gradient
        random_num = np.random.randint(0, len(y), size=batch_size)
        r_y = y[random_num]
        r_tx = tx[random_num]

        grad = compute_gradient(r_y, r_tx, w)
        grad = grad/np.linalg.norm(grad)

        w = w - gamma*grad

        # store loss in order to check the convergence
        loss = compute_mse_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1]-losses[-2]) < threshold:
            break

    return w,losses[-1]




def least_squares(y, tx):
    """calculate the least squares solution."""
    XTX = np.dot(tx.T,tx)
    XTY = np.dot(tx.T,y)
    w = np.linalg.solve(XTX, XTY)
    loss = compute_mse_loss(y, tx, w)

    return w,loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression method with lambda_(regularization parameter)"""
    reg = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    XTX = np.dot(tx.T,tx) + reg
    XTY = np.dot(tx.T,y)
    w = np.linalg.solve(XTX, XTY)
    loss = compute_mse_loss(y, tx, w)

    return w,loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression"""
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):

        #Compute gradient and normalize
        grad = compute_lg_grad(y, tx, w)
        grad = grad / np.linalg.norm(grad)

        w = w - gamma * grad
        # store loss in order to check the convergence
        loss = compute_lg_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1]-losses[-2]) < threshold:
            break

    return w, losses[-1]

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Regularized logistic regression with lambda_(regularization parameter)"""

    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):

        #Compute gradient and normalize it
        grad = compute_lg_grad(y, tx, w)
        grad =  grad + 2*lambda_*w
        grad = grad / np.linalg.norm(grad)

        w = w - gamma * grad

        # store loss in order to check the convergence
        loss = compute_lg_loss(y, tx, w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]