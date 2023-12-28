

import numpy as np
import pandas as pd
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FLDR

X_df = pd.read_csv('mnist_train.csv')
y_df = pd.read_csv('mnist_test.csv')

X_train = X_df.iloc[:, 1:].to_numpy()
y_train = X_df.label.to_numpy()
X_test = y_df.iloc[:, 1:].to_numpy()
y_test = y_df.label.to_numpy()

col_index = {i:col for i, col in enumerate(X_df.columns[1:])}

print(X_train.shape, y_train.shape)


def FLDR(X, y):
    '''
    X, y = ndarray
    out -> ndarray
    '''
    # n dim, n labels
    _, dim = X.shape
    n_labels = np.unique(y).shape[0]


    # std per feature (per col)
    # init with zeros
    std = np.zeros((dim,))
    # get deviations
    for i in range(0, 10):
        label_idx = np.where(y==i)[0]
        std = np.add(std, np.std(X[label_idx], axis=0)) 


    # mu per label (n labels, dim)
    label_mu = []
    # add means for each unique labels
    for i in range(0, 10):
        label_idx = np.where(y==i)[0]
        mu = np.mean(X[label_idx].T, 1)
        label_mu.append(mu.T) 
    label_mu = np.array(label_mu)
    # mu_over_std = mean / std
    mu_over_std = label_mu / std

    # calculations
    idx_tmp = []
    corr_tmp = []
    for i in range(n_labels):
        WW1 = np.delete(mu_over_std, i, axis=0).T
        WW2 = np.array([mu_over_std[i]] * (n_labels-1)) # make copies of rows
        WW = WW2.T - WW1
        rankW = np.min(WW, axis=1)
        
        u = np.sort(-np.abs(rankW.T))
        v = np.argsort(-np.abs(rankW.T)) # argsort returns ordered index in place

        idx_tmp.append(v)
        corr_tmp.append(u)

    idx_tmp = np.array(idx_tmp).T.reshape(1, n_labels*dim)
    corr_tmp = np.array(corr_tmp).T.reshape(1, n_labels*dim)
    
    corr_tmp_sort = np.sort(corr_tmp) # sort ascending
    idx_tmp_sort = idx_tmp.T[np.argsort(corr_tmp)]

    idx_tmp = np.flip(idx_tmp_sort) # sort descending
    corr_tmp = np.flip(corr_tmp_sort)


    # get indices
    [u, v] = np.unique(idx_tmp, return_index=True)
    # w,s
    w = np.sort(v)
    s = np.argsort(v)
    val_res = corr_tmp.T[w][::-1] # reversed
    idx_res = u[s][::-1] # reversed

    # feature rank, values
    return idx_res, val_res


fisher_feature_sorted, f_coef = FLDR(X_train, y_train)
fisher_feature_sorted[:20], f_coef[:20]

# (array([ 73,  74,  72,  71,  75, 358, 359,  70, 330,  76, 386, 102, 331,
#         739, 738, 387,  69, 101, 740, 737], dtype=int64),
#  array([[-0.26952155],
#         [-0.26922692],
#         [-0.25354988],
#         [-0.23060583],
#         [-0.22960327],
#         [-0.22159649],
#         [-0.21342038],
#         [-0.20738063],
#         [-0.19762811],
#         [-0.1957859 ],
#         [-0.18795048],
#         [-0.18312061],
#         [-0.18291892],
#         [-0.18204378],
#         [-0.18138069],
#         [-0.18067425],
#         [-0.17439867],
#         [-0.17301314],
#         [-0.16963973],
#         [-0.16803073]]))