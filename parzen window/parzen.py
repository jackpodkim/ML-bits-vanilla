# data
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_norm, y, test_size=0.2, random_state=2023)
X_tr.shape, X_te.shape, y_tr.shape, y_te.shape


### parzen window
def parzen_window(X_train, X_test, y_train, y_test, spread=0.1):
    def gaussian_kernel(X1, X2, spread=0.1):
        if not isinstance(X2, np.ndarray):
            X2 = np.array(X2)

        X1 = X1.reshape(1, -1) # test
        # X2 = X2.reshape(1, -1) # train

        n_row1, dim1 = X1.shape
        n_row2, dim2 = X2.shape  

        N = n_row2
        D = dim2
        K = np.zeros([n_row1, n_row2])

        for i in range(n_row1):
            for j in range(n_row2):
                part1 = 1/((2*np.pi)**0.5*spread)**D
                part2 = np.exp(-0.5*((np.dot((X1[i, :]-X2[j,:]), (X1[i, :]-X2[j,:]).T))/(spread**2)))
                K[i, j] = 1/N * part1 * part2

        return K
    
    # find labels
    n_labels = np.unique(y_train)
    
    # sort by labels
    sorted_y_train_idx = y_train.argsort()
    sorted_y_test_idx = y_test.argsort()

    X_train = X_train[sorted_y_train_idx]
    X_test = X_test[sorted_y_test_idx]
    y_train = np.sort(y_train).reshape(-1, 1)
    y_test = np.sort(y_test).reshape(-1, 1)

    # divide train dataset into label arrays
    X_train_divided = []
    for label in range(len(n_labels)):
        idx = np.where(y_train==label)[0]
        X_train_divided.append(X_train[idx])
    
    # print(X_train_divided[0].shape)
    
    prob = []
    for i in range(len(n_labels)):
        prob.append(np.zeros(len(X_test)).reshape(len(X_test), 1))

    
    for i in range(len(X_test)):
        if i%1000==0: print(i, '/', len(X_test))
        prob[0][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[0], spread))
        prob[1][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[1], spread))
        prob[2][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[2], spread))
        prob[3][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[3], spread))
        prob[4][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[4], spread))
        prob[5][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[5], spread))
        prob[6][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[6], spread))
        prob[7][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[7], spread))
        prob[8][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[8], spread))
        prob[9][i] = np.sum(gaussian_kernel(X_test[i], X_train_divided[9], spread))

    y_tmp = np.stack((prob[0], 
                      prob[1], 
                      prob[2], 
                      prob[3], 
                      prob[4], 
                      prob[5], 
                      prob[6], 
                      prob[7], 
                      prob[8], 
                      prob[9]))
        
    y_pred = y_tmp.argmax(axis=0).reshape(-1, 1)
    # print('y_pred: ', y_pred)

    acc = np.where(y_test == y_pred)[0].shape[0] / y_pred.shape[0]
    
    # Accuracy
    print("Accuracy: ", acc)
    return acc

parzen_window(X_tr, X_te, y_tr, y_te)
