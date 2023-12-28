

# data
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X_norm, y, test_size=0.2, random_state=2023)
X_tr.shape, X_te.shape, y_tr.shape, y_te.shape

### bayes classifier

def bayes(X_train, X_test, y_train, y_test):
    # find labels
    labels = np.unique(y_train)

    # sort by labels
    sorted_y_train_idx = y_train.argsort()
    sorted_y_test_idx = y_test.argsort()

    X_train = X_train[sorted_y_train_idx]
    X_test = X_test[sorted_y_test_idx]
    y_train = np.sort(y_train).reshape(-1, 1)
    y_test = np.sort(y_test).reshape(-1, 1)

    # divide train dataset into label arrays
    X_train_divided = []
    for label in labels:
        idx = np.where(y_train==label)[0]
        X_train_divided.append(X_train[idx])


    # get mean, std on each column per labels. 
    X_means = []
    X_stds = []
    X_probs = []
    for label in labels:
        # filter index
        idx = np.where(y_train==label)[0]
        # get stats
        label_means = X_train[idx].mean(axis=0)
        label_std = X_train[idx].std(axis=0)
        label_prob = X_train[idx].shape[0] / X_train.shape[0] # n_records / total records
        X_means.append(label_means)
        X_stds.append(label_std)
        X_probs.append(label_prob)

    X_means, X_stds, X_probs = np.array(X_means), np.array(X_stds), np.array(X_probs)


    # gaussian pdf
    # f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
    def pdf(x, mean, std):
        part1 = 1/((2*np.pi)**0.5*std)
        part2 = np.exp(-(x-mean)**2/(2*std**2))
        return part1 * part2
    
    # predict
    pred = []
    # iterate on each row
    for i in range(len(X_test)):
        _pdf = pdf(X_test[i], X_means, X_stds)
        _pdf = np.prod(_pdf, axis=1) # products of rows (classes)
        res = X_probs * _pdf

        pred.append(res.argmax())
    
    pred = np.array(pred).reshape(-1, 1)
    # print(pred, y_test)

    acc = np.where(pred==y_test)[0].shape[0]/y_test.shape[0]
    print('accuracy: ', acc)
    return acc


# bayes(X_tr, X_te[:10], y_tr, y_te[:10])
bayes(X_tr, X_te, y_tr, y_te)
