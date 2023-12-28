import numpy as np

### svm
class SVM:
    def __init__(self, C=3, kernel_name=None):
        self.C = C
        self.kernel_name = kernel_name


    def train(self, X, y):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False

        num_data, _ = X.shape

        # kernel matrix K
        K = np.zeros((num_data, num_data)) # kernel_matrix
        for i in range(num_data):
            if i%1000==0: print(f'K {i}/{num_data}')
            for j in range(num_data):
                K[i, j] = self.kernel(X[i], X[j])

        # quad prog
        P = np.outer(y, y).astype(np.double) * K
        q = -np.ones(num_data)
        G = np.vstack([-np.eye(num_data), np.eye(num_data)])
        h = np.hstack([np.zeros(num_data), np.ones(num_data) * self.C])
        A = y.reshape(1, -1)
        b = np.array([0.])

        P = matrix(P.astype(np.double), tc='d')
        q = matrix(q, tc='d')
        G = matrix(G, tc='d')
        h = matrix(h, tc='d')
        A = matrix(A.astype(np.double), tc='d')
        b = matrix(b, tc='d')

        qp_res = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(qp_res['x'])

        eps = 1e-3
        sv_inx = np.where(alpha > eps)[0]

        # output
        self.sv_inx = sv_inx
        self.alpha = alpha[sv_inx]
        self.sv_X = X[sv_inx]
        self.sv_y = y[sv_inx]

        # bias y[s]-ai*yi*xi
        self.b = self.sv_y[0]
        for i in range(len(self.alpha)):
            self.b -= self.alpha[i] * self.sv_y[i] * self.kernel(self.sv_X[i], self.sv_X[0])
        print('train done.')
        gc.collect()


    def predict(self, X):
        y_pred = []
        for Xi, row in enumerate(X):
            if Xi%1000==0: print(f'predict {Xi}/{len(X)}')
            res = 0
            for i in range(len(self.alpha)):
                w = self.alpha[i] * self.sv_y[i] * self.kernel(self.sv_X[i], row)
                res += w
            res += self.b
            y_pred.append(np.sign(res))

        
        return np.array(y_pred)


    def kernel(self, X1, X2, arg=1):
        if self.kernel_name=='linear':
            return np.dot(X1.T, X2) 
        elif self.kernel_name=='rbf':
            return np.exp(-0.5 * (np.linalg.norm(X1-X2)**2)/(arg**2))
        else:
            return 0
