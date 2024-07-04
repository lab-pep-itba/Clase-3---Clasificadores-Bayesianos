import numpy as np
from scipy.stats import multivariate_normal

# class LDA()

class QDA():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        value_counts = y.value_counts().to_dict()
        self.classes_ = np.array(list(value_counts.keys()))
        self.priors_ = np.array([v/len(y) for k, v in value_counts.items()])
        self.means_ = []
        self.covariance_ = []
        for c in self.classes_:
            data_by_class = X[y==c]
            self.means_.append(data_by_class.mean().values)
            self.covariance_.append(data_by_class.cov().values)
        self.means_ = np.array(self.means_)
        self.covariance_ = np.array(self.covariance_)
        
    def predict(self, X):
        probs = []
        prob_total = 0
        for i, c in enumerate(self.classes_):
            likelihood = multivariate_normal.pdf(X.values, self.means_[i], self.covariance_[i])
            probs.append(
                self.priors_[i] * likelihood
            )
            prob_total = prob_total + self.priors_[i] * likelihood
        probs = np.array(probs)
        probs = (probs / prob_total).T
        classes_indexes = np.argmax(probs, axis=1)
        return self.classes_[classes_indexes]
    
    def score(self, X, y):
        y_ = self.predict(X)
        

