class TemplateClassifier(BaseEstimator, ClassifierMixin):
        """This class is a transformer for sklearn users"""
     def __init__(self, demo_param='demo'):
         self.demo_param = demo_param

     def fit(self, X, y):


        return self
     def predict(self, X):

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
       return self.y_[closest]