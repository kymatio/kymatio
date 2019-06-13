from sklearn import

class transformer(BaseEstimator, ClassifierMixin):
        """This class is a transformer for sklearn users"""
     def __init__(self, demo_param='demo'):
         self.demo_param = demo_param
         self.end_shape = ()

     def transform(self, data):
         n_sample = data.numelements()/np.prod(self.end_shape)
         data = data.reshape()


        return self
     def predict(self, X):

         closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
       return self.y_[closest]