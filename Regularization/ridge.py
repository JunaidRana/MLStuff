from sklearn.linear_model import Ridge
import numpy as np 



n_samples, n_features= 10, 5
np.random.seed(0)
y=np.random.randn(n_samples)
X=np.random.randn(n_samples, n_features)
clf=Ridge(alpha=1.0)
clf.fit(X,y)

Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)

