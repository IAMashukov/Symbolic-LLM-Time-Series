import numpy as np 


X = np.load("./X_train.npy", allow_pickle= True)
y = np.load("./y_train.npy", allow_pickle= True)


print(X.shape)
print(y.shape)

print(np.unique(y, return_counts= True))