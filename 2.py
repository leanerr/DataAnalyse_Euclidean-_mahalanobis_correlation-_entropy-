# # محاسبه فاصله مهلنوبیس

# In[45]:


from numpy import linalg as LA
import pandas as pd
import numpy as np
import math
data = pd.read_excel('./dataset2.xls', header=None ,names=['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'])
data
# Function to find mean. 
def mean(arr, n): 

    sum = 0
    for i in range(0, n): 
        sum = sum + arr[i] 
        return sum / n 

# Function to find covariance. 
def covariance(arr1, arr2, n): 

	sum = 0
	for i in range(0, n): 
		sum = (sum + (arr1[i] - mean(arr1, n)) *(arr2[i] - mean(arr2, n))) 
	return sum / (n - 1) 

def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
        inv_covmat = LA.inv(cov)
        left_term = np.dot(x_minus_mu, inv_covmat)
        mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

data_x = data[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']].head(50)
data_x['mahala'] = mahalanobis(x=data_x, data=data[['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']])
data_x.head()

