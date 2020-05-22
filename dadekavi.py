#!/usr/bin/env python
# coding: utf-8

# # فراخوانی دیتا از دیتا ست و نمایش چند سطر ابتدایی

# In[6]:


import numpy as np
import pandas as pd
import math
data = pd.read_excel('./dataset2.xls', header=None,)
data.head()


# # محاسبه فاصله اقلیدسی

# با وارد کردن دو ستون دلخواه از بین اعداد 0 تا 11 فاصله اقلیدسی داده های موحود در این دو ستون با هم محاسبه میشود 

# In[7]:


a = data[0]
b = data[1]
# c = daat[3]
# d = daat[4]
# e = daat[5]
# f = daat[6]
# g = daat[7]
# h = daat[8]
# i = daat[9]
# j = daat[10]
# k = daat[11]
# f=[(a - b) ** 2)+((b - c) ** 2)+(c - d) ** 2)+(d - e) ** 2)+(e - f) ** 2)+(f - g) ** 2)+(g - h) ** 2)+(h - i) ** 2)+(i - j) ** 2)+(j - k) ** 2)]
# f

dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(a, b)]))
print("Euclidean distance from cols (0 to 1): ",dist)

x = int(input("enter target_col 1(0,11) x :"))
y = int(input("enter target_col 2(0,11) y :"))

x = data[x]
y = data[y]
distance = math.sqrt(sum([(a - b) ** 2 for a, b   in zip(x, y)]))
print("Euclidean distance from x to y: ",distance)



# نمایش ماتریس فاصله اقلیدسی برای تمامی حالات دیتاست

# In[21]:


rows,cols = data.shape
X = [[ 0 for row in range(cols)] for column in range(cols)]

for i in range(cols):
    for j in range(cols):
        dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(data[i], data[j])]))
        X[i][j] = float(dist)
np.matrix(X)

        


# In[22]:


from numpy import linalg as LA
def compute_ED_method1(data):
    m,n = data.shape
    D = np.zeros((n , n))
    for i in range (n):
        for j in range (i+1 , n):
            D[i , j] = LA.norm(data[:,i] - data[:,j])**2
            D[j , i] = D[i , j]
    return D

a = np.array(data)
compute_ED_method1(a)


# # محاسبه فاصله مهلنوبیس

# In[45]:


from numpy import linalg as LA
import pandas as pd
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


# # محاسبه ضریب همبستگی

# In[53]:


# Python Program to find correlation coefficient. 
import math 
data = pd.read_excel('./dataset2.xls', header=None )
# function that returns correlation coefficient. 
def correlationCoefficient(X, Y, n) : 
	sum_X = 0
	sum_Y = 0
	sum_XY = 0
	squareSum_X = 0
	squareSum_Y = 0
	
	
	i = 0
	while i < n : 
		# sum of elements of array X. 
		sum_X = sum_X + X[i] 
		
		# sum of elements of array Y. 
		sum_Y = sum_Y + Y[i] 
		
		# sum of X[i] * Y[i]. 
		sum_XY = sum_XY + X[i] * Y[i] 
		
		# sum of square of array elements. 
		squareSum_X = squareSum_X + X[i] * X[i] 
		squareSum_Y = squareSum_Y + Y[i] * Y[i] 
		
		i = i + 1
	
	# use formula for calculating correlation 
	# coefficient.
	soorat = (float)(n * sum_XY - sum_X * sum_Y)
	makhraj = (float)(math.sqrt((n * squareSum_X - sum_X * sum_X)* (n * squareSum_Y - sum_Y * sum_Y)))
	
	corr = soorat / makhraj
	return corr 
	
# Driver function 
x = int(input("Enter the first target col(0,11):"))
y = int(input("Enter the second target col(0,11):"))

X = data[x]
Y = data[y]

# Find the size of array. 
n = len(X) 

# Function call to correlationCoefficient. 
print ('{0:.6f}'.format(correlationCoefficient(X, Y, n))) 


# # محاسبه انتروپی

# با وارد کردن ستون دلخواه (از بین اعداد 0 تا 11) انتروپی برای داده های آن ستون محاسبه میگردد

# In[52]:


import numpy as np
import pandas as pd


data = pd.read_excel('./dataset2.xls', header=None )

def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    #removing repeated data using unique python method 
    #to return a tuple containing the list of unique values in arr and a list of their corresponding frequencies.
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy



x = int(input("please enter the target col (0,11) to calcualte entropy:"))
entropy(data[x])

