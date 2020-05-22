
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
