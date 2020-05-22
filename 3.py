# # محاسبه ضریب همبستگی

# In[53]:


# Python Program to find correlation coefficient. 
import math
import pandas as pd

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

