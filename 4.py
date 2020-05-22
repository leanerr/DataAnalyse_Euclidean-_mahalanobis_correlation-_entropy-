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
    ent = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    print(str(ent))





x = int(input("please enter the target col (0,11) to calcualte entropy:"))
entropy(data[x])

