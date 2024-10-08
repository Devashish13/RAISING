#!/usr/bin/env python
# coding: utf-8

# # RAISING implementation

# RAISING is a neural network(NN) framework implemented in a two stage approach, first performing hyperparameter tuning to devise the best NN architecture then training on the data to estimate the feature importance. Here we describe the implementation of RAISING on a simulated data.

# ## Package installation

# Assuming package folder has been downloaded, package folder contains RAISING, README_RAISING.html, simulated data(sim1.csv), and this document explaining the implementation of RAISING.To install the RAISING package, open your terminal, navigate to the package directory, and run the following command:

# In[10]:


#pip install -e RAISING


# ## Import required libraries

# In[11]:


import pandas as pd
from RAISING.hp_tune_select import *


# ## Load simulated data 
# We will use the simulated data from capblancq et. al.(2018) stored in sim1.csv. There are 64 population and authors have sampled 10 individuals from each of the the population at the end of simulations. Simulated data has 640 individuals and 1000 Loci which is input data and output data has 640 individuals and 10 environmental factors.

# In[12]:


df = pd.read_csv("sim1.csv")


# We will store the input genotype matrix into X_data variable and environmental matrix in y_data

# In[15]:


X_data = df.iloc[:,14:]
X_data.columns = ["X" + str(i) for i in range(1, X_data.shape[1]+1)]

y_data = df.iloc[:,1:11]
print(X_data.head())
print(y_data.head())


# ## Stage 1 : Hyperparameter tuning

# We will use hp_optimization() function defined in RAISING for hyperparameter tuning.The description of the function can be found in README file **README_RAISING.html**

# In[ ]:


model = hp_optimization(input_data=X_data, output_data=y_data, objective_fun="val_loss", output_class="continuous", max_trials=10)


# In[17]:


print(model.summary())


# ## Stage 2 : Feature Selection
# 
# We will use feature_importance() function defined in RAISING for feature selection.The description of the function can be found in README file **README_RAISING.html**

# In[ ]:


GenFeat_df = feature_importance(input_data=X_data, output_data=y_data, feature_set=X_data.columns.to_list(), iteration=10, feature_method="DeepFeatImp")


# We will scale the feature importance f genetic loci between [0,1] corresponding to linear environment gradient(envir2). We will generate a plot where we will have on the x-axis  (genetic loci position) and on the y-axis (feature importance corresponding to linear environment)

# In[21]:


def minmaxscale(vec):
        return((vec - min(vec))/(max(vec) - min(vec)))

GenFeat_df["scaled_feat"] = minmaxscale(vec = GenFeat_df["envir2"])
GenFeat_df["loci_index"] = GenFeat_df.index+1


# In[22]:


import matplotlib.pyplot as plt

actual_pos = [101,111,121,131,141,151,161,171,181,191]

plt.figure(figsize=(10, 6))
plt.plot(GenFeat_df['loci_index'], GenFeat_df['scaled_feat'], marker='o', linestyle='-', color='b')

highlight_df = GenFeat_df[GenFeat_df['loci_index'].isin(actual_pos)]
plt.scatter(highlight_df['loci_index'], highlight_df['scaled_feat'], color='red', zorder=5)

plt.title('Line Plot of Scaled Features by Loci corresponding to linear environment')
plt.xlabel('Loci Index')
plt.ylabel('Scaled Feature')
plt.grid(True)
plt.show()


# In[ ]:




