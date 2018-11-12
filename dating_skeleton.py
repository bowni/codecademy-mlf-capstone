
# coding: utf-8

# # Codecademy ML fundamentals - Capstone project

# ## Load and explore dataset

# *Load libraries and datasets*

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


df = pd.read_csv("profiles.csv")


# *Explore quickly the dataset*

# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


list(df.columns)


# *Explore a bit more and plot a few features*

# In[6]:


df.income.value_counts()


# In[8]:


df_explo = df.groupby('orientation').size()
df_explo.plot(kind='bar')


# In[9]:


df_explo = df.groupby('body_type').size()
df_explo.plot(kind='bar')


# In[10]:


plt.hist(df.age, bins=100)
plt.show()


# In[82]:


plt.hist(df.height.dropna(), bins=100)
plt.xlabel('Height')
plt.ylabel('# samples')
plt.title('Height distribution')
plt.show()


# In[12]:


plt.hist(df.income.dropna(), bins=100)
plt.show()


# In[83]:


df.income.value_counts()


# In[15]:


df.isna().sum()


# ## Create new features
# 
# We create 2 new features, extracted from the categorical features 'religion' and 'offspring'. Beyond the base options they describe, these 2 variables contain additional information about:
# - the seriousness of the religion involvement, across religions
# - the desire to have kids, whether having kids already or not
# 
# We extract this information based on the text indication of the level of religion seriousness (resp. kids desire). We consider that no indication of religion seriousness means an average level of seriousness about it, and no indication of kids desire means that the person doesn't know if it wants kids or not. The enables ordering the levels of religion seriousness (resp. kids desire)

# In[16]:


df.religion.value_counts()


# In[17]:


df['religion_seriousness'] = 0
df['religion_seriousness'].loc[df['religion'].str.contains('very serious', na=False)] = 2
df['religion_seriousness'].loc[df['religion'].str.contains('somewhat serious', na=False)] = 1
df['religion_seriousness'].loc[df['religion'].str.contains('not too serious', na=False)] = -1
df['religion_seriousness'].loc[df['religion'].str.contains('laughing', na=False)] = -2


# In[18]:


df['religion_seriousness'].value_counts()


# In[19]:


df.offspring.value_counts()


# In[68]:


df['kids_desire'] = 0
df['kids_desire'].loc[df['offspring'].str.contains('wants them', na=False)] = 2
df['kids_desire'].loc[df['offspring'].str.contains('wants more', na=False)] = 2
df['kids_desire'].loc[df['offspring'].str.contains('wants kids', na=False)] = 2
df['kids_desire'].loc[df['offspring'].str.contains('might want them', na=False)] = 1
df['kids_desire'].loc[df['offspring'].str.contains('might want more', na=False)] = 1
df['kids_desire'].loc[df['offspring'].str.contains('might want kids', na=False)] = 1
df['kids_desire'].loc[df['offspring'].str.contains('doesn&rsquo;t want', na=False)] = -1


# In[69]:


df['kids_desire'].value_counts()


# In[84]:


df_explo = df.groupby('religion_seriousness').size()
df_explo.plot(kind='bar')


# In[85]:


df_explo = df.groupby('kids_desire').size()
df_explo.plot(kind='bar')


# ## Formulate a ML question and prepare the data
# 
# The body type seems like an interesting characteristic to study. In dating situation it contributes significantly to the first impression the two persons have from each other. Some people work actively on shaping their bodies, other suffer it. There are a lot of stereotypes and clichés about body types / shapes, but on the other hand there are also real correlations that exist.
# We are wondering if we could manage to predict it from other descriptive features like:
# - age, height, sex
# - drinks, smokes
# - religion_seriousness, kids_desire
# 
# 

# In[22]:


df.shape


# In[23]:


df.dropna(subset=['body_type'], how='any', inplace = True)
df.shape


# In[70]:


df.body_type.value_counts()


# In[72]:


target_feature = df.get(['body_type'])
body_type_mapping = {'skinny': 0, 'thin': 1, 'fit': 2, 'athletic': 3, 
       'jacked': 4, 'average': 5, 'used up': 6, 'a little extra': 7, 'curvy': 8, 'full figured': 9, 
       'rather not say': 10, 'overweight': 11}
target_feature['body_type_code'] = target_feature.body_type.map(body_type_mapping)
target_feature_pruned = target_feature.drop('body_type', 1)
target_feature_pruned.body_type_code.value_counts()


# In[73]:


my_features = df.get(['age', 'height', 'sex', 'drinks', 'smokes', 'religion_seriousness', 'kids_desire'])
# my_features.head(5)
my_features.isna().sum()


# In[74]:


# Fill empty values
my_features['height'].fillna(my_features['height'].dropna().median(), inplace = True)
my_features['drinks'].fillna(my_features['drinks'].dropna().mode()[0], inplace = True)
my_features['smokes'].fillna(my_features['smokes'].dropna().mode()[0], inplace = True)
my_features.isna().sum()


# In[75]:


# Convert ordered categorical values to numerical values
drinks_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4}
my_features["drinks_code"] = my_features.drinks.map(drinks_mapping)
my_features["smokes_code"] = my_features.smokes.map(smokes_mapping)
my_features.isna().sum()


# In[76]:


# Convert unordered categorical values to dummy variables
rich_features = pd.concat([my_features.get(['age', 'height', 'religion_seriousness', 'kids_desire', 'drinks_code', 'smokes_code']),
                           pd.get_dummies(my_features.sex, prefix='sex')],
                          axis=1)
rich_features.head(5)


# In[77]:


# Drop useless columns
rich_features_pruned = rich_features.drop('sex_m', 1)
rich_features_pruned.head(5)


# *Let's normalize the data*

# In[78]:


from sklearn.preprocessing import MinMaxScaler

x = rich_features_pruned.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

rich_features_normalized = pd.DataFrame(x_scaled, columns=rich_features_pruned.columns)
rich_features_normalized.head(5)


# ## Modelling
# 
# We are going to compare K-Nearest Neighbors and Support Vector Machines approaches.

# In[79]:


# %%time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

features_array = rich_features_normalized.values
target_array = target_feature_pruned.values

features_train, features_test, target_train, target_test = train_test_split(
    features_array, target_array, test_size=0.20, random_state=42)


# In[81]:


# %%time

from sklearn.neighbors import KNeighborsClassifier

accuracies = []
for k in range(30, 251):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(features_train, target_train.ravel())
    accuracies.append(classifier.score(features_test, target_test))

k_list = range(30, 251)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Body Type Classifier Accuracy")
plt.show()


# In[86]:


# %%time

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(features_train, target_train.ravel())

print(classifier.score(features_test, target_test))


# We are going to compare K-Nearest Neighbors Regressor and Multiple Linear Regression

# In[89]:


# %%time

from sklearn.neighbors import KNeighborsRegressor

knnregressor = KNeighborsRegressor(n_neighbors = 200, weights = "distance")
knnregressor.fit(features_train, target_train.ravel())
print(knnregressor.score(features_test, target_test.ravel()))


# In[88]:


# %%time

from sklearn.linear_model import LinearRegression

mlregressor = LinearRegression()
mlregressor.fit(features_train, target_train.ravel())
print(mlregressor.score(features_test, target_test.ravel()))


# This doesn't prove to be conclusive. Our features [age, height, sex, religion_seriousness, kids_desire, smokes, drinks] don't classify well the body type.
# 
