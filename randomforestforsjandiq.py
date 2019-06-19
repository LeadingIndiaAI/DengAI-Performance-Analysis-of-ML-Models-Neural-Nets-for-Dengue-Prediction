import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
train = pd.read_csv("dengue_features_train.csv")
train_labels = pd.read_csv("dengue_labels_train.csv")
test = pd.read_csv("dengue_features_test.csv")

train_target = train_labels['total_cases']
train_target.head()

test_city = test['city']
test_year = test['year']
test_weekofyear = test['weekofyear']
train.describe()
train.describe(include = ['O'])
test.describe()
test.describe(include = ['O'])
df = train.append(test, ignore_index = True)
cats = []
for col in df.columns.values:
    if df[col].dtype == 'object':
        cats.append(col)
        
df_cat = df[cats]
df_cont = df.drop(cats, axis=1)

for col in df_cont.columns.values:
    df_cont[col] = df_cont[col].fillna(df[col].median())

for col in df_cat.columns.values:
    df_cat[col] = df_cat[col].fillna(df_cat[col].value_counts().index[0])

df_cat = df_cat.drop(['week_start_date'], axis = 1)

df_cat = pd.get_dummies(df_cat)

df = df_cont.join(df_cat)
df.shape

train = df.iloc[0:train.shape[0]]
test = df.iloc[train.shape[0]:]






from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
scorer = make_scorer(mean_absolute_error, False)

# Correlation tells relation between two attributes.
# Correlation requires continous data. Hence, ignore categorical data
# Calculates pearson co-efficient for all combinations
df_cont_corr = df_cont.corr()
# Set the threshold to select only highly correlated attributes
threshold = 0.8
# List of pairs along with correlation above threshold
corr_list = []
j_columns = []
#Search for the highly correlated pairs
for i in range(0,df_cont.shape[1]):
    if i not in np.ravel(j_columns):
        for j in range(i+1,df_cont.shape[1]):
            if (df_cont_corr.iloc[i,j] >= threshold and df_cont_corr.iloc[i,j] < 1) or (df_cont_corr.iloc[i,j] < 0 and df_cont_corr.iloc[i,j] <= -threshold):
                corr_list.append([df_cont_corr.iloc[i,j],i,j]) #store correlation and columns index
                j_columns.append([j])
                
                
                

j_columns = np.ravel(j_columns)
j_columns

for v,i,j in corr_list:
    print ("%s and %s = %.2f" % (i,j,v))
    
    
    
features_to_drop = []
for i in j_columns:
    features_to_drop.append(df_cont.columns[i])
    
    
features_to_drop

df_cont = df_cont.drop(features_to_drop, axis = 1)

df = df_cont.join(df_cat)
df.shape

train = df.iloc[0:train.shape[0]]
test = df.iloc[train.shape[0]:]
df_cont_norm = df_cont
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
for col in df_cont_norm.columns.values:
            df_cont_norm[col]= sc.fit_transform(df_cont_norm[col].values.reshape(1,-1))[0]

    
df_norm = df_cont_norm.join(df_cat)

train = df_norm.iloc[0:train.shape[0]]
test = df_norm.iloc[train.shape[0]:]

cols=["alpha", "score","std_score"]
cv_score_Lasso = pd.DataFrame(columns=cols)

alphas = [0.0001,0.001, 0.05, 0.1, 1, 5, 10, 20, 30, 50]

for alpha in alphas:
    model_Lasso = Lasso(alpha = alpha)
    cv_score_Lasso_entry = np.sqrt(-cross_val_score(estimator=model_Lasso, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).mean()
    cv_score_Lasso_std = np.sqrt(-cross_val_score(estimator=model_Lasso, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).std()
    cv_score_Lasso = cv_score_Lasso.append(pd.DataFrame([[alpha, cv_score_Lasso_entry,cv_score_Lasso_std]], columns=cols))
print(cv_score_Lasso)

cols=["alpha", "score","std_score"]
cv_score_Ridge = pd.DataFrame(columns=cols)

alphas= [0.0001,0.001, 0.05, 0.1, 1, 5, 10, 20, 30, 50]

for alpha in alphas:
    model_Ridge = Ridge(alpha = alpha)
    cv_score_Ridge_entry = np.sqrt(-cross_val_score(estimator=model_Ridge, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).mean()
    cv_score_Ridge_std = np.sqrt(-cross_val_score(estimator=model_Ridge, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).std()
    cv_score_Ridge = cv_score_Ridge.append(pd.DataFrame([[alpha, cv_score_Ridge_entry,cv_score_Ridge_std]], columns=cols))
print(cv_score_Ridge)

model_linear = LinearRegression()
cv_score_Linear = np.sqrt(-cross_val_score(estimator=model_linear, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).mean()
cv_score_Linear_std = np.sqrt(-cross_val_score(estimator=model_linear, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).std()
print(cv_score_Linear,cv_score_Linear_std)


model_RFR = RandomForestRegressor(n_estimators =500,random_state=8)
cv_score_RFR = np.sqrt(-cross_val_score(estimator=model_RFR, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).mean()
cv_score_RFR_std = np.sqrt(-cross_val_score(estimator=model_RFR, X=train, y=np.ravel(train_target), cv=10, scoring = scorer)).std()
print(cv_score_RFR,cv_score_RFR_std)

train = df.iloc[0:train.shape[0]]
test = df.iloc[train.shape[0]:]

model_RFR.fit(train, np.ravel(train_target))

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=8,
           verbose=0, warm_start=False)


predictions = model_RFR.predict(test)

predictions = predictions.astype(int)

