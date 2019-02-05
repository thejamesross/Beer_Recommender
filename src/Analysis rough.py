

import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt

from surprise import Reader
from surprise.dataset import Dataset

from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.prediction_algorithms.knns import KNNBasic
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.prediction_algorithms.matrix_factorization import SVDpp

from surprise.model_selection import train_test_split
from surprise.model_selection import RepeatedKFold
from surprise.model_selection import KFold

from surprise.model_selection.search import GridSearchCV
from surprise.evaluate import GridSearch

from surprise import accuracy

os.chdir('P:/temp/Jamie/Ad Hoc/Python/')


ratings = [beer.ratings for beer in receps]
ratings = pd.concat(ratings)
ratings.to_csv('ratings.csv')
#ratings = pd.read_csv('ratings.csv')
ratings = ratings[[x in ['10', '20', '30', '40', '50'] for x in ratings.Rating]]
ratings['Rating'] = pd.to_numeric(ratings['Rating']) / 10
ratings = ratings[['User', 'Recipe', 'Rating']]
ratings.column = ['userID', 'itemID', 'rating']
#%% Summary Stats
print('Recipes Collected:\t' + str(len(fls)))
print('\nUnique Useable Recipes:\t' + str(len(set(ratings.Recipe))))
print('\nUnique Users:\t' + str(len(set(ratings.User))))
print('\nTotal Shape:\t' + str(ratings.shape[0]) + '\n\n')
print(ratings.head())
print('\n')
print(ratings.Rating.plot(kind = 'hist', title = 'Distribution of Ratings'))
#%% Most rated recipes

#To do:
    # User loading
    # Recipe loading
    #Impact of adding data
    #Hard Examples
    #Predict on new user    

ratings.\
    groupby('Recipe')['User'].\
           count().\
                sort_values(ascending=False).\
                           head(10)

#%% Most active users -- Check correlation of numbers with rating/time?
ratings.\
    groupby('User')['Recipe'].\
           count().\
                sort_values(ascending=False).\
                           head(10)

#%% Distribution of Ratings
print(ratings.Rating.describe())
print(set(ratings.Rating))

#%% Build train - test split
reader = Reader(rating_scale=(1, 5))
data   = Dataset.load_from_df(ratings, reader)

random.seed(42)
random.shuffle(data.raw_ratings)
cut_off = int(len(data.raw_ratings) * 0.75)

train_ratings = data.raw_ratings[:cut_off]
test_ratings  = data.raw_ratings[cut_off:]

data.raw_ratings = train_ratings

#%% Evaluate baseline on all, bias and test error
def evaluator(algo, df, cv_method, verbose = False):
    """
    wrapper to streamline evaluation
    """

    results = []    
    cntr    = 0
    for train_set, test_set in cv_method.split(df):
        algo.fit(train_set)
        predictions = algo.test(test_set)
        accrcy      = accuracy.rmse(predictions)
        results.append(pd.DataFrame({'Fold'  : [cntr], 
                                     'RMSE'  : [accrcy]}))
        cntr += 1
    
    results = pd.concat(results)
    
    if verbose:
        return results
    else:
        return [results['RMSE'].mean(), results['RMSE'].std()]

cv_method    = RepeatedKFold(
                      n_splits = 5, 
                      n_repeats = 3, 
                      random_state = 42)

algo         = BaselineOnly()  

#CV
cv_score     = evaluator(algo, 
                         data, 
                         cv_method)
#Bias
train_set    = data.build_full_trainset()
algo.fit(train_set)
preds        = algo.test(train_set.build_testset())
bias         = accuracy.rmse(preds)

#Error
test_set     = data.construct_testset(test_ratings)
preds        = algo.test(test_set)
err          = accuracy.rmse(preds)

#Baseline CV
print('\nBaseline CV avg:\t' + str(cv_score[0]))
print('\nBaseline CV std:\t' + str(cv_score[1]))
print('\nBaseline bias:\t' + str(bias))
print('\nBaseline OOS error:\t' + str(err))


#%%

ratings_eda = pd.DataFrame(train_ratings)
ratings_eda = ratings_eda[[0, 1, 2]]
ratings_eda.columns = ['User', 
                       'Recipe',
                       'Rating']
ratings_sd = ratings_eda.groupby('Recipe').agg(['mean', 'std', 'count'])
ratings_sd.columns = ratings_sd.columns.levels[1] 

ratings_sd['count'].hist()
ratings_sd['count'].describe()


ratings_sd.groupby('count')['mean'].count()

ratings_sd[ratings_sd['count'] > 2].sort_values('std', ascending=False)


#%% Ignore

res = []
for ii in np.exp2(np.arange(4, 10)):
    data.raw_ratings = train_ratings[:int(ii)]
    cv_score     = evaluator(algo, 
                             data, 
                             cv_method)
    res.append({'Number' : ii,
                'Mean'   : cv_score[0],
                'STDev'  : cv_score[1]})
data.raw_ratings = train_ratings        
res = pd.DataFrame.from_dict(res)
    
# Baseline CV avg:
# 0.592112495061874
# Baseline CV std:
# 0.14580305114950923






#%% Train knn with means

# 0.590432
#0.144686
cv_method    = RepeatedKFold(n_splits = 5, 
                             n_repeats = 5, 
                             random_state = 1)

#algo =  KNNBasic()

param_grid = {'k'     : [4, 8, 16, 32, 64],
              'min_k' : [1, 2, 3, 4]}

clf_knn = GridSearchCV(KNNBasic,
                   param_grid,
                   cv = cv_method)

clf_knn.fit(data)
results_df = pd.DataFrame.from_dict(clf_knn.cv_results)
results_df.sort_values('mean_test_rmse', inplace = True)
results_df.head(10)

#%% Train SVD


#0.587609
#0.134100

param_grid = {'n_factors'  : np.arange(25, 150, 25),
              'n_epochs'   : np.arange(5, 20, 5),
              'lr_all'     : np.arange(0, 0.08, 0.02),
              'reg_all'    : np.arange(0, 0.08, 0.02)}

param_grid_sm = {'n_factors'  : np.arange(40, 60, 2),
                 'n_epochs'   : np.arange(8, 13, 1),
                 'lr_all'     : np.arange(0.03, 0.08, 0.01),
                 'reg_all'    : np.arange(0.04, 0.1, 0.01)}

clf_svd = GridSearchCV(SVD,
                       param_grid_sm,
                       cv = cv_method)
clf_svd.fit(data)
results_df = pd.DataFrame.from_dict(clf_svd.cv_results)


results_df.sort_values('mean_test_rmse', inplace = True)
results_df.head(10)

#%% Train SVD


#0.587609
#0.134100

param_grid = {}

clf_svd = GridSearchCV(BaselineOnly,
                   param_grid)

clf_svd.fit(data)
results_df = pd.DataFrame.from_dict(clf_svd.cv_results)
results_df.sort_values('mean_test_rmse', inplace = True)
results_df.head(10)



#%% Train SVDpp

# 0.587619
# 0.140615

param_grid = {'n_factors'  : np.arange(25, 100, 25),
              'n_epochs'   : np.arange(10, 25, 5),
              'lr_all'     : np.arange(0.02, 0.1, 0.02),
              'reg_all'    : np.arange(0.02, 0.1, 0.02)}

clf_svdpp = GridSearchCV(SVDpp,
                   param_grid)

clf_svdpp.fit(data)
results_df = pd.DataFrame.from_dict(clf_svdpp.cv_results)
results_df.sort_values('mean_test_rmse', inplace = True)
results_df.head(10)


results_df.\
    groupby('param_reg_all')['mean_test_rmse'].\
           mean().plot(title='Average OOS Error by Number of Latent Factors')
           
           
           .plot(kind = 'boxplot')

#%%
new_clf = clf_svd.best_estimator['rmse']
new_clf.fit(data.build_full_trainset())

train = data.build_full_trainset()

train_frame = pd.DataFrame(new_clf.qi[:,:5])
train_frame['Recipe'] = np.nan
           
for key, value in zip(train._raw2inner_id_items.keys(),
                      train._raw2inner_id_items.values()):
    train_frame['Recipe'].iloc[value] = key

def to_rating(x):
    try:
        x = pd.to_numeric(x) / 10
        x = x.mean()
    except:
        x = None
        
    return x

df_desc = [{'Recipe' : beer.ratings.Recipe.iloc[0],
            'Style'  : beer.style,
            'ABV'    : beer.abv,
            'Taste'  : beer.taste_rat,
            'Bitter' : beer.bitter,
            'Rating' : to_rating(beer.ratings.Rating),
            'Name'   : beer.name} for beer in receps]
df_desc = pd.DataFrame.from_dict(df_desc)

df_desc = pd.merge(train_frame, 
                   df_desc, 
                     how = 'left',
                     on  = 'Recipe')
df_desc['Name'] = [x.split(' - ')[0] for x in df_desc['Name']]
df_desc['Diff'] = df_desc[0] + df_desc[1]
df_desc['Recipe']= [x.replace('recipes/', '') for x in df_desc['Recipe']]

df_desc.\
    sort_values(1)[['Recipe', 'Bitter', 'ABV', 'Style', 'Name', 'Rating']].\
               tail(10)
               .to_csv('low 1.csv')


for i in range(df_desc.shape[0]):
    x = df_desc['1'].iloc[i]
    y = df_desc['2'].iloc[i]
    z = df_desc['2'].iloc[i]
    plt.scatter(x, y, z, marker=',', color='red')
    plt.text(x, y, df_desc['Style'].iloc[i], fontsize=9)
plt.show()

df_desc.plot.scatter(0, 1)





train_frame = pd.DataFrame(new_clf.pu[:,:5])
train_frame['User'] = np.nan
           
for key, value in zip(train._raw2inner_id_users.keys(),
                      train._raw2inner_id_users.values()):
    train_frame['User'].iloc[value] = key


train_frame.\
    sort_values(0)[[0, 1, 'User']].\
               tail(10)
               
               .to_csv('low 1.csv')



#%%

new_clf = clf_svd.best_estimator['rmse']
cv      = KFold(n_splits = 5, random_state = 1234)

store   = []
for train_set, test_set in cv.split(data):
    new_clf.fit(train_set)
    predictions = new_clf.test(test_set)
    
    for pred, known in zip(predictions, test_set):
        known = list(known)
        known.append(pred.est)
        store.append(known)
store = [{'User'   : x[0],
          'Recipe' : x[1],
          'True'   : x[2],
          'Pred'   : x[3]} for x in store]
    
store = pd.DataFrame.from_dict(store)
store['error'] = (store['Pred'] - store['True'])**2
store['cnt']   = 1
     
store = store.\
    groupby('Recipe')[['error', 'cnt']].\
           sum().\
              sort_values('error', ascending = False)

#%%Model Evaluation

cv_method    = RepeatedKFold(
                      n_splits = 5, 
                      n_repeats = 10, 
                      random_state = 42)

for algo in [clf_knn.best_estimator['rmse'],
             clf_svd.best_estimator['rmse'],
             clf_svdpp.best_estimator['rmse']]:            

    #CV
    cv_score     = evaluator(algo, 
                             data, 
                             cv_method)
    #Bias
    train_set    = data.build_full_trainset()
    algo.fit(train_set)
    preds        = algo.test(train_set.build_testset())
    bias         = accuracy.rmse(preds)
    
    #Error
    test_set     = data.construct_testset(test_ratings)
    preds        = algo.test(test_set)
    err          = accuracy.rmse(preds)
    
    #Baseline CV
    print('\nModel CV avg:\t' + str(cv_score[0]))
    print('\nModel CV std:\t' + str(cv_score[1]))
    print('\nModel bias:\t' + str(bias))
    print('\nModel OOS error:\t' + str(err))
    

print('Mod Score CV error:\n' + str(err))

#%% Get recall and accuracy estimates -- maybe do this on full dataset
# Can vary threshold as well
def precision_recall_at_k(predictions, k=10, threshold=4):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

store_outer = []
store_inner = []

data.raw_ratings = train_ratings + test_ratings

cv = KFold()

for kk in range(1, 21):
    for jj in np.arange(2.5, 5, 0.25):
        store_inner = []
        for trainset, testset in cv.split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, 
                                                        k=kk, 
                                                        threshold=jj)
            
            #Fix here to add threshold
            store_inner.append(pd.DataFrame({'Precision'  : [sum(prec for prec in precisions.values()) / len(precisions)],
                                        'recalls'    : [sum(rec for rec in recalls.values()) / len(recalls)],
                                        'Count'      : [len(predictions)]}))    
        store_inner = pd.concat(store_inner)
        store_inner['Precision'] = store_inner['Precision'] * store_inner['Count']
        store_inner['recalls']   = store_inner['recalls'] * store_inner['Count']
        store_inner              = store_inner.sum()
        store_inner['Precision'] = store_inner['Precision'] / store_inner['Count']
        store_inner['recalls']   = store_inner['recalls'] / store_inner['Count']
        store_inner['k']         = kk
        store_inner['threshold'] = jj
        store_outer.append(store_inner)

store_outer = pd.concat(store_outer, axis = 1)
store_outer = store_outer.transpose()
store_outer.index = store_outer.k
store_outer['Recall'] = store_outer['recalls']
store_outer[store_outer['threshold']==4.5][['Precision', 'Recall']].plot(title = 'Precision and Recall at 4.5 Threshold\nVarying the Number of Items Shown')


#%%
cv = KFold()

hold_data = data.raw_ratings

results_m   = []
results_s   = []
results_l   = []
n_samples   = 500
for ii in [10, 100, 250, 500, 750, 800]:
    for jj in range(n_samples):    
        random.shuffle(hold_data)
        train_ratings    = hold_data[:ii]
        test_ratings     = hold_data[ii:]
        data.raw_ratings = train_ratings
        
        algo.fit(data.build_full_trainset())        
        preds        = algo.test(data.construct_testset(test_ratings))        
        err          = accuracy.rmse(preds)



        results_m.append(err)
        results_s.append(accrcy**2)
    results_l.append({'Count' : ii,
                      'Mean'  : sum(results_m) / 500,
                      'Second': sum(results_s) / 500})
    print(ii)
    
data.raw_ratings = hold_data

cv_method    = RepeatedKFold(
                      n_splits = 5, 
                      n_repeats = 3, 
                      random_state = 42)


res = evaluator(algo,
                  data,
                  cv_method)    
results_l.append({'Count'     : len(hold_data),
                      'Mean'  : res[0],
                      'Second': res[0]**2 + res[1]}) #frmo inital cv
df = pd.DataFrame.from_dict(results_l)

df['Training Set Size'] = df['Count']
df.index = df['Training Set Size']
df.head(6).Mean.plot(title = 'Bootsrapped Estimate of Adding More Data')

#%%
#Get all predictions

res = []
for key, value in zip(train_set._raw2inner_id_items.keys(),
                      train_set._raw2inner_id_items.values()):
    pred = algo.predict(train_set._raw2inner_id_users['2891/tx-brewer'], 
                        value)
    res.append({'User' : '2891/tx-brewer',
                'Item' : key,
                'Value' :pred.est})
res    = pd.DataFrame.from_dict(res)
prev_serv = ratings[ratings['User'] == '2891/tx-brewer']['Recipe'].values

res    = res[[x not in prev_serv for x in res['Item']]]
res.sort_values('Value', inplace = True, ascending = False)
res    = res.head(30)
res['Item'] = [x.replace('recipes/', '') for x in res['Item']]
res = pd.merge(res,
            df_desc[['Recipe', 'ABV', 'Bitter', 'Name', 'Rating', 'Style', 'Taste']],
            left_on = 'Item', 
            right_on = 'Recipe')
res = res.head(9)
res = res[['Item', 'Recipe', 'User', 'Value', 'ABV', 'Bitter', 'Name', 'Rating', 'Style', 'Taste']]


res[['Name', 'Recipe', 'Style', 'ABV', 'Bitter', 'Value', 'Rating']].to_csv('reccomendation.csv')

#%%
# Mod Score CV error:
# 0.6787965350047481

# Baseline CV error:
# 0.6833814839684862

store_inner = []
for trainset, testset in cv.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    
    store_inner.append(pd.DataFrame.from_dict([{'UID' : x.uid,
                         'IID' : x.iid,
                         'r_ui': x.r_ui,
                         'est' : x.est} for x in predictions]))    
df = pd.concat(store_inner)
df['Error'] = (df['est'] - df['r_ui'])**2
df_IID = df.groupby('UID')['Error'].agg({'mean', 'count'}).reset_index()
df_IID[df_IID['count'] > 2].sort_values('mean', ascending=False).head()
df_IID.columns = '

#%% Train on all data, and predict on new user


# do attribute type using q and p matrices






train_ratings = data.raw_ratings[:cut_off]
test_ratings  = data.raw_ratings[cut_off:]
test_set     = data.construct_testset(test_ratings)
preds        = algo.test(test_set)

