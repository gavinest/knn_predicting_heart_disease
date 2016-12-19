import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
'imports customs confusion matrix module. in confusion_mat.py'
from confusion_mat import ConfusionMat


def replace_q(x):
    '''used by clean_data function to remove question marks from dataset'''
    if x.strip() == '?':
        return np.nan
    else:
        return x

def clean_data():
    '''
    imports and cleans data set

    INPUTS: None

    OUTPUTS:
        pandas dataframe of cleaned data
        list - categorical data columns
        list - continuous data columns
    '''
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    df = pd.read_csv(url, index_col=False, names=cols, header=None)

    #replace '?' with 'np.nan' for now to avoid errors in eda
    #cast dtypes as float
    df['ca'] = df['ca'].apply(replace_q).astype('float64')
    df['thal'] = df['thal'].apply(replace_q).astype('float64')

    #drop rows with nans for now. only looses 6 rows. consider filling with mean values in future
    df.dropna(inplace=True)

    #separate categorical variables from continuous
    cat_var = [col for col in df.columns.tolist() if len(df[col].unique()) <=5]
    cont_var = [col for col in df.columns.tolist() if len(df[col].unique()) > 5]

    #map booleans to predictor column see eda for details on why
    df['num'] = df['num'] > 0
    df['num'] = df['num'].map({False: 0, True: 1})
    return df, cat_var, cont_var

def make_kfolds(x, y):
    '''
    creates generator object for model evaluation from training sets

    INPUTS: x_train set and y_train set
    OUPUTS: yields next kfold split of training data when called by 'train_score_one object'

    '''
    kf = KFold(n_splits=5, random_state=42)
    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        yield x_train, x_test, y_train, y_test

def train_score_one(splits, k=1, use_kfolds=False):
    '''
    INPUT: list (train_test_split datasets)

    prints accuracy and confusion matrix values for one model.
    default knn parameters
    '''
    x_train, x_test, y_train, y_test = splits

    if use_kfolds:
        kf = make_kfolds(x_train, y_train)
        kf_scores = []
        for fold in kf:
            xfold_train, xfold_test, yfold_train, yfold_test = fold
            knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            knn.fit(xfold_train, yfold_train)
            kf_scores.append(knn.score(xfold_test, yfold_test))
        print k, np.mean(kf_scores)
        return np.mean(kf_scores)
    else:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(x_train, y_train)
        print k, knn.score(x_test, y_test)
        # ConfusionMat(y_test, knn.predict(x_test))
        return knn.score(x_test, y_test)

def find_k(max_k, splits, use_kfolds=False):
    '''
    increment over k to find best k and plot. calls train_score_one function.

    INPUTS:
        - INT max k to iterate to
        - LIST splits of training/ test data in list form
        - BOOL whether to use KFolds or not

    OUTPUTS:
        - graph of k

    '''
    scores = []
    for k in range(1, max_k + 1):
        scores.append(train_score_one(splits, k=k, use_kfolds=use_kfolds))

    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.plot(range(1, max_k+1), scores)
    ax.set_xlabel('k')
    ax.set_ylabel('Accuracy')
    ax.set_title('KNearestNeighbors varying K')
    ax.set_xticks(range(1, max_k+1))
    plt.savefig('./imgs/find_k.png')
    plt.show()

def scree_plot(pca, title=None):
    '''
    scree plot for identifying principal components in PCA.

    INPUTS:
        - pca object
        - STRING title of plot
    '''

    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.bar(ind, vals, 0.35)

    for i in xrange(num_components):
     ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    plt.xticks(ind, ['PC-{}'.format(i) for i in range(1,len(ind)+1)], rotation='vertical')

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 13+0.45)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")

    if title is not None:
     plt.title(title)


if __name__ == '__main__':
    df, cat_var, cont_var = clean_data()

    'separate predictor column(y) from features(x), scale all features'
    y = df.pop('num').values
    x = scale(df.values, axis=0)

    'train/test/split with random_state for reproducibility of results'
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    'Manipulated code below here to interact with functions above if __name__ == __main__ block and to perform analysis.'

    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    pca = PCA(n_components=10, random_state=42)
    reduced_data = pca.fit_transform(x_train)
    # scree_plot(pca, title='Principal Component Analysis')

    'print results of k_folds on knn module'
    # print 'kfolds,', train_score_one([reduced_data, pca.transform(x_test), y_train, y_test], k=1, use_kfolds=True)

    'PCA occurs here'
    knn.fit(reduced_data, y_train)
    print knn.score(pca.transform(x_test), y_test)
    print ConfusionMat(y_test, knn.predict(pca.transform(x_test)))

    'uncomment for running find_k function'
    find_k(max_k=20, splits=[reduced_data, pca.transform(x_test), y_train, y_test], use_kfolds=True)
