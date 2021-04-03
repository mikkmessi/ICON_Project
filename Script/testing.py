import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# Creazione dizionario classificatori, che contiene come chiavi il nome dei regressori, come valori un istanza di essi
dict_regressors = {
    "Linear Regression": LinearRegression(),
    "Naive Bayes": ARDRegression(),
    "Random Forest": RandomForestRegressor(),
}

pca = PCA(n_components=40)


def best_regressor(X_train, Y_train, X_test, Y_test, no_regressors=3):
    '''
        Given training and test sets, trains each regressor in dict_regressors, compute their train and test score,
        prints them for each regressor and returns the best regressor based on the mean between
        train and test score.

    :param      X_train: list
    :param      Y_train: list
    :param      X_test: list
    :param      Y_test: list
    :param      no_regressors: integer

    :return:    best_model: regressor
    '''
    dict_models = {}
    massimo = 0

    for regressor_name, regressor in list(dict_regressors.items())[:no_regressors]:

        regressor.fit(X_train, Y_train)
        train_score = regressor.score(X_train, Y_train)
        test_score = regressor.score(X_test, Y_test)
        mean = (train_score + test_score) / 2

        if mean > massimo:
            massimo = mean
            dict_best_model = {'name': regressor_name, 'model': regressor, 'train_score': train_score, 'test_score': test_score,
                                        'media': mean}

        dict_models[regressor_name] = {'model': regressor, 'train_score': train_score, 'test_score': test_score,
                                        'media': mean}

    print(dict_models)

    return dict_best_model


def principal_component_analysis(X_train_std, X_test_std):
    '''
        Given training and test sets, reduce the number of features based on variance ratio

    :param      X_train_std:    list
    :param      X_test_std:     list
    '''
    pca_test = PCA(n_components=83)
    pca_test.fit(X_train_std)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle='--', x=20, ymin=0, ymax=1)
    plt.show()

    evr = pca_test.explained_variance_ratio_
    cvr = np.cumsum(pca_test.explained_variance_ratio_)
    pca_df = pd.DataFrame()
    pca_df['Cumulative Variance Ratio'] = cvr
    pca_df['Explained Variance Ratio'] = evr
    print(pca_df.head(20))

    pca.fit(X_train_std)

    X_train_pca = pca.transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    return X_train_pca, X_test_pca


def hypertuning(best_model, X_train_std, Y_train, X_test_std, Y_test):
    '''
        Perform RandomizedSearchCV and GridSearchCV in order to find the best combination of parameters for the regressor

    :param     best_model: regressor
    :param      X_train: list
    :param      Y_train: list
    '''

    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start=1, stop=15, num=15)]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=50, num=10)]
    bootstrap = [True, False]
    param_dist = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }
    rs = RandomizedSearchCV(best_model, param_dist, n_iter=100, cv=3, verbose=1, n_jobs=-1, random_state=0)
    rs.fit(X_train_std, Y_train)
    print(rs.best_params_)

    rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    rs_df = rs_df.drop(
        ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'params', 'split0_test_score',
         'split1_test_score', 'split2_test_score', 'std_test_score'], axis=1)
    print(rs_df.head(10))

    fig, axs = plt.subplots(ncols=3, nrows=2)
    sns.set(style="whitegrid", color_codes=True, font_scale=2)
    fig.set_size_inches(20, 15)

    sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0, 0], color='lightgrey')
    axs[0, 0].set_title(label='n_estimators', size=30, weight='bold')

    sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0, 1], color='coral')
    axs[0, 1].set_title(label='min_samples_split', size=30, weight='bold')

    sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0, 2], color='lightgreen')
    axs[0, 2].set_title(label='min_samples_leaf', size=30, weight='bold')

    sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1, 0], color='wheat')
    axs[1, 0].set_title(label='max_features', size=30, weight='bold')

    sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1, 1], color='lightpink')
    axs[1, 1].set_title(label='max_depth', size=30, weight='bold')

    sns.barplot(x='param_bootstrap', y='mean_test_score', data=rs_df, ax=axs[1, 2], color='skyblue')
    axs[1, 2].set_title(label='bootstrap', size=30, weight='bold')

    plt.show()

    n_estimators = [300, 500, 700, 1000]
    max_features = ['sqrt']
    max_depth = [2, 3, 7, 15]
    min_samples_split = [2, 7, 12, 23]
    min_samples_leaf = [2, 7, 12, 18]
    bootstrap = [False]
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}
    gs = GridSearchCV(best_model, param_grid, cv=3, verbose=1, n_jobs=-1)
    gs.fit(X_train_std, Y_train)
    rfc = gs.best_estimator_

    print(gs.best_params_)
    print(rfc)

    return rs.score(X_train_std, Y_train), rs.score(X_test_std, Y_test), gs.score(X_train_std, Y_train), gs.score(X_test_std, Y_test),


def importance(best_model, dataset):
    '''
        Shows dataset's features importance given a model of regression

    :param     best_model: regressor
    :param     dataset: pandas dataframe
    '''

    dataset = dataset.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra', 'Mf'], axis=1)

    feature_list = list(dataset.columns)
    # Get numerical feature importances
    importances = list(best_model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    return 0
