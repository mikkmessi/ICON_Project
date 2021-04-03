'''
Created on 10 mar 2021

@author: kecco
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

ss = StandardScaler()
pca = PCA(n_components=0.5)

FILE_PATH = "https://raw.githubusercontent.com/mikkmessi/ICON_Project/main/Dataset/"

# lista di moduli da comparare per trovare il modulo migliore per la formazione migliore
modules_list = ['3-4-3',
                '3-5-2',
                '4-5-1',
                '4-4-2',
                '4-3-3',
                '5-3-2',
                '5-4-1']

# Creazione dizionario classificatori, che contiene come chiavi il nome dei regressori, come valori un istanza di essi
dict_regressors = {
    "Linear Regression": LinearRegression(),
    "Naive Bayes": ARDRegression(),
    "Random Forest": RandomForestRegressor(),
}


def load_and_model(sheet_name):
    '''
        The function loads an excel file from path in a pandas dataframe then
        does a one-hot encode of a categorical variable.
    :param      file_path:      string
    :param      sheet_name:     string
    :return:    stats:          pandas dataframe
    '''

    stats = pd.read_excel(FILE_PATH + "Dataset_NOPOR.xlsx", sheet_name=sheet_name)
    stats = pd.concat([stats, pd.get_dummies(stats['Ruolo'], prefix='Ruolo')], axis=1)
    
    return stats


def split_and_std(stats):
    '''
        Once removed all the string parameters from the dataframe, splits "stats" in train and test sets and scales
        the feature sets.

    :param      stats: pandas dataframe
    :return:    X_train_std, X_test_std, Y_train, Y_test: list
    '''

    stats = stats.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

    X = stats[["Partite_giocate", "PG_Titolare",    "Min_giocati", "Min_90", "Reti", "Assist", "Reti_no_rig", "Reti_rig",
                "Rig_tot", "Amm", "Esp", "Reti_90", "Assist_90", "Compl", "Tent", "%Tot", "Dist",    "Dist_prog",
                "Pass_Assist", "Pass_tiro", "Pass_terzo", "Pass_area", "Cross_area", "Pass_prog", "Tocchi", "Drib_vinti",
                "Drib_tot", "%Drib_vinti", "Giocatori_sup", "Tunnel", "Controlli_palla", "Dist_controllo",
                "Dist_controllo_vs_rete", "Prog_controllo_area_avv", "Controllo_area", "Controllo_perso",
                "Contrasto_perso", "Dest", "Ricevuti", "Ricevuti_prog", "Tiri_reti", "Tiri", "Tiri_specchio",
                "%Tiri_specchio", "Tiri_specchio_90", "Goal_tiro", "Dist_avg_tiri", "Tiri_puniz", "Contr", "Contr_vinti",
                "Dribbl_blocked", "Dribbl_no_block", "Dribbl_sub", "%Dribbl_blocked", "Press", "Press_vinti", "%Press_vinti",
                "Blocchi", "Tiri_block", "Tiri_porta_block", "Pass_block", "Intercett", "Tkl_Int", "Salvat", "Err_to_tiro",
                "Azioni_tiro", "Pass_tiro_gioco", "Pass_tiro_no_gioco", "Dribbling_tiro", "Tiri_tiro", "Falli_sub_tiro",
                "Azioni_dif_tiro", "Azioni_gol", "Pass_gol_gioco", "Pass_gol_no_gioco", "Dribbling_gol", "Tiri_gol",
                "Falli_gol", "Azioni_dif_gol", "Azioni_Autogol", "Ruolo_Att", "Ruolo_Dif", "Ruolo_Cen"]].values
        
    Y = stats["Mf"].values
        
    # suddividiamo il dataseet in due dataset, uno di training ed uno di test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    # standardizzo il train set creando un modello di standardizzazione
    X_train_std = ss.fit_transform(X_train)
    X_test_std = ss.transform(X_test)
    
    return X_train_std, X_test_std, Y_train, Y_test


def batch_classify(X_train, Y_train, X_test, Y_test, no_regressors=3):
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
    
    for classifier_name, regressor in list(dict_regressors.items())[:no_regressors]:
        
        regressor.fit(X_train, Y_train)
        train_score = regressor.score(X_train, Y_train)
        test_score = regressor.score(X_test, Y_test)
        mean = (train_score + test_score) / 2
        
        if mean > massimo:
            
            massimo = mean
            best_model = regressor
        
        dict_models[classifier_name] = {'model': regressor, 'train_score': train_score, 'test_score': test_score, 'media': mean}
    
    print(dict_models)
            
    return best_model
            
    
def team(sheet_name):
    '''
        Reads from a txt file the players' names, extract them from the complete dataframe, saves in "df_my_team_full" the complete
        set of information, including the string values, then drop them and scale their numeric values, in df_my_team_std.

    :param      file_path:         string
    :param      sheet_name:        string
    :return:    df_my_team_std:    transformed array, with no string parameters
    :return:    df_my_team_full:   pandas dataframe
    '''

    df_my_team_names = pd.read_csv(FILE_PATH + "My_team_NOPOR.txt", sep=',', header=None, names=["Nome_Cognome", "Squadra"])

    all_players = pd.read_excel(FILE_PATH + "Dataset_NOPOR.xlsx", sheet_name=sheet_name)

    # pd.set_option("display.max_columns", None)
    df_my_team = pd.DataFrame()

    for player in list(df_my_team_names["Nome_Cognome"]):
        players = list(all_players["Nome_Cognome"])
        for each_player in players:
            player_name = each_player.split("\\")
            if player == player_name[1]:
                df_my_team = df_my_team.append(all_players.loc[all_players["Nome_Cognome"] == each_player])

    df_my_team_full = df_my_team.copy()

    df_my_team = pd.concat([df_my_team, pd.get_dummies(df_my_team['Ruolo'], prefix='Ruolo')], axis=1)
    df_my_team = df_my_team.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)
    
    df_my_team_std = ss.transform(df_my_team)
        
    return df_my_team_std, df_my_team_full


def final_weight(dataset_name, sheet_name, player_id, next_fb_day):
    '''
        Given a player, it gets information on the next match of his team from the excel file and returns
        their sum, scaled by 100.

    :param      dataset_name:   string
    :param      sheet_name:     string
    :param      player_id:      string
    :param      next_fb_day:   integer

    :return:    f_weight:       float
    '''

    BEST_SCORER = 0.1                                       # CONSTANT

    # reading excel files
    all_players = pd.read_excel(FILE_PATH + dataset_name, sheet_name=sheet_name, index_col="ID")
    calendario = pd.read_excel(FILE_PATH + "Calendario_2021.xlsx")
    classifica = pd.read_excel(FILE_PATH + "Classifica.xlsx", sheet_name=sheet_name, index_col="Squadra")

    player = all_players.loc[player_id]
    player_team = player["Squadra"]

    # retrieving matches for the day
    matches = list(calendario[next_fb_day])

    # retrieving opponent team for the player specified
    for match in matches:
        if player_team in match:
            teams_of_match = match.split("-")

    for each_team in teams_of_match:
        if each_team != player_team:
            vs_team = each_team

    # player and opponent team in rank (whole row)
    p_team = classifica.loc[player_team]
    vs_team = classifica.loc[vs_team]

    # deviation between each team position
    dev_pos = vs_team["Pos"] - p_team["Pos"]                # FIRST METRIC

    # deviation between goals scored and conceded
    vs_dev_goals = vs_team["Diff_reti"] * (-1)              # SECOND METRIC

    p_dev_goals = p_team["Diff_reti"]                       # THIRD METRIC

    # bonus applied in case the player is the best scorer of his team
    bonus_best_scorer = 0                                   # By default 0, if player is not the best scorer of the team
    name_player = player["Nome_Cognome"].split("\\")

    if name_player[0] in p_team["Miglior_marcatore"]:
        bonus_best_scorer = BEST_SCORER                     # FOURTH METRIC

    last_five = p_team["Ultime_5"]                          # last five results for the player team
    lf_ratio_p_team = last_five.count("V") / 5              # FIFTH METRIC

    last_five = vs_team["Ultime_5"]                         # last five results for the opponent team
    lf_ratio_vs_team = (-1) * (last_five.count("V") / 5)    # SIXTH METRIC
    
    freq_best_XI = 0                                        # SEVENTH METRIC
    
    for i in range(24, 29):
        
        best_XI = pd.read_excel(FILE_PATH + "Best_XI.xlsx", sheet_name="g" + str(i))
        
        if name_player[1] in list(best_XI["Nome_Cognome"]):
            freq_best_XI = freq_best_XI + 1
    
    freq_best_XI = freq_best_XI / 5
    
    # final weight for the single player
    f_weight = round((dev_pos + vs_dev_goals + p_dev_goals + bonus_best_scorer + lf_ratio_p_team + lf_ratio_vs_team + freq_best_XI)/100, 5)

    return f_weight


def principal_component_analysis(X_train_std, X_test_std):
    '''
        Given training and test sets, reduce the number of features based on variance ratio
    
    :param      X_train_std:    list
    :param      X_test_std:     list
    '''
    pca_test = PCA(n_components=89)
    pca_test.fit(X_train_std)
    sns.set(style='whitegrid')
    plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
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


def hypertuning(best_model, X_train_std, Y_train):
    '''
        Perform RandomizedSearchCV and GridSearchCV in order to find the best combination of parameters for the regressor
    
    :param     best_model: regressor
    :param      X_train: list
    :param      Y_train: list
    '''
    
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    bootstrap = [True, False]
    param_dist = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
        }
    rs = RandomizedSearchCV(best_model, param_dist, n_iter = 100, cv = 3, verbose = 1, n_jobs=-1, random_state=0)
    rs.fit(X_train_std, Y_train)
    print(rs.best_params_)
       
    rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    rs_df = rs_df.drop([ 'mean_fit_time','std_fit_time','mean_score_time','std_score_time','params','split0_test_score','split1_test_score','split2_test_score','std_test_score'],axis=1)
    print(rs_df.head(10))
    
    fig, axs = plt.subplots(ncols=3, nrows=2)
    sns.set(style="whitegrid", color_codes=True, font_scale = 2)
    fig.set_size_inches(30,25)
    
    sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0,0], color='lightgrey')
    axs[0,0].set_title(label = 'n_estimators', size=30, weight='bold')
    
    sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0,1], color='coral')
    axs[0,1].set_title(label = 'min_samples_split', size=30, weight='bold')
    
    sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0,2], color='lightgreen')
    axs[0,2].set_title(label = 'min_samples_leaf', size=30, weight='bold')
    
    sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1,0], color='wheat')
    axs[1,0].set_title(label = 'max_features', size=30, weight='bold')
    
    sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1,1], color='lightpink')
    axs[1,1].set_title(label = 'max_depth', size=30, weight='bold')
    
    sns.barplot(x='param_bootstrap',y='mean_test_score', data=rs_df, ax=axs[1,2], color='skyblue')
    axs[1,2].set_title(label = 'bootstrap', size=30, weight='bold')

    plt.show()
    
    n_estimators = [300,500,700,1000]
    max_features = ['sqrt']
    max_depth = [2,3,7,14,15]
    min_samples_split = [2,7,12,23,34]
    min_samples_leaf = [2,7,12,18,23,28]
    bootstrap = [False]
    param_grid ={'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    gs = GridSearchCV(best_model, param_grid, cv = 3, verbose = 1, n_jobs=-1)
    gs.fit(X_train_std, Y_train)
    rfc = gs.best_estimator_
    
    print(gs.best_params_)
    print(rfc)          
    
    return 0


def importance(best_model, dataset):
    '''
        Shows dataset's features importance given a model of regression
    
    :param     best_model: regressor
    :param     dataset: pandas dataframe
    '''
    
    feature_list = list(dataset.columns)
    # Get numerical feature importances
    importances = list(best_model.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    
    return 0


def final_score(my_team_full, prediction, next_fb_day, dataset_name, sheet_name):
    '''
        Sums up each player "Media Fantavoto" prediction with each player "final_weight".
                
    :param      my_team_full: pandas dataframe          user's full team dataframe, including string characteristics
    :param      prediction: list                        list of predictions of "Media Fantavoto" for team players
    :param      football_day: integer                   the football day when the match is played
    :param      dataset_name: string
    :param      sheet_name: string

    :return:    pandas dataframe                        final score for all players in user's team
    '''

    # dictionary to transform into a pandas dataframe
    dict_final_score = {
        'ID': [],
        'Nome_Cognome': [],
        'Ruolo': [],
        'Prediction': [],
        'Final_weight': [],
        'Final_score': []
    }

    i = 0

    for player in list(my_team_full['ID']):
        row_player = my_team_full.loc[my_team_full["ID"] == player]

        player_id = int(row_player["ID"].values)
        player_name = row_player["Nome_Cognome"].values[0]
        player_name = player_name.split('\\')
        player_role = row_player["Ruolo"].values[0]
        weight = final_weight(dataset_name, sheet_name, player, next_fb_day)

        dict_final_score['ID'].append(player_id)
        dict_final_score['Nome_Cognome'].append(player_name[1])
        dict_final_score['Ruolo'].append(player_role)
        dict_final_score['Prediction'].append(prediction[i])
        dict_final_score['Final_weight'].append(weight)
        dict_final_score['Final_score'].append(prediction[i] + weight)

        i += 1

    df_final_score = pd.DataFrame(dict_final_score)

    return df_final_score


def best_goalkeeper(next_fb_day, dataset_name="Portieri.xlsx", sheet_name="g28"):

    ss = StandardScaler()

    # Training and testing
    dataset_por = pd.read_excel(FILE_PATH + dataset_name, sheet_name="g26")

    dataset_por = dataset_por.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

    X = dataset_por[["Partite_giocate", "PG_titolare", "Min_giocati", "Min_90", "Reti_sub",
                     "Reti_sub_90", "Tiri_sub", "Parate", "Porta_inviolata", "Rig_tot", "Rig_concessi",
                     "Rig_salvati", "Rig_mancati"]].values

    Y = dataset_por["Mf"].values

    # suddividiamo il dataseet in due dataset, uno di training ed uno di test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # standardizzo il train set creando un modello di standardizzazione
    X_train_std = ss.fit_transform(X_train)
    X_test_std = ss.transform(X_test)

    best_model = batch_classify(X_train_std, Y_train, X_test_std, Y_test)

    # Predicting
    df_my_team_names = pd.read_csv(FILE_PATH + "My_team_POR.txt", sep=',', header=None, names=["Nome_Cognome", "Squadra"])

    all_players = pd.read_excel(FILE_PATH + dataset_name, sheet_name=sheet_name)  # last football day played (giornata "corrente")

    df_my_team = pd.DataFrame()

    for player in list(df_my_team_names["Nome_Cognome"]):
        players = list(all_players["Nome_Cognome"])
        for each_player in players:
            player_name = each_player.split("\\")
            if player == player_name[1]:
                df_my_team = df_my_team.append(all_players.loc[all_players["Nome_Cognome"] == each_player])

    df_my_team_full = df_my_team.copy()
    df_my_team = df_my_team.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

    df_my_team_std = ss.transform(df_my_team)

    prediction_list = list(best_model.predict(df_my_team_std))

    df_final_score_gk = final_score(df_my_team_full, prediction_list, next_fb_day, dataset_name=dataset_name, sheet_name=sheet_name)  # last football day played (giornata "corrente")

    return df_final_score_gk


def best_eleven(df_final_score, df_final_score_gk):

    df_final_score_gk = df_final_score_gk.sort_values(by=['Final_score'], ascending=False)
    print(df_final_score_gk)
    print()

    # separare i giocatori per ruolo, ordinarli dopo la separazione
    df_fs_dif = df_final_score.loc[df_final_score["Ruolo"] == "Dif"]            # dataframe_finalscore_difensori
    df_fs_dif = df_fs_dif.sort_values(by=['Final_score'], ascending=False)

    df_fs_cen = df_final_score.loc[df_final_score["Ruolo"] == "Cen"]
    df_fs_cen = df_fs_cen.sort_values(by=['Final_score'], ascending=False)

    df_fs_att = df_final_score.loc[df_final_score["Ruolo"] == "Att"]
    df_fs_att = df_fs_att.sort_values(by=['Final_score'], ascending=False)

    goalkeeper = df_final_score_gk.iloc[0]

    best_team_score = 0

    for module in modules_list:
        # taking the best players for the module
        module_str = module.split("-")

        nbr_dif, nbr_cen, nbr_att = int(module_str[0]), int(module_str[1]), int(module_str[2])

        df_best_eleven = pd.DataFrame(columns=list(df_fs_att))

        df_best_eleven = df_best_eleven.append(goalkeeper)
        df_best_eleven = df_best_eleven.append(df_fs_dif[:nbr_dif])
        df_best_eleven = df_best_eleven.append(df_fs_cen[:nbr_cen])
        df_best_eleven = df_best_eleven.append(df_fs_att[:nbr_att])

        team_score = 0

        for player_score in df_best_eleven['Final_score']:
            team_score += player_score

        if team_score > best_team_score:
            df_best_team = df_best_eleven.copy()
            best_module = module
            best_team_score = team_score

    return df_best_team, best_module

