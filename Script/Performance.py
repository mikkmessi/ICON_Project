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

ss = StandardScaler()

N_PLAYERS = 3                                           # COSTANTE

# Creazione dizionario classificatori, che contiene come chiavi il nome dei class., come valori un istanza di essi

dict_regressors = {
    "Linear Regression": LinearRegression(),
    "Naive Bayes": ARDRegression(),
    "Random Forest": RandomForestRegressor(),
}


def load_and_model(file_path):
    '''
        The function loads an excel file from path in a pandas dataframe then
        does a one-hot encode of a categorical variable.
    :param      file_path: string
    :return:    stats: pandas dataframe
    '''

    stats = pd.read_excel(file_path)
    stats = pd.concat([stats, pd.get_dummies(stats['Ruolo'], prefix='Ruolo')], axis=1)
    
    return stats


def split(stats):
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
                "Contrasto_perso", "Dest", "Ricevuti", "Ricevuti_prog", "Tiri_Reti", "Tiri", "Tiri_specchio",
                "%Tiri_specchio", "Tiri_specchio_90", "Goal_tiro", "Dist_avg_tiri", "Tiri_puniz", "Contr", "Contr_vinti",
                "Dribb_blocked", "Dribb_no_block", "Dribb_sub", "%Dribb_blocked", "Press", "Press_vinti", "%Press_vinti",
                "Blocchi", "Tiri_block", "Tiri_porta_block", "Pass_block", "Intercett", "Tkl_Int", "Salvat", "Err_to_tiro",
                "Azioni_tiro", "Pass_tiro_gioco", "Pass_tiro_no_gioco", "Dribbling_tiro", "Tiri_tiro", "Falli_sub_tiro",
                "Azioni_dif_tiro", "Azioni_gol", "Pass_gol_gioco", "Pass_gol_no_gioco", "Dribbling_gol", "Tiri_gol",
                "Falli_gol", "Azioni_dif_gol", "Azioni_Autogol", "Ruolo_Att", "Ruolo_Dif", "Ruolo_Cen", "Ruolo_CenAtt",
                "Ruolo_AttCen", "Ruolo_DifAtt", "Ruolo_DifCen", "Ruolo_CenDif", "Ruolo_AttDif"]].values
        
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
            
    
def team(file_path):
    '''
        Reads from input players' IDs, extract them from the complete dataframe, saves in "my_team_full" the complete
        set of information, including the string values, then drop them and scale their numeric values.

    :param      file_path: string
    :return:    my_team_std: transformed array, with no string parameters
    :return:    my_team_full: pandas dataframe
    '''
              
    ids = np.empty([N_PLAYERS])
             
    for i in range(N_PLAYERS):
        ids[i] = input("Inserisci ID giocatore: ")

    stats = load_and_model(file_path)

    my_team = pd.DataFrame()
    
    for i in range(N_PLAYERS):
        my_team = my_team.append(stats.loc[stats['ID'] == ids[i]])

    my_team_full = my_team.copy()
    my_team = my_team.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)
    
    my_team_std = ss.transform(my_team)
        
    return my_team_std, my_team_full


def final_weight(player_id, match_day):
    '''
        Given a player, it gets information on the next match of his team from the excel file and returns
        their sum, scaled by 100.
    :param      player_id:   string
    :param      match_day:   integer
    :return:
    '''

    BEST_SCORER = 0.1                                       # CONSTANT

    # reading excel files
    file_path = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\"
    all_players = pd.read_excel(file_path + "Dataset_g26_NOPOR.xlsx", index_col="ID")
    calendario = pd.read_excel(file_path + "Calendario_2021.xlsx")
    classifica = pd.read_excel(file_path + "Statistiche_g26_v2.0.xlsx", sheet_name="Classifica", index_col="Squadra")

    player = all_players.loc[player_id]
    player_team = player["Squadra"]

    # retrieving matches for the day
    matches = list(calendario[match_day])

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

    # final weight for the single player
    f_weight = round((dev_pos + vs_dev_goals + p_dev_goals + bonus_best_scorer + lf_ratio_p_team + lf_ratio_vs_team)/100, 3)

    return f_weight
