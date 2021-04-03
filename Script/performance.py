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


def final_weight(all_players, calendario, classifica, player_id, next_fb_day):
    '''
        Given a player, it gets information on the next match of his team from the excel file and returns
        their sum, scaled by 100.

    :param      all_players:    pandas dataframe
    :param      calendario:     pandas dataframe
    :param      classifica:     pandas dataframe
    :param      player_id:      string
    :param      next_fb_day:    integer

    :return:    f_weight:       float
    '''

    BEST_SCORER = 0.1                                       # CONSTANT

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
        
        best_XI = pd.read_excel(FILE_PATH + "Best_XI.xlsx", sheet_name=str(i))
        
        if name_player[1] in list(best_XI["Nome_Cognome"]):
            freq_best_XI = freq_best_XI + 1
    
    freq_best_XI = freq_best_XI / 5
    
    # final weight for the single player
    f_weight = round((dev_pos + vs_dev_goals + p_dev_goals + bonus_best_scorer + lf_ratio_p_team + lf_ratio_vs_team + freq_best_XI)/100, 5)

    return f_weight


def final_score(my_team_full, prediction, next_fb_day, dataset_name, sheet_name):
    '''
        Sums up each player "Media Fantavoto" prediction with each player "final_weight".
                
    :param      my_team_full: pandas dataframe          user's full team dataframe, including string characteristics
    :param      prediction: list                        list of predictions of "Media Fantavoto" for team players
    :param      next_fb_day: integer                   the football day when the match is played
    :param      dataset_name: string
    :param      sheet_name: string

    :return:    pandas dataframe                        final score for all players in user's team
    '''

    # reading excel files
    all_players = pd.read_excel(FILE_PATH + dataset_name, sheet_name=sheet_name, index_col="ID")
    calendario = pd.read_excel(FILE_PATH + "Calendario_2021.xlsx")
    classifica = pd.read_excel(FILE_PATH + "Classifica.xlsx", sheet_name=sheet_name, index_col="Squadra")

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
        weight = final_weight(all_players, calendario, classifica, player, next_fb_day)

        dict_final_score['ID'].append(player_id)
        dict_final_score['Nome_Cognome'].append(player_name[1])
        dict_final_score['Ruolo'].append(player_role)
        dict_final_score['Prediction'].append(prediction[i])
        dict_final_score['Final_weight'].append(weight)
        dict_final_score['Final_score'].append(prediction[i] + weight)

        i += 1

    df_final_score = pd.DataFrame(dict_final_score)

    return df_final_score


def best_goalkeeper(TRAIN_FB_DAY, next_fb_day, sheet_name, dataset_name="Portieri.xlsx"):

    ss = StandardScaler()

    # Training and testing
    dataset_por = pd.read_excel(FILE_PATH + dataset_name, sheet_name=str(TRAIN_FB_DAY))
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

    best_model = best_regressor(X_train_std, Y_train, X_test_std, Y_test)

    # Predicting
    df_my_team_names = pd.read_csv(FILE_PATH + "My_team_POR.txt", sep=',', header=None, names=["Nome_Cognome", "Squadra"])

    all_players = pd.read_excel(FILE_PATH + dataset_name, sheet_name=sheet_name)  # last football day played (giornata "corrente")

    df_my_team = pd.DataFrame()
    players = list(all_players["Nome_Cognome"])

    for player in list(df_my_team_names["Nome_Cognome"]) or len(df_my_team) < 3:
        i = 0
        exit = False
        while not exit:
            player_name = players[i].split("\\")
            if player == player_name[1]:
                df_my_team = df_my_team.append(all_players.loc[all_players["Nome_Cognome"] == players[i]])
                exit = True
            i += 1

    df_my_team_full = df_my_team.copy()
    df_my_team = df_my_team.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

    df_my_team_std = ss.transform(df_my_team)

    prediction_list = list(best_model.predict(df_my_team_std))

    df_final_score_gk = final_score(df_my_team_full, prediction_list, next_fb_day, dataset_name=dataset_name, sheet_name=sheet_name)  # last football day played (giornata "corrente")

    return df_final_score_gk


def best_eleven(df_final_score, df_final_score_gk):

    df_final_score_gk = df_final_score_gk.sort_values(by=['Final_score'], ascending=False)

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

