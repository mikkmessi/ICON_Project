import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FILE_PATH = "https://raw.githubusercontent.com/mikkmessi/ICON_Project/main/Dataset/"
ss = StandardScaler()


def load_and_model(sheet_name):
    '''
        Carica un file excel dal path in un dataframe pandas e converte le feature nominali in integer.

    :param      sheet_name:     string
    :return:    stats:          pandas dataframe
    '''

    stats = pd.read_excel(FILE_PATH + "Dataset_NOPOR.xlsx", sheet_name=sheet_name)
    stats = pd.concat([stats, pd.get_dummies(stats['Ruolo'], prefix='Ruolo')], axis=1)

    return stats


def split_and_std(stats):
    '''
        Una volta rimossi i parametri letterali dal dataframe, la funzione divide "stats" in train e test set e
        standardizza i valori delle feature.

    :param      stats: pandas dataframe
    :return:    X_train_std, X_test_std, Y_train, Y_test: list
    '''

    stats = stats.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

    X = stats[["Partite_giocate", "PG_Titolare", "Min_giocati", "Min_90", "Reti", "Assist", "Reti_no_rig", "Reti_rig",
               "Rig_tot", "Amm", "Esp", "Reti_90", "Assist_90", "Compl", "Tent", "%Tot", "Dist", "Dist_prog",
               "Pass_Assist", "Pass_tiro", "Pass_terzo", "Pass_area", "Cross_area", "Pass_prog", "Tocchi", "Drib_vinti",
               "Drib_tot", "%Drib_vinti", "Giocatori_sup", "Tunnel", "Controlli_palla", "Dist_controllo",
               "Dist_controllo_vs_rete", "Prog_controllo_area_avv", "Controllo_area", "Controllo_perso",
               "Contrasto_perso", "Dest", "Ricevuti", "Ricevuti_prog", "Tiri_reti", "Tiri", "Tiri_specchio",
               "%Tiri_specchio", "Tiri_specchio_90", "Goal_tiro", "Dist_avg_tiri", "Tiri_puniz", "Contr", "Contr_vinti",
               "Dribbl_blocked", "Dribbl_no_block", "Dribbl_sub", "%Dribbl_blocked", "Press", "Press_vinti",
               "%Press_vinti",
               "Blocchi", "Tiri_block", "Tiri_porta_block", "Pass_block", "Intercett", "Tkl_Int", "Salvat",
               "Err_to_tiro",
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


def get_team(sheet_name):
    '''
        Legge i nomi da un file txt, li estrae dal dataframe completo, li salva in un secondo dataframe.
        Crea un terzo dataframe senza valori letterali per standardizzarne i valori.

    :param      sheet_name:        string
    :return:    df_my_team_std:    transformed array
    :return:    df_my_team_full:   pandas dataframe
    '''

    df_my_team_names = pd.read_csv(FILE_PATH + "My_team_NOPOR.txt", sep=',', header=None,
                                   names=["Nome_Cognome", "Squadra"])

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
