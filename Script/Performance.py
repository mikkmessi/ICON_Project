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

n_players = 3

def load_and_model(file_path):
    
    #"C:\\Users\\kecco\\Documents\\GitHub\\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"
    stats = pd.read_excel(file_path)
    modeling = stats.copy()
    modeling = pd.concat([modeling,pd.get_dummies(modeling['Ruolo'],prefix='Ruolo')],axis=1)
    modeling = modeling.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)
    
    return modeling
    
def split(modeling):
    
    X = modeling[["Partite_giocate", "PG_Titolare",    "Min_giocati", "Min_90", "Reti", "Assist", "Reti_no_rig", "Reti_rig",
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
              "Falli_gol", "Azioni_dif_gol", "Azioni_Autogol","Ruolo_Att","Ruolo_Dif","Ruolo_Cen","Ruolo_CenAtt",
              "Ruolo_AttCen","Ruolo_DifAtt","Ruolo_DifCen","Ruolo_CenDif","Ruolo_AttDif"]].values
        
    Y = modeling["Mf"].values
        
    # suddividiamo il dataseet in due dataset, uno di training ed uno di test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # standardizzo il train set creando un modello di standardizzazione
    X_train_std = ss.fit_transform(X_train)
    X_test_std = ss.transform(X_test)
    
    return X_train_std,X_test_std,Y_train,Y_test
    
#Creazione dizionario classificatori, che contiene come chiavi il nome dei class., come valori un istanza di essi
    
dict_classifiers = {
    
    "Linear Regression": LinearRegression(),
    "Naive Bayes": ARDRegression(),    
    "Random Forest": RandomForestRegressor(),
    
    }    


def batch_classify(X_train,Y_train,X_test,Y_test,no_classifiers= 3):
    
    dict_models = {}
    massimo = 0
    
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        
        classifier.fit(X_train,Y_train)
        train_score = classifier.score(X_train,Y_train)
        test_score = classifier.score(X_test,Y_test)
        mean = (train_score + test_score) / 2
        
        if mean > massimo :
            
            massimo = mean
            best_model = classifier
        
        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'media': mean}
        
    
    print(dict_models)
            
    return dict_models,best_model        
            
    
def team(file_path):
              
    ids = np.empty([n_players])
             
    for i in range (n_players) :
        ids[i] = input("Inserisci ID giocatore: ")
        
    stats_test = pd.read_excel(file_path)
    modeling_test = stats_test.copy()
    modeling_test = pd.concat([modeling_test,pd.get_dummies(modeling_test['Ruolo'],prefix='Ruolo')],axis=1)

    my_team = pd.DataFrame()
    
    for i in range(n_players):
        my_team = my_team.append(modeling_test.loc[modeling_test['ID'] == ids[i]])

    my_team_c = my_team.copy()
    my_team = my_team.drop(['ID','Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)
    
    my_team_std = ss.transform(my_team)
        
    return my_team_std,my_team_c

            
            