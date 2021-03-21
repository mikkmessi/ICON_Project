'''
Created on 23 feb 2021

@author: kecco
'''

'''
import pyodbc

conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:\\Users\\kecco\\Desktop\\Database_g26.accdb;')
cursor = conn.cursor()
cursor.execute('select * from Giocatori')

mas=0

for row in cursor.fetchall():
    if (float(row.Partite_giocate) > mas):
        mas=float(row.Partite_giocate)

print(mas)

'''
# import requests as req
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model

file_path = "C:\\Users\\kecco\\Documents\\GitHub\\ICON_Project\\Dataset\\Dataset_g26.xlsx"
stats = pd.read_excel(file_path)

modeling = stats.copy()
modeling = pd.concat([modeling,pd.get_dummies(modeling['Ruolo'],prefix='Ruolo')],axis=1)
modeling = modeling.drop(['ID', 'Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)

'''
corr = modeling.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f,ax = plt.subplots(figsize=(22, 18))
cmap = sns.diverging_palette(0, 150, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
'''

# associamo a X i valori di input di tutte le colonne
# associamo a Y i valori di output

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
              "Falli_gol", "Azioni_dif_gol", "Azioni_Autogol","Ruolo_Att","Ruolo_Dif","Ruolo_Cen","Ruolo_Por","Ruolo_CenAtt","Ruolo_AttCen",
              "Ruolo_DifAtt","Ruolo_DifCen","Ruolo_CenDif","Ruolo_AttDif"]].values
Y = modeling["Mf"].values

# suddividiamo il dataseet in due dataset, uno di training ed uno di test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)


# standardizzo il train set creando un modello di standardizzazione
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# istanziamo la classe di calcolo della regressione lineare di SciKitLearn
# la addestriamo e prediciamo i valori con il set di test
lRegr = linear_model.LinearRegression()
lRegr.fit(X_train_std, Y_train)
Y_pred = lRegr.predict(X_test_std)

# calcoliamo l'errore quadratico medio e il coefficiente di determinazione
errore = mean_squared_error(Y_test, Y_pred)
print("Errore: ", errore)
score = r2_score(Y_test, Y_pred)
print("Score: ", score)
print(Y_pred)

print("Valore del bias: ", lRegr.intercept_)

# visualizziamo i valore dei pesi e del bias trovati
# for i in lRegr.coef_:
#      print("Valore del peso i-esimo: ",i)

stats_test = pd.read_excel("C:\\Users\\kecco\Documents\\GitHub\\ICON_Project\\Dataset\\Dataset_g27.xlsx", sheet_name="TuttoInsiemeOrdinato")
modeling_test = stats_test.copy()
modeling_test = pd.concat([modeling_test,pd.get_dummies(modeling_test['Ruolo'],prefix='Ruolo')],axis=1)

myTeam = modeling_test.loc[(modeling_test['ID'] == 373) | (modeling_test['ID'] == 464) | (modeling_test['ID'] == 402) | (modeling_test['ID'] == 374) | 
                           (modeling_test['ID'] == 77) | (modeling_test['ID'] == 43) | (modeling_test['ID'] == 28) | (modeling_test['ID'] == 576) |
                           (modeling_test['ID'] == 338) | (modeling_test['ID'] == 416)]

#print(myTeam)
myTeam = myTeam.drop(['ID','Nome_Cognome', 'Ruolo', 'Squadra'], axis=1)


# standardizzo il set creando un modello di standardizzazione

myTeam_std = ss.transform(myTeam)
prediction = lRegr.predict(myTeam_std)

print(prediction)






