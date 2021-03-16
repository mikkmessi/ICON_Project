'''
Created on 23 feb 2021

@author: kecco
'''
from pickle import TRUE

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

stats=pd.read_excel("C:\\Users\kecco\\Desktop\\Dataset\\TuttoInsiemeOrdinato.xlsx")

modeling=stats.copy()
modeling=modeling.drop(['Giocatori_ID','TuttoInsiemeOrdinato_Nome_Cognome','Ruolo','Squadra'], axis=1)

corr=modeling.corr()
mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f,ax=plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(0,150,as_cmap=True)
sns.heatmap(corr,mask=mask,cmap=cmap,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5})
plt.show()

