'''
    Questo script contiene l'algoritmo per allenare il modello di Random Forest sui dataset dei giocatori e dei portieri
    per calcolarne quindi la predizione della prestazione e stampare a schermo la formazione consigliata.

    Team: Michele Messina, Francesco Zingariello
'''
import line_up as lu
import testing as test

TRAIN_FB_DAY = 26                                                           # COSTANTE
rfr = test.RandomForestRegressor(warm_start=True)

dataset = lu.load_and_model(str(TRAIN_FB_DAY))

X_train, X_test, Y_train, Y_test = lu.split_and_std(dataset)

print("Training giocatori...")
rfr.fit(X_train, Y_train)

next_fb_day = int(input("Inserire la giornata per cui predire la formazione: "))
current_fb_day = next_fb_day - 1   # ultima giornata di campionato giocata, usata per predire i voti

my_team, my_team_full = lu.get_team(str(current_fb_day))                     # 25 players of user's team

prediction_list = list(rfr.predict(my_team))

df_final_score = lu.final_score(my_team_full, prediction_list, next_fb_day, "Dataset_NOPOR.xlsx", sheet_name=str(current_fb_day))

print("Training e prediction portieri...")
df_final_score_gk = lu.best_goalkeeper(TRAIN_FB_DAY, next_fb_day, str(current_fb_day))

best_team, best_module = lu.best_eleven(df_final_score, df_final_score_gk)

print()
print("Il modulo migliore per la prossima partita sembrerebbe essere il modulo ", best_module, "con la formazione: ")
print()
print(best_team)

rfr.n_estimators += 100                                                 # Simula l'aggiornamento del modello in seguito all'aggiunta di nuovi dati
rfr.fit(X_test, Y_test)
