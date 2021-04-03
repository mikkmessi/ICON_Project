'''
Created on 10 mar 2021

@author: kecco
'''
import performance as p

TRAIN_FB_DAY = 26                                                           # COSTANTE

dataset = p.load_and_model(str(TRAIN_FB_DAY))

X_train, X_test, Y_train, Y_test = p.split_and_std(dataset)

print("Training e testing giocatori...")
best_model = p.best_regressor(X_train, Y_train, X_test, Y_test)

next_fb_day = int(input("Inserire la giornata per cui predire la formazione: "))
current_fb_day = next_fb_day - 1                                              # current football day, to use for predictions

my_team, my_team_full = p.team(str(current_fb_day))                     # 25 players of user's team

prediction_list = list(best_model.predict(my_team))

df_final_score = p.final_score(my_team_full, prediction_list, next_fb_day, "Dataset_NOPOR.xlsx", sheet_name=str(current_fb_day))

print("Training and testing portieri...")
df_final_score_gk = p.best_goalkeeper(TRAIN_FB_DAY, next_fb_day, str(current_fb_day))

best_team, best_module = p.best_eleven(df_final_score, df_final_score_gk)

print()
print("Il modulo migliore per la prossima partita sembrerebbe essere il modulo ", best_module, "con la formazione: ")
print()
print(best_team)
