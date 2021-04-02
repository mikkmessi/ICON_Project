'''
Created on 10 mar 2021

@author: kecco
'''
import performance as p

current_fb_day = 27             # giornata su cui fare il learning e predire i voti della giornata successiva (27)
next_fb_day = 28                # giornata successiva di cui predire i voti

#file_path_1 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"

#file_path_2 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g27_NOPOR.xlsx"

# file_path = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\"

dataset = p.load_and_model("g26")

X_train, X_test, Y_train, Y_test = p.split_and_std(dataset)
 
best_model = p.batch_classify(X_train, Y_train, X_test, Y_test)

my_team, my_team_full = p.team("g27")  # current_fb_day                            # 25 players of user's team

prediction_list = list(best_model.predict(my_team))

df_final_score = p.final_score(my_team_full, prediction_list, next_fb_day, sheet_name="g27")
print(df_final_score)
print()

df_final_score_gk = p.best_goalkeeper()

best_team, best_module = p.best_eleven(df_final_score, df_final_score_gk)

print("Il modulo migliore per la prossima partita sembrerebbe essere il modulo ", best_module, "con la formazione: ")
print()
print(best_team)
