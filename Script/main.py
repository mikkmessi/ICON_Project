'''
Created on 10 mar 2021

@author: kecco
'''
import performance as p

match_day = 27

#file_path_1 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"

#file_path_2 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g27_NOPOR.xlsx"

file_path_1 = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"

file_path_2 = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\Dataset_g27_NOPOR.xlsx"

dataset = p.load_and_model(file_path_1)

X_train_std, X_test_std, Y_train, Y_test = p.split(dataset)
 
dict_models, best_model = p.batch_classify(X_train_std, Y_train, X_test_std, Y_test)

my_team, my_team_full = p.team(file_path_2)

prediction_list = list(best_model.predict(my_team))

df_final_score = p.final_score(my_team_full, prediction_list, match_day)
print(df_final_score)
print()

# print the dataframe in a descending order by "Final_score" values
df_final_score = df_final_score.sort_values(by=["Final_score"], ascending=False)
print(df_final_score)
