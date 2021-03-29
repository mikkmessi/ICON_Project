'''
Created on 10 mar 2021

@author: kecco
'''
import performance as p

#file_path_1 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"

#file_path_2 = "C:\\Users\kecco\\Documents\\GitHub\ICON_Project\\Dataset\\Dataset_g27_NOPOR.xlsx"

file_path_1 = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\Dataset_g26_NOPOR.xlsx"

file_path_2 = "D:\\UniDispense\\ICON\\ICON_Project\\Dataset\\Dataset_g27_NOPOR.xlsx"

dataset = p.load_and_model(file_path_1)

X_train_std, X_test_std, Y_train, Y_test = p.split(dataset)
 
dict_models, best_model = p.batch_classify(X_train_std, Y_train, X_test_std, Y_test)

my_team, copia = p.team(file_path_2)

prediction = list(best_model.predict(my_team))

weights = {}

for player in list(copia["ID"]):
    weights[player] = p.final_weight(player, 27)

print(copia)
print(prediction)
print(weights)