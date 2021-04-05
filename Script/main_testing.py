'''
    Questo script viene utilzzato per testare e confrontare i modelli migliori a cui sottoporre il dataset per l'apprendimento.

    Team: Michele Messina, Francesco Zingariello
'''
import line_up as lu
import testing as test

TRAIN_FB_DAY = 26                                                           # COSTANTE

dataset_train = lu.load_and_model(str(TRAIN_FB_DAY))

X_train, X_test, Y_train, Y_test = lu.split_and_std(dataset_train)

print("Training e testing giocatori SENZA PCA: \n")
dict_best_model = test.best_regressor(X_train, Y_train, X_test, Y_test)
best_model = dict_best_model['model']

print("\nImportanze date dal modello alle feature nel dataset di training SENZA PCA: \n")
test.importance(best_model, dataset_train)

print("\nPCA sul dataset...\n")
X_train_pca, X_test_pca = test.principal_component_analysis(X_train, X_test)

print("\nTraining e testing giocatori CON PCA: \n")
dict_best_model_pca = test.best_regressor(X_train_pca, Y_train, X_test_pca, Y_test)
best_model_pca = dict_best_model_pca['model']

print("\nImportanze date dal modello alle feature nel dataset di training CON PCA: \n")
test.importance(best_model_pca, dataset_train)

if dict_best_model['name'] == 'Random Forest':
    print("\nHypertuning RANDOM FOREST SENZA PCA:\n")
    rs_score_train, rs_score_test, gs_score_train, gs_score_test = test.hypertuning(best_model, X_train, Y_train, X_test, Y_test)

    print("\nHypertuning RANDOM FOREST CON PCA:\n")
    rs_score_pca_train, rs_score_pca_test, gs_score_pca_train, gs_score_pca_test = test.hypertuning(best_model_pca, X_train_pca, Y_train, X_test_pca, Y_test)
    
dict_models = {'No PCA': [dict_best_model['train_score'], dict_best_model['test_score']],
               'PCA': [dict_best_model_pca['train_score'], dict_best_model_pca['test_score']]}

print("SCORE------------------------------------------------------------------")
print("{:<10}\t{:<23}\t{:<15}".format("Model", "Train score", "Test score"))
for key, values in dict_models.items():
    train_score, test_score = values
    print("{:<15}\t{:<9}\t{:<15}".format(key, train_score, test_score))

if dict_best_model['name'] == 'Random Forest':

    dict_models = {'HT RS': [rs_score_train, rs_score_test],
                   'HT GS': [gs_score_train, gs_score_test],
                   'HT PCA RS': [rs_score_pca_train, rs_score_pca_test],
                   'HT PCA GS': [gs_score_pca_train, gs_score_pca_test]}

    for key, values in dict_models.items():
        train_score, test_score = values
        print("{:<15}\t{:<9}\t{:<15}".format(key, train_score, test_score))
print("END--------------------------------------------------------------------")
