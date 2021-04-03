import performance as p
import testing as test

TRAIN_FB_DAY = 26                                                           # COSTANTE

dataset_train = p.load_and_model(str(TRAIN_FB_DAY))

X_train, X_test, Y_train, Y_test = p.split_and_std(dataset_train)

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

print("SCORE-------------------------------")
print("Best Model no PCA: \t\t\tTrain score: ", dict_best_model['train_score'], "\tTest score: ", dict_best_model['test_score'],
      "\nBest Model with PCA: \t\tTrain score: ", dict_best_model_pca['train_score'], "\tTest score: ", dict_best_model_pca['test_score'])

if dict_best_model['name'] == 'Random Forest':
    print("Best Model HT no PCA: \tRS -> Train score: ", rs_score_train, "\tTest score: ", rs_score_test,
                                  "\tGS ->\t\tTrain score: ", gs_score_train, "\tTest score: ", gs_score_test,
          "\nBest Model HT with PCA: \tRS ->\tTrain score: ", rs_score_pca_train, "\tTest score: ", rs_score_pca_test,
                                    "\tGS ->\tTrain score: ", gs_score_pca_train, "\tTest score: ", gs_score_pca_test)
print("SCORE END---------------------------")
