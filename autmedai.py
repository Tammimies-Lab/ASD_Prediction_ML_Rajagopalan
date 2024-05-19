'''
This is machine learning model development code accompanying the following paper
Title: AutMedAI: Predicting Autism from a Minimal Set of Medical and Background Information using Machine Learning
Authors: Shyam Sundar Rajagopalan, Yali Zhang, Ashraf Yahia, and Kristiina Tammimies
Correspondence: Dr. Kristiina Tammimies, kristiina.tammimies@ki.se
'''


from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import pickle
from hyperopt import tpe, Trials, hp, fmin, STATUS_OK, space_eval
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


### Compute confusion metrics
def confusion_metrics(conf_matrix):
    # save confusion matrix and slice into four pieces
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    # calculate precision
    conf_precision = (TP / float(TP + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))

    return conf_accuracy, conf_sensitivity, conf_specificity, conf_f1

### Objective function used for Bayesian optimization
def bo_objective(params, X_train, y_train, X_val, y_val):

    model = XGBClassifier(seed=0, **params)

    # if using val set
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred_val, pos_label=1)
    score = metrics.auc(fpr, tpr)

    # Loss is negative score
    loss = - score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

### Hyperparameter search using Bayesian Optimization
def get_optimal_param(space, X_tr, y_tr, X_valid, y_valid):

    # Optimize
    rstate = np.random.default_rng(0)
    fmin_objective = partial(bo_objective, X_train=X_tr, y_train=y_tr, X_val=X_valid, y_val=y_valid)
    best_idx = fmin(fn=fmin_objective, space=space, algo=tpe.suggest, max_evals=48, trials=Trials(),rstate=rstate)

    return best_idx


### Model training and cross-validation
def cross_validation(_X, _y):
    results = {}
    train_acc_list, train_auc_list, test_acc_list, test_auc_list = ([] for i in range(4))
    train_precision_list, train_recall_list, train_f1_list = ([] for i in range(3))
    test_precision_list, test_recall_list, test_f1_list = ([] for i in range(3))
    test_sensitivity_list, test_specificity_list = ([] for i in range(2))
    model_list = []
    transforms_list = []

    # 10-fold cross-validation
    cv = 10
    kf = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=1)
    for fold, (train_index, test_index) in enumerate(kf.split(_X, _y), 1):
        print('Cross validation ', fold)
        X_train = _X[train_index]
        y_train = _y[train_index]
        X_test = _X[test_index]
        y_test = _y[test_index]

        # generate val set from train set 0.25 * 0.8 = 0.2
        X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                                          random_state=1)  # 0.25 x 0.8 = 0.2

        # Data transformation for numerical and categorical variables.
        # define column indexes for the variables with "numerical" and "categorical" values
        category_col_ix = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 26, 27]
        numeric_col_ix = [1,  14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        # define transforms
        train_test_transforms = ColumnTransformer(transformers=[('standard_scaler', StandardScaler(), numeric_col_ix),
                                                     ('onehot_encoder', OneHotEncoder(handle_unknown="ignore"),
                                                      category_col_ix)],
                                       remainder='passthrough')
        # transform data
        X_train_transformed = train_test_transforms.fit_transform(X_train)
        X_test_transformed = train_test_transforms.transform(X_test)


        # define transforms for validation set
        train_val_transforms = ColumnTransformer(transformers=[('standard_scaler', StandardScaler(), numeric_col_ix),
                                                                ('onehot_encoder',
                                                                 OneHotEncoder(handle_unknown="ignore"),
                                                                 category_col_ix)],
                                                  remainder='passthrough')
        # transform for validation set used for hyperparameter tuning
        X_train_new = train_val_transforms.fit_transform(X_train_new)
        X_val = train_val_transforms.transform(X_val)


        # Bayesian optimization
        # search space for BO
        space = {
            'learning_rate': hp.choice('learning_rate', [0.0001, 0.001, 0.01, 0.1, 1]),
            'max_depth': hp.choice('max_depth', range(3, 21, 3)),
            'gamma': hp.choice('gamma', [i / 10.0 for i in range(0, 5)]),
            'colsample_bytree': hp.choice('colsample_bytree', [i / 10.0 for i in range(3, 10)]),
            'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100]),
            'n_estimators': hp.choice('n_estimators', range(0, 100, 10))
        }
        best_idx = get_optimal_param(space, X_train_new, y_train_new, X_val, y_val)
        best_param = space_eval(space, best_idx)
        print(best_param)

        # Train model using the best parameters
        model_optim = XGBClassifier(seed=0,
                              colsample_bytree=best_param['colsample_bytree'],
                              gamma=best_param['gamma'],
                              learning_rate=best_param['learning_rate'],
                              max_depth=best_param['max_depth'],
                              reg_alpha=best_param['reg_alpha'],
                              reg_lambda=best_param['reg_lambda'],
                              n_estimators=best_param['n_estimators'],
                              early_stopping_rounds=10
                              )

        eval_set = [(X_val, y_val)]

        # XGBoost model training
        model_optim.fit(X_train_transformed, y_train, eval_set=eval_set, verbose=True)

        # Model prediction
        y_pred_train = model_optim.predict(X_train_transformed)  # just for comparison purposes
        y_pred_test = model_optim.predict(X_test_transformed)

        # Calculate metrics
        train_scores = precision_recall_fscore_support(y_train, y_pred_train, pos_label=1,
                                                       average='binary')  # scores for class 1 - ASD
        test_scores = precision_recall_fscore_support(y_test, y_pred_test, pos_label=1,
                                                      average='binary')  # scores for class 1 - ASD
        # accuracy
        train_acc = balanced_accuracy_score(y_train, y_pred_train)
        test_acc = balanced_accuracy_score(y_test, y_pred_test)

        # precision and recall
        test_precision = average_precision_score(y_test, model_optim.predict_proba(X_test_transformed)[:, 1], average=None)

        # test auc
        y_pred_prob = model_optim.predict_proba(X_test_transformed)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_prob)
        # train auc
        y_train_pred_prob = model_optim.predict_proba(X_train_transformed)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_pred_prob)

        # Creating the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred_test)
        accuracy, sensitivity, specificity, f1 = confusion_metrics(cm)

        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        test_acc_list.append(test_acc)
        test_auc_list.append(test_auc)
        train_precision_list.append(train_scores[0])
        train_recall_list.append(train_scores[1])
        train_f1_list.append(train_scores[2])
        test_precision_list.append(test_precision)
        test_recall_list.append(test_scores[1])
        test_sensitivity_list.append(sensitivity)
        test_specificity_list.append(specificity)
        test_f1_list.append(f1)
        model_list.append(model_optim)
        transforms_list.append(train_test_transforms)

    results['train_accuracy'] = train_acc_list
    results['train_auc'] = train_auc_list
    results['train_precision'] = train_precision_list
    results['train_recall'] = train_recall_list
    results['train_f1'] = train_f1_list
    results['test_accuracy'] = test_acc_list
    results['test_auc'] = test_auc_list
    results['test_precision'] = test_precision_list
    results['test_recall'] = test_recall_list
    results['test_f1'] = test_f1_list
    results['test_sensitivity'] = test_sensitivity_list
    results['test_specificity'] = test_specificity_list
    results['model_list'] = model_list
    results['transforms_list'] = transforms_list

    return results

### Main function having model development pipeline
def main():

    # load SPARK v8 pre-processed dataset
    spark_v8_data_processed = "v8_data_comp_all.pkl"
    X, y, features, _  = pickle.load(open(spark_v8_data_processed, 'rb'))

    # 10-fold cross-validation using XGBoost
    result = cross_validation(X, y)

    # print results
    cv = 10
    print("Mean test accuracy across %d folds: %2.3f" % (cv, np.mean(result['test_accuracy'])))
    # print("Std test accuracy across %d folds: %2.3f" % (cv, np.std(result['test_accuracy'])))
    print("Mean train auc across %d folds: %2.3f" % (cv, np.mean(result['train_auc'])))
    print("Mean test auc across %d folds: %2.3f" % (cv, np.mean(result['test_auc'])))
    print("Std test auc across %d folds: %2.3f" % (cv, np.std(result['test_auc'])))
    print("Mean test sensitivity across %d folds: %2.3f" % (cv, np.mean(result['test_sensitivity'])))
    print("Mean test specificity across %d folds: %2.3f" % (cv, np.mean(result['test_specificity'])))
    print("Mean test precision across %d folds: %2.3f" % (cv, np.mean(result['test_precision'])))
    # print("Mean test recall across %d folds: %2.3f" % (cv, np.mean(result['test_recall'])))
    print("Mean test f1 across %d folds: %2.3f" % (cv, np.mean(result['test_f1'])))

    print('Done')

### Program begins here
if __name__ == "__main__":
    main()
