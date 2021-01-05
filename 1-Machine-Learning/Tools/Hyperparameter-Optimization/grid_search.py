import lightgbm as lgb
from sklearn.model_selection import GridSearchCV


def GridSearch(clf, params, X, y):
    cscv = GridSearchCV(clf, params, scoring='neg_mean_squared_error', n_jobs=1, cv=5)
    cscv.fit(X, y)

    print(cscv.cv_results_)
    print(cscv.best_params_)


if __name__ == '__main__':
    train_X, train_y = get_data()

    param = {
        'objective': 'regression',
        'n_estimators': 275,
        'max_depth': 6,
        'min_child_samples': 20,
        'reg_lambd': 0.1,
        'reg_alpha': 0.1,
        'metric': 'rmse',
        'colsample_bytree': 1,
        'subsample': 0.8,
        'num_leaves' : 40,
        'random_state': 2018
        }
    regr = lgb.LGBMRegressor(**param)

    adj_params = {'n_estimators': range(100, 400, 10),
                 'min_child_weight': range(3, 20, 2),
                 'colsample_bytree': np.arange(0.4, 1.0),
                 'max_depth': range(5, 15, 2),
                 'subsample': np.arange(0.5, 1.0, 0.1),
                 'reg_lambda': np.arange(0.1, 1.0, 0.2),
                 'reg_alpha': np.arange(0.1, 1.0, 0.2),
                 'min_child_samples': range(10, 30)}

    GridSearch(regr , adj_params , train_X, train_y)