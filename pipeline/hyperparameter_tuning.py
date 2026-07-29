from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# Since we have small set of parameters so we considered GridSearchCV for tuning
class ModelTuning:
    def __init__(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def hyper_param_rf(self):
        param_grid_rf={'n_estimators': [100,200,300],'max_depth': [10, 20],
                       'min_samples_split': [2, 5],'min_samples_leaf': [1, 2] }
        rf_model = RandomForestRegressor()
        grid_rf = GridSearchCV(estimator=rf_model,param_grid=param_grid_rf,cv=5,
                                   scoring='neg_root_mean_squared_error',verbose=2,n_jobs=-1)
        grid_rf.fit(self.X_train,self.y_train)
        best_params = grid_rf.best_params_
        best_rmse = -grid_rf.best_score_
        print("Random Forest")
        print(f"Best Parameters: {best_params}")
        print(f"Best RMSE: {best_rmse:.4f}")

    def hyper_param_ridge(self):
        param_grid_ridge = {'alpha': [0.01, 0.1, 1, 10, 100]}
        ridge = Ridge()
        grid_ridge = GridSearchCV(estimator=ridge,param_grid=param_grid_ridge,cv=5,
                                   scoring='neg_root_mean_squared_error',verbose=2,n_jobs=-1)
        grid_ridge.fit(self.X_train,self.y_train)
        best_alpha = grid_ridge.best_params_['alpha']
        best_rmse = -grid_ridge.best_score_
        print("Ridge Regression")
        print(f"Best alpha: {best_alpha}")
        print(f"Best RMSE: {best_rmse:.4f}")
