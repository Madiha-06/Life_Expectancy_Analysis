import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
import joblib
from sklearn.preprocessing import StandardScaler
from pipeline.data_splitting import DataPartition


class ModelTraining:
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
#Training as well as saving the model
    def random_forest(self):
        rf_model = RandomForestRegressor(n_estimators=200,max_depth=None,random_state=42,n_jobs=-1)
        rf_model.fit(self.X_train,self.y_train)
        joblib.dump(rf_model, "rf_cost_predictor.pkl")
        print("Random Forest model saved as rf_cost_predictor.pkl")
        y_pred_rf= rf_model.predict(self.X_test)
        rmse_rf = np.sqrt(mean_squared_error(self.y_test, y_pred_rf))
        mae_rf = np.mean(np.abs(self.y_test - y_pred_rf))
        r2_rf = rf_model.score(self.X_test,self.y_test)
        print(f"Test Set Performance for Random Forest Regression:")
        print(f"RMSE: {rmse_rf:.4f} years")
        print(f"MAE: {mae_rf:.4f} years")
        print(f"R² Score: {r2_rf:.4f} ")
        return y_pred_rf

    def linear_regression(self):
        lr_model = LinearRegression()
        lr_model.fit(self.X_train,self.y_train)
        joblib.dump(lr_model, "lr_cost_predictor.pkl")
        print("Linear Regression model saved as lr_cost_predictor.pkl")
        y_pred_lr = lr_model.predict(self.X_test)
        rmse_lr = np.sqrt(mean_squared_error(self.y_test, y_pred_lr))
        mae_lr = np.mean(np.abs(self.y_test - y_pred_lr))
        r2_lr = lr_model.score(self.X_test,self.y_test)
        print(f"Test Set Performance Test Set Performance for Linear Regression:")
        print(f"RMSE: {rmse_lr:.4f} years")
        print(f"MAE: {mae_lr:.4f} years")
        print(f"R² Score: {r2_lr:.4f}")
        return y_pred_lr

