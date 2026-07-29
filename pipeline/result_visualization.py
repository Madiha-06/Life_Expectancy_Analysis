import matplotlib.pyplot as plt
from pipeline.model_training import ModelTraining

class ResultVisualization:
    def __init__(self,y_test):
        self.y_test = y_test

    def random_forest_plot(self,y_pred_rf):
      plt.figure(figsize=(7, 6))
      plt.scatter(self.y_test,y_pred_rf,alpha=0.5,color='green')
      plt.plot([self.y_test.min(),self.y_test.max()],[self.y_test.min(),self.y_test.max()],'r--', lw=2)
      plt.xlabel('Actual Life Expectancy')
      plt.ylabel('Predicted Life Expectancy')
      plt.title('Random Forest: Actual vs Predicted')
      plt.show()

    def linear_regression_plot(self,y_pred_lr):
        plt.figure(figsize=(7, 6))
        plt.scatter(self.y_test,y_pred_lr,alpha=0.5,color='blue')
        plt.plot([self.y_test.min(),self.y_test.max()],[self.y_test.min(),self.y_test.max()],'r--', lw=2)
        plt.xlabel('Actual Life Expectancy')
        plt.ylabel('Predicted Life Expectancy')
        plt.title('Linear Regression: Actual vs Predicted')
        plt.show()