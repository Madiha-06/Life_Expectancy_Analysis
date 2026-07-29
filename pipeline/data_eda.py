from pipeline.base import Base
from pipeline.data_splitting import DataPartition
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDA(Base):
    def __init__(self,dataset,target_col):
        super().__init__(dataset,target_col)

    def data_info(self):
        print("Dataset Information")
        print(self.dataset.columns)
        print(self.dataset.info())
        print(self.dataset.describe())

    def target_info(self):
        print("Target Information")
        print(self.dataset[self.target_col].describe())
        # total values
        print(self.dataset[self.target_col].value_counts())

    def target_correlation(self):
        # correlation w.r.t target only
        cor=self.dataset.corr(numeric_only=True)[self.target_col]
        sns.heatmap(cor.to_frame(),annot=True)
        plt.show()
        return cor
# Separation of data into different categories
    def categorical_columns(self):
        categorical_features = [feature for feature in self.dataset.columns if self.dataset[feature].dtypes=='str']
        return categorical_features

    def numerical_columns(self):
        num_features = [feature for feature in self.dataset.columns if self.dataset[feature].dtypes!='str']
        discrete_features=[]
        continuous_features=[]
        for feature in num_features:
            unique_count = self.dataset[feature].nunique()
            if unique_count<=16:
                discrete_features.append(feature)
            else:
                continuous_features.append(feature)
        return num_features,discrete_features,continuous_features

    def avg_target_features(self,categorical_features,discrete_features,continuous_features):
        for feature in discrete_features + categorical_features:
            if feature!= 'Country':
              plt.figure(figsize=(15,12))
            # categorical/discrete: bar chart of median target
              grouped = self.dataset.groupby(feature)[self.target_col].median()
              grouped.plot.bar(color=plt.cm.tab20(range(len(grouped))))
              plt.ylabel(f'Median {self.target_col}')
              plt.xlabel(feature)
              plt.show()
        for feature in continuous_features:
        # continuous: scatter plot instead
            plt.figure(figsize=(15,12))
            plt.scatter(self.dataset[feature],self.dataset[self.target_col],color='blue',alpha=0.7,label=feature)
            plt.ylabel(self.target_col)
            plt.xlabel(feature)
            plt.title(feature)
            plt.tight_layout()
            plt.show()







