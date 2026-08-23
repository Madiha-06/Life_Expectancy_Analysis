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
        # Specific correlation with health and socio-economic or socio-political issues
        health_cols = ['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 'Hepatitis B',
                       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
                       'Diphtheria ', ' HIV/AIDS', ' thinness  1-19 years', ' thinness 5-9 years']
        correlation_health = self.dataset[health_cols].corr()[self.target_col]
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_health.to_frame(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation: Life Expectancy vs Health Factors')
        plt.show()
        socio_cols = ['Life expectancy ', 'GDP', 'Schooling', 'Income composition of resources',
                      'Total expenditure']
        correlation_socio = self.dataset[socio_cols].corr()[self.target_col]
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_socio.to_frame(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlation: Life Expectancy vs Socioeconomic Factors')
        plt.show()
        correlation = self.dataset.corr(numeric_only=True)[self.target_col].abs() > 0.3
        print(correlation)
        return correlation

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

    def raw_rel_target_features(self,categorical_features,discrete_features,continuous_features):
        for feature in discrete_features + categorical_features:
            if feature!= 'Country':
              plt.figure(figsize=(6,4))
            # categorical/discrete: bar chart of median target
              grouped = self.dataset.groupby(feature)[self.target_col].median()
              grouped.plot.bar(color=plt.cm.tab20(range(len(grouped))))
              plt.ylabel(f'Median {self.target_col}')
              plt.xlabel(feature)
              plt.title('Raw relationship between features')
              plt.show()
        for feature in continuous_features:
        # continuous: scatter plot instead
            plt.figure(figsize=(6,4))
            plt.scatter(self.dataset[feature],self.dataset[self.target_col],color='blue',alpha=0.7,label=feature)
            plt.ylabel(self.target_col)
            plt.xlabel(feature)
            plt.title('Raw relationship between features')
            plt.tight_layout()
            plt.show()

    def correlated_feature_rel(self):
        corr_values = self.dataset.corr(numeric_only=True)[self.target_col]
        significant_features = corr_values[corr_values.abs() > 0.3].index.tolist()
        significant_features = [feature for feature in significant_features if feature != self.target_col]
        for feature in significant_features:
            plt.figure(figsize=(6, 4))
            sns.regplot(data=self.dataset, x=feature, y=self.target_col, scatter_kws={'alpha': 0.3})
            plt.title(f'{feature} vs Life Expectancy')
            plt.show()

    def yearly_trend(self):
      # Global trend of life expectancy
      global_yearly_trend = self.dataset.groupby('Year')['Life expectancy '].median()
      plt.figure(figsize=(6, 4))
      plt.plot(global_yearly_trend.index, global_yearly_trend.values, marker='o')
      plt.xlabel('Year')
      plt.ylabel('Median Life Expectancy')
      plt.title('Global Life Expectancy Trend (2000-2015)')
      plt.show()

      # Status wise trend of life expectancy
      yearly_by_status = self.dataset.groupby(['Year','Status'])['Life expectancy '].median().unstack()
      yearly_by_status.plot(figsize=(6, 4), marker='o')
      plt.xlabel('Year')
      plt.ylabel('Median Life Expectancy')
      plt.title('Life Expectancy Trend: Developed vs Developing')
      plt.show()













