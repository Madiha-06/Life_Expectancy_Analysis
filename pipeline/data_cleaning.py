import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import legend
from pipeline import data_splitting
from sklearn.impute import SimpleImputer
from pipeline.base import Base

class DataCleaning(Base):
   def __init__(self,dataset,target_col):
        super().__init__(dataset,target_col)

   def look_outliers(self, num_features):
       for feature in self.dataset.columns:
           if feature not in ['Country', 'Year', 'Status']:
               plt.figure(figsize=(5, 4))
               sns.boxplot(y=self.dataset[feature])
               plt.title(f'{feature} - Outliers')
               Q1 = self.dataset[feature].quantile(0.25)
               Q3 = self.dataset[feature].quantile(0.75)
               IQR = Q3 - Q1
               lower_bound = Q1 - 1.5 * IQR
               upper_bound = Q3 + 1.5 * IQR
               outliers = self.dataset[(self.dataset[feature] < lower_bound) | (self.dataset[feature] > upper_bound)]
               print(f'Number of outliers: {len(outliers)}')
               print(f'{feature} -> Lower Bound: {lower_bound} - Upper Bound: {upper_bound}')
               plt.show()

   def look_skewness(self):
       for feature in self.dataset.columns:
           if feature not in ['Country', 'Year', 'Status']:
               median_val = self.dataset[feature].median()
               plt.figure(figsize=(5, 4))
               plt.title(f'{feature} - skewness')
               plt.hist(self.dataset[feature])
               plt.axvline(x=median_val, color='red', linestyle='dashed', label=f'Median: {median_val:.1f}')
               plt.legend()
               plt.show()

   def analyze_nan(self):
       nan_sum=self.dataset.isna().sum()
       print(nan_sum)
       nan_val_cols=nan_sum[nan_sum>0]
       if nan_val_cols.empty:
          print("No missing values")
       else:
          nan_sum.plot(kind='bar')
          plt.title('Count of Values per column for Missing value Analysis', size=16)
          plt.show()

   def handling_nan(self, strategy):
       # Apply special-case fixes first
       self.dataset.loc[self.dataset['Adult Mortality'] < 50, 'Adult Mortality'] = np.nan
       self.dataset['Adult Mortality'] = self.dataset.groupby('Country')['Adult Mortality'].transform(
           lambda x: x.interpolate())
       self.dataset['Schooling'] = self.dataset['Schooling'].replace(0,np.nan)
       self.dataset['Income composition of resources'] = self.dataset['Income composition of resources'].replace(0,np.nan)
       numeric_cols = self.dataset.select_dtypes(include='number').columns
       if len(numeric_cols) == 0:
           return None
       if self.dataset[numeric_cols].isna().sum().sum() > 0:
           imp = SimpleImputer(strategy=strategy)
           data = imp.fit_transform(self.dataset[numeric_cols])
           self.dataset[numeric_cols] = pd.DataFrame(data, columns=numeric_cols, index=self.dataset.index)
           print(self.dataset[numeric_cols].isna().sum())
       return self.dataset

   def handling_outliers(self):
       self.dataset['Life expectancy '] = self.dataset['Life expectancy '].clip(lower=39, upper=89)
       # percentage based clipping
       for col in ['Hepatitis B','Polio','Diphtheria ']:
           self.dataset[col] = self.dataset[col].clip(lower=0, upper=100)
       return self.dataset


