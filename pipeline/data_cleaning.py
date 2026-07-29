import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import legend

from pipeline import data_splitting
from sklearn.impute import SimpleImputer

from pipeline.base import Base


class DataCleaning(Base):
   def __init__(self,dataset,target_col):
        super().__init__(dataset,target_col)

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
          for col in nan_sum.index:
            if nan_sum[col]>0:
              sns.histplot(self.dataset[col],bins=20)
              plt.axvline(self.dataset[col].median(),color='red',label='Median')
              plt.legend()
              plt.title(f'Histogram of {col}',size=16)
              plt.show()

   def handling_nan(self,strategy):
    numeric_cols = self.dataset.select_dtypes(include='number').columns
    if len(numeric_cols)==0:
      return None
    if self.dataset[numeric_cols].isna().sum().sum()>0:
      imp=SimpleImputer(strategy=strategy)
      data=imp.fit_transform(self.dataset[numeric_cols])
      self.dataset[numeric_cols] = pd.DataFrame(data, columns=numeric_cols, index=self.dataset.index)
      print(self.dataset[numeric_cols].isna().sum())
    return self.dataset




