from sklearn.preprocessing import StandardScaler
import pandas as pd
from pipeline.base import Base

class DataTransformation(Base):
    def __init__(self, dataset,target_col):
        super().__init__(dataset,target_col)

    def cat_encoding(self,col,var1,var2):
        self.dataset[col + '_encoded'] = self.dataset[col].map({var1: 1, var2: 0})
        self.dataset.drop(columns=[col])
        return self.dataset

# Scaling features only
    def scaling(self):
        scaler = StandardScaler()
        df_scaled=self.dataset.copy()
        numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns
        cols_to_scale=[col for col in numeric_cols if col != self.target_col]
        df_scaled[cols_to_scale]=scaler.fit_transform(df_scaled[cols_to_scale])
        print("Numeric columns scaled successfully.")
        print(f"Scaled columns: {len(numeric_cols)}")
        print(f"Final DataFrame shape: {df_scaled.shape}")
        return df_scaled




