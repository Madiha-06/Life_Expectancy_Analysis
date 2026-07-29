import sklearn
from sklearn.model_selection import train_test_split
from pipeline.base import Base


class DataPartition(Base):
    def __init__(self,dataset,target_col):
        super().__init__(dataset,target_col)
# Splitting into features and target
    def split_X(self):
        self.x=self.dataset.drop(columns=[self.target_col])
        return self.x
    def split_y(self):
        self.y=self.dataset[self.target_col]
        return self.y
# Splitting into training and test data
    def train_test_splitting(self,x,y):
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        return X_train, X_test, y_train, y_test
