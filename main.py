# importing all the necessary pipelines
from pipeline.data_cleaning import DataCleaning
from pipeline.data_collection import DataCollector
from pipeline.data_splitting import DataPartition
from pipeline.data_eda import EDA
from pipeline.data_transformation import DataTransformation
from pipeline.feature_selection import FeatureSelection
from pipeline.hyperparameter_tuning import ModelTuning
from pipeline.model_training import ModelTraining
from pipeline.result_visualization import ResultVisualization

link=r'C:\Users\madih\OneDrive\Desktop\Life Expectancy Data.csv'

data=DataCollector(link)
df=data.data_collect()
print("Data loaded")
print("Head")
print(df.head())
print('Tail')
print(df.tail())
target_col='Life expectancy '

# handling missing values
nan_val=DataCleaning(df,target_col)
nan_val.analyze_nan()
# Since data is positive/negative skewed we will use median to handle missing values
clean_data=nan_val.handling_nan('median')

#Exploratory Data Analysis
df_eda=EDA(df,target_col)
print('Data Information')
print(df_eda.data_info())
print('Target Information')
print(df_eda.target_info())
# Correlation between Life expectancy and other features (helpful for feature selection)
print('Correlation heatmap')
df_eda.target_correlation()
cat_col=df_eda.categorical_columns()
num_col,disc_col,cont_col=df_eda.numerical_columns()
print('Categorical Columns:',cat_col)
print("Numerical Columns:",num_col)
print("Disc Columns:",disc_col)
print("Cont Columns:",cont_col)
#Relationship between Life expectancy and other features
df_eda.avg_target_features(cat_col,disc_col,cont_col)

# Data Transformation
df_to_transform=DataTransformation(clean_data,target_col)
# encoding categorical variables. Here we have 'Status' column which will have developed=1 and developing=0 values
df_encoded=df_to_transform.cat_encoding('Status','Developed','Developing')
df_to_scaled=DataTransformation(df_encoded,target_col)
# Scaling the numerical columns
transformed_data=df_to_scaled.scaling()

# Combining columns
mortality_cols=['Adult Mortality','infant deaths','under-five deaths ']
transformed_data['Avg Mortality']=transformed_data[mortality_cols].mean(axis=1)
#Further drop: separate mortality features
# Data Partition
partition=DataPartition(transformed_data,target_col)
X=partition.split_X()
y=partition.split_y()

# Feature Selection (Using correlation)
selection=FeatureSelection(X,y)
print("Feature Selection")
selected=selection.select_features()
# Since thinness(1-19) and thinness(1-5) serving same purpose has nearly same correlation we'll drop thinness(5-9)
# 'Country' has ordered as each country is repeated sequentially from 2000-2015 therefore it will also be dropped.Thus not considered beforehand
X_features=X[selected]
X_features=X_features.drop(columns=[' thinness 5-9 years','Adult Mortality','infant deaths','under-five deaths '] )

#Splitting data for model training
X_train,X_test,y_train,y_test=partition.train_test_splitting(X_features,y)
model=ModelTraining(X_train,X_test,y_train,y_test)
# Model Evaluation,Saving and Comparison
#Random Forest Regression
prediction_rf=model.random_forest()
#Linear Regression
prediction_lr=model.linear_regression()

#Hyperparameter Tuning and Results
tune=ModelTuning(X_train,y_train)
# Random Forest
tune.hyper_param_rf()
#Linear Regression (Ridge Regression)
tune.hyper_param_ridge()

plots=ResultVisualization(y_test)
# For Random Forest
plots.random_forest_plot(prediction_rf)
# For Linear regression
plots.linear_regression_plot(prediction_lr)
