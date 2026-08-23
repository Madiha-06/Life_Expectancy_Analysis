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
unclean_val=DataCleaning(df,target_col)
# Separating column according to types initially
df_eda=EDA(df,target_col)
cat_col=df_eda.categorical_columns()
num_col,disc_col,cont_col=df_eda.numerical_columns()
print('Categorical Columns:',cat_col)
print("Numerical Columns:",num_col)
print("Disc Columns:",disc_col)
print("Cont Columns:",cont_col)

unclean_val.look_skewness()
unclean_val.analyze_nan()
#Looking and handling outliers
unclean_val.look_outliers(num_col)
# In order to verify outliers we performed following operations with features
#df[df['Life expectancy '] < 44.2][['Country','Year','Life expectancy ','Adult Mortality',' HIV/AIDS','infant deaths','GDP','Schooling']]
#df[df['Adult Mortality']>459][['Country','Year','Adult Mortality','Life expectancy ','Total expenditure',' HIV/AIDS','Schooling']]
#df[df['infant deaths']>55][['Country','Year','infant deaths','Adult Mortality','GDP','Schooling','Hepatitis B',' HIV/AIDS']].sort_values('infant deaths', ascending=False)
#df[df['percentage expenditure']>1096][['Country','Year','percentage expenditure','GDP']] ( will be dropped due to mismatched scaling)
#df[df['Hepatitis B'] < 47][['Country','Status','Year','Hepatitis B','Life expectancy ','Adult Mortality','GDP','Schooling']].sort_values('Hepatitis B')
#df[df['Measles ']>900.625][['Country','Year','Status','Measles ','Life expectancy ','infant deaths','under-five deaths ','Hepatitis B','Polio','GDP','Schooling']].sort_values('Measles ', ascending=False)
#df[df['under-five deaths ']>70][['Country','Year','Status','under-five deaths ','Life expectancy ','Polio','Measles ','GDP','Schooling']].sort_values('under-five deaths ', ascending=False)
#df[df['Polio'] < 49.5][['Country','Year','Polio','Life expectancy ','Adult Mortality','Measles ','GDP','Schooling']].sort_values('Polio',ascending=False)
#df[df['Country']=='India'][['Year','Population']].sort_values('Year') (population will also be dropped due to mismatched scaling)
#df[df[' thinness  1-19 years'] > 15.59][['Country','Year',' thinness  1-19 years','Life expectancy ','GDP','Schooling','Adult Mortality']].sort_values(' thinness  1-19 years', ascending=False)
#(df['Income composition of resources'] == 0).sum()
#df[df['Income composition of resources']==0][['Country','Year','Income composition of resources','GDP','Schooling','Life expectancy ']].sort_values('Country')
#(df['Schooling']==0).sum()
#df[df['Schooling']==0][['Country','Year','Schooling','Income composition of resources','GDP','Life expectancy ']].sort_values('Country')
# for country in ['China', 'United States of America', 'Nigeria', 'Brazil', 'Pakistan', 'Bangladesh','India']:
#     print(country)
#     print(df[df['Country']==country][['Year','Adult Mortality']].sort_values('Year'))
#     print()
#df[df['Adult Mortality'] < 50]['Year'].value_counts().sort_index()
unclean_val.handling_outliers()
# Since data is positive/negative skewed we will use median to handle missing values
clean_data=unclean_val.handling_nan('median')
# 'Country' is ordered as each country is repeated sequentially from 2000-2015 therefore it will also be dropped.Thus not considered beforehand
clean_data.drop(columns=['Population','percentage expenditure','Country'])
#Exploratory Data Analysis
print('Data Information')
print(df_eda.data_info())
print('Target Information')
print(df_eda.target_info())
#Raw relationship between Life expectancy and other features
df_eda.raw_rel_target_features(cat_col,disc_col,cont_col)
correlation=df_eda.target_correlation()
df_eda.correlated_feature_rel()
df_eda.yearly_trend()


# Data Transformation
df_to_transform=DataTransformation(clean_data,target_col)
# Here we have 'Status' column which will have developed=1 and developing=0 values
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
