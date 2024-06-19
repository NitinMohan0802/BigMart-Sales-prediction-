# BigMart-Sales-prediction-
# Nitin Mohan Notebook: Big Mart Sales Prediction
# Project Overview
This project aims to predict the sales of products across different outlets of a retail store using machine learning techniques. The dataset used contains various features related to the products and outlets, and the goal is to build a regression model to predict the sales (Item_Outlet_Sales).

Steps
1. Extracting the Compressed Dataset
Code:
python
Copy code
from zipfile import ZipFile
dataset = '/content/archive (2).zip'
with ZipFile(dataset,'r') as zip:
    zip.extractall()
Explanation: This step extracts the compressed dataset from a ZIP file, making it available for further processing.
2. Importing Libraries
Libraries Used:
numpy, pandas: Data manipulation and analysis.
matplotlib, seaborn: Data visualization.
sklearn.preprocessing.LabelEncoder, sklearn.model_selection.train_test_split, sklearn.metrics: Data preprocessing, splitting, and evaluation.
xgboost.XGBRegressor: Model building.
3. Data Collection & Analysis
Loading Data:

python
Copy code
big_mart_data = pd.read_csv('/content/Train.csv')
This step reads the dataset into a pandas DataFrame.

Exploratory Data Analysis (EDA):

Shape and Info:

python
Copy code
big_mart_data.shape
big_mart_data.info()
Provides an overview of the dataset's dimensions and data types.

Handling Missing Values:

python
Copy code
big_mart_data.isnull().sum()
Identifies columns with missing values.

Imputing Missing Values:

python
Copy code
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)
Fills missing numerical values with the mean and categorical values with the mode.

4. Data Analysis
Descriptive Statistics:

python
Copy code
big_mart_data.describe()
Provides summary statistics for numerical columns.

Visualization:

python
Copy code
sns.distplot(big_mart_data['Item_Weight'])
Various plots to understand the distribution and relationships of different features.

5. Data Preprocessing
Categorical Variable Encoding:

python
Copy code
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
Converts categorical variables into numerical format using Label Encoding.

Handling Inconsistent Labels:

python
Copy code
big_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
Ensures consistency in categorical labels.

6. Splitting Features and Target
Defining Features and Target:

python
Copy code
X = big_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = big_mart_data['Item_Outlet_Sales']
Separates the dataset into features (X) and target (Y).

Train-Test Split:

python
Copy code
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
Splits the data into training and testing sets.

7. Model Training
Training the Model:
python
Copy code
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
Uses the XGBoost Regressor to train the model on the training data.
8. Evaluation
Prediction and Evaluation:
python
Copy code
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
Evaluates the model using the R-squared metric for both training and testing data. Higher R-squared values indicate a better fit.
Summary
This project involves several steps from data extraction, loading, preprocessing, and visualization to model building and evaluation. The XGBoost Regressor model is used for predicting sales, and its performance is evaluated using the R-squared metric. The key tasks include handling missing values, encoding categorical variables, and ensuring consistent data formats, which are crucial for building an effective machine learning model.
