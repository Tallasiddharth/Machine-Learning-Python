

# Logistic Regression

#-------------Logistic Regression------------------------------
#Import Libraries
import pandas as pd
import seaborn as sns

#Import data 
HR_data = pd.read_csv("D:\\skillfiles\\files\\HR_Data.csv")
HR_data.head(5)
HR_data.tail(5)

print("No. of employees in original dataset:" +str(len(HR_data.index)))
      
#Analyzing Data
sns.countplot(x="left",data=HR_data)

sns.countplot(x="left",hue="salary",data=HR_data)

#CHECKING DATA TYPE OF A VARIABLE AND CONVERTING IT INTO ANOTHER TYPE-----
HR_data.info()

#Identifying/Finding missing values if any----
HR_data.isnull()
HR_data.isnull().sum()

#There are lot of string value var in dataset which have to be converted to numerical
#values for applying machine learning algoritm. Hence, we will now convert string var 
#to numerical var.
HR_data.info()
pd.get_dummies(HR_data["role"])

pd.get_dummies(HR_data["role"],drop_first=True)

Role_Dummy = pd.get_dummies(HR_data["role"],drop_first=True)
Role_Dummy.head(5)
 
pd.get_dummies(HR_data["salary"])
Salary_Dummy = pd.get_dummies(HR_data["salary"],drop_first=True)
Salary_Dummy.head(5)


#Now, lets concatenate these dummy var columns in our dataset.
HR_data = pd.concat([HR_data,Role_Dummy,Salary_Dummy],axis=1)
HR_data.head(5)

#dropping the columns whose dummy var have been created
HR_data.drop(["role","salary"],axis=1,inplace=True)
HR_data.head(5)

#Splitting the dataset into Train & Test dataset
x=HR_data.drop("left",axis=1)
y=HR_data["left"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test,predictions)
accuracy_score(y_test,predictions)
#Hence, accuracy = (2651+343)\(2651+230+526+343) = 79.84%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)

#----To Improve the accuracy of the model, lets go with Backward ELimination Method &
# rebuild the logisitc model again with few independent variables--------
HR_data_1 = HR_data
HR_data_1.head(5)

#--------------------------Backward Elimination--------------------------------
#Backward elimination is a feature selection technique while building a machine learning model. It is used
#to remove those features that do not have significant effect on dependent variable or prediction of output.

#Step: 1- Preparation of Backward Elimination:
#Importing the library:
import statsmodels.api as sm

#Adding a column in matrix of features:
x1=HR_data_1.drop("left",axis=1)
y1=HR_data_1["left"]
import numpy as nm
x1 = nm.append(arr = nm.ones((14999,1)).astype(int), values=x1, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

#for fitting the model, we will create a regressor_OLS object of new class OLS of statsmodels library. 
#Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()

#In the above summary table, we can clearly see the p-values of all the variables. 
#And remove the ind var with p-value greater than 0.05
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,11,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()

#Since we are left with all the significant Ind. Var, hence, we can
#now predict the Dep. Var values efficiently using these Ind. Var.

#-------Building Logistic Regression model using ind var: age, sibsip, sex, pclass & embarked--------  
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.25, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_BE_test,predictions)
accuracy_score(y_BE_test,predictions)
#Accuracy = (2656+315)/(2656+225+554+315) = 79.2%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)


#Calculating the coefficients:

#So, ur final Predicitve Modelling Equation becomes:
#Left = 
#exp(-0.639 -4.15*satisfaction level -0.727*last evaluation -0.33*no. of projects -0.0046*avg monthly hours +0.27*exp in company -1.44*work accident
# -1.046*promotion in last 5 years -0.655*R&D +0.17*hr - 0.45*management +1.8*low salary + 1.26*medium salary
# \
#exp(3.74 -0.03*age -0.27*sibsp -2.52*sex(male) -1.03*pclass(2) -2.1*pclass(3) -0.33*embd(S)) + 1
