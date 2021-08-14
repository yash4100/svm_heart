import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv('framingham.csv')
data.drop(['education'],axis=1,inplace=True)


data.dropna(axis=0, inplace=True)

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight='balanced')

# define Boruta feature selection method
feat_selector = BorutaPy(forest, n_estimators='auto', verbose=2)
 
# find all relevant features
feat_selector.fit(X, y)


top_features = data.columns[:-1][feat_selector.ranking_<=6].tolist()

top_features

X_top = data[top_features]
y = data['TenYearCHD']

res = sm.Logit(y,X_top).fit()
res.summary()


X = data[top_features]
y = data.iloc[:,-1]

# the numbers before smote
num_before = dict(Counter(y))

#perform smoting

# define pipeline
over = SMOTE(sampling_strategy=0.8)
under = RandomUnderSampler(sampling_strategy=0.8)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# transform the dataset
X_smote, y_smote = pipeline.fit_resample(X, y)


#the numbers after smote
num_after =dict(Counter(y_smote))


new_data = pd.concat([pd.DataFrame(X_smote), pd.DataFrame(y_smote)], axis=1)
new_data.shape
new_data.columns = ['age', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']

X_new = new_data[top_features]
y_new= new_data.iloc[:,-1]


X_train,X_test,y_train,y_test = train_test_split(X_new,y_new,test_size=.2,random_state=42)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)

#grid search for optimum parameters
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
svm_clf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=10)



# search for optimun parameters using gridsearch
params= {'n_neighbors': np.arange(1, 10)}
grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = params, 
                           scoring = 'accuracy', cv = 10, n_jobs = -1)
knn_clf = GridSearchCV(KNeighborsClassifier(),params,cv=3, n_jobs=-1)

# train the model
knn_clf.fit(X_train,y_train)

#saving model to disk

knn_clf.predict([[63,305,180,90,35,95,150]])

pickle.dump(knn_clf,open('modelfinal.pkl','wb'))
model=pickle.load(open('modelfinal.pkl','rb'))
print(model.predict([[63,305,180,90,35,95,150]]))

#53,200,180,114,31,62,83
#63,305,180,90,35,95,150
