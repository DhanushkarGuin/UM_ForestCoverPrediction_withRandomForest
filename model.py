import numpy as np
import pandas as pd

dataset = pd.read_csv('train.csv')
# print(dataset.head())

dataset.drop(['Id'], axis = 1, inplace= True)
# print(dataset.head())

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print('Predictions:', y_pred)

from sklearn.metrics import roc_auc_score
y_pred_proba = model.predict_proba(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("ROC AUC Score (OvR):", roc_auc)

from sklearn.metrics import precision_score, recall_score
print('Precision:', precision_score(y_test,y_pred, average='weighted'))
print('Recall:', recall_score(y_test,y_pred, average='weighted'))

elevation = int(input('Enter elevation in meters:'))
aspect = int(input('Enter aspect in degrees:'))
slope = int(input('Enter slope in degrees:'))
horizontal_distance_to_hydrology = int(input('Enter horizontal dist to nearest surface water features:'))
vertical_distance_to_hydrology = int(input('Enter vertical dist to nearest surface water features:'))
horizontal_distance_to_roadways = int(input('Enter horizontal dist to nearest roadway:'))
hillshade_9am = int(input('Enter hillshade index at 9am:'))
hillshade_noon = int(input('Enter hillshade index at noon:'))
hillshade_3pm = int(input('Enter hillshade index at 3pm:'))
horizontal_distance_to_fire_points = int(input('Enter horizontal dist to nearest wildfire ignition points:'))
wildernessArea = int(input('Enter wilderness area designation(1-4):'))
soil_type = int(input('Enter type of soil(1-40):'))

wildernessArea_encoded = [0,0,0,0]
if wildernessArea == 1:
    wildernessArea_encoded[0] = 1
elif wildernessArea == 2:
    wildernessArea_encoded[1] = 1
elif wildernessArea == 3:
    wildernessArea_encoded[2] = 1
else:
    wildernessArea_encoded[3] = 1

soil_type_encoded = [0] * 40  
if 1 <= soil_type <= 40:
    soil_type_encoded[soil_type - 1] = 1 

user_input = [elevation,aspect,slope,
            horizontal_distance_to_hydrology,vertical_distance_to_hydrology,horizontal_distance_to_roadways,
            hillshade_9am,hillshade_noon,hillshade_3pm,
            horizontal_distance_to_fire_points] + wildernessArea_encoded + soil_type_encoded

user_input = np.array(user_input).reshape(1, -1)

predictions = model.predict(user_input)
print('Predictions:', predictions)







