import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('train.csv')
df.drop(columns=['Id'], inplace=True)

X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Feature engineering: convert binary wilderness and soil columns into categorical features
# These are one-hot encoded binary columns: Wilderness_Area1..4 and Soil_Type1..40
wilderness_cols = [col for col in X.columns if 'Wilderness_Area' in col]
soil_cols = [col for col in X.columns if 'Soil_Type' in col]
numerical_cols = [col for col in X.columns if col not in wilderness_cols + soil_cols]

preprocessor = ColumnTransformer(transformers=[
    ('scaler', StandardScaler(), numerical_cols),
    ('wilderness', 'passthrough', wilderness_cols),
    ('soil', 'passthrough', soil_cols),
    ('num', 'passthrough', numerical_cols)
], remainder='drop')

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

with open('forest_cover_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)