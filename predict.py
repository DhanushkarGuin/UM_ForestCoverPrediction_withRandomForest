import numpy as np
import pandas as pd
import pickle

with open('forest_cover_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

elevation = int(input('Elevation: '))
aspect = int(input('Aspect: '))
slope = int(input('Slope: '))
hd_hydrology = int(input('Horizontal Distance to Hydrology: '))
vd_hydrology = int(input('Vertical Distance to Hydrology: '))
hd_roadways = int(input('Horizontal Distance to Roadways: '))
hs_9am = int(input('Hillshade at 9am: '))
hs_noon = int(input('Hillshade at Noon: '))
hs_3pm = int(input('Hillshade at 3pm: '))
hd_fire = int(input('Horizontal Distance to Fire Points: '))
wilderness = int(input('Wilderness Area (1–4): '))
soil_type = int(input('Soil Type (1–40): '))

wilderness_encoded = [1 if i == (wilderness - 1) else 0 for i in range(4)]

soil_encoded = [1 if i == (soil_type - 1) else 0 for i in range(40)]

user_input = [
    elevation, aspect, slope, hd_hydrology, vd_hydrology, hd_roadways,
    hs_9am, hs_noon, hs_3pm, hd_fire
] + wilderness_encoded + soil_encoded

feature_names = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area{i+1}' for i in range(4)] + [f'Soil_Type{i+1}' for i in range(40)]

user_input_df = pd.DataFrame([user_input], columns=feature_names)

prediction = pipeline.predict(user_input_df)
print("Predicted Cover Type:", prediction[0])