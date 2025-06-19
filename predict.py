import numpy as np
import pickle

# Load pipeline
with open('forest_cover_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Gather user input
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

# Encode wilderness area (4 binary cols)
wilderness_encoded = [1 if i == (wilderness - 1) else 0 for i in range(4)]

# Encode soil type (40 binary cols)
soil_encoded = [1 if i == (soil_type - 1) else 0 for i in range(40)]

# Combine all inputs
user_input = [
    elevation, aspect, slope, hd_hydrology, vd_hydrology, hd_roadways,
    hs_9am, hs_noon, hs_3pm, hd_fire
] + wilderness_encoded + soil_encoded

user_input = np.array(user_input).reshape(1, -1)

# Predict
prediction = pipeline.predict(user_input)
print("Predicted Cover Type:", prediction[0])