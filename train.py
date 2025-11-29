import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib 

# 1. Get Data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
df = pd.read_csv(url).dropna()

# 2. Prepare Features (X) and Target (y)
# converting text species to numbers: Adelie=0, Chinstrap=1, Gentoo=2
df['species_code'] = df['species'].astype('category').cat.codes
X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species_code']

# 3. Train
model = RandomForestClassifier()
model.fit(X, y)

# 4. Save the "Brain" (Serialization)
# We use joblib (faster than pickle for large numpy arrays)
joblib.dump(model, "penguin_model.joblib")
print("Model trained and saved as penguin_model.joblib")
