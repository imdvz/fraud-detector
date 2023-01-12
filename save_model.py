import pandas as pd
import pickle
from lightgbm import LGBMClassifier

# Read pre-processed dataset
df_final = pd.read_csv("cleaned_fraud_data.csv")

# Seperate data in independet and dependent variables
X = df_final[['category', 'amt', 'gender', 'dob', 'transaction_hour']]
y = df_final[['is_fraud']]

# Train LightGBM Classifier
model = LGBMClassifier()
model.fit(X_train, y_train)

# Save LightGBM model
pickle_out = open("lgbm_model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()