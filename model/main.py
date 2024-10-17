import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import h5py

# Load the dataset
diabete_df = pd.read_csv('data/diabetes.csv')

# Prepare the features and target
X = diabete_df.drop('Outcome', axis=1)
y = diabete_df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save model coefficients in H5 format (optional)
with h5py.File('diabetes_model.h5', 'w') as h5f:
    h5f.create_dataset('coefficients', data=model.coef_)
    h5f.create_dataset('intercept', data=model.intercept_)

# Make a prediction with new input data
input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)
input_data_nparray = np.asarray(input_data)
reshaped_input_data = input_data_nparray.reshape(1, -1)

# Scale the input data
scaled_input_data = scaler.transform(reshaped_input_data)

# Make a prediction
prediction = model.predict(scaled_input_data)

if prediction == 1:
    print('This person has diabetes.')
else:
    print('This person does not have diabetes.')
