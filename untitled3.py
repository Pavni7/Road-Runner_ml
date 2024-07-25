import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)  # One feature22pip install scikit-learn

y = 2 * X.squeeze() + 1 + 0.1 * np.random.rand(100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the machine learning model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model using pickle
with open('roadrunner_speed_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Load the trained machine learning model using pickle
with open('roadrunner_speed_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions using the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate the model (you can replace this with your actual evaluation code)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")