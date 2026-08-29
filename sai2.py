import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page title
st.title("🏠 House Price Prediction")

st.write("Enter the house details to predict its price in Indian Rupees.")

# Load dataset
data = pd.read_csv("house_prices.csv")

# Remove missing values
data = data.dropna()

# Select the 4 features
X = data[["bedrooms", "bathrooms", "sqft_living", "sqft_lot"]]
y = data["price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# User inputs
bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


sqft_living = st.number_input(
    "Living Area (sqft)",
    min_value=300,
    max_value=10000,
    value=2000
)

sqft_lot = st.number_input(
    "Lot Size (sqft)",
    min_value=500,
    max_value=50000,
    value=5000
)

# Prediction
if st.button("Predict House Price"):

    # Create input data
    new_house = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot]
    })

    # Predict price in USD
    predicted_price_usd = model.predict(new_house)[0]

    # Convert USD to INR
    usd_to_inr = 95.43
    predicted_price_inr = predicted_price_usd * usd_to_inr

    # Display result
    st.success(
        f"🏠 Predicted House Price: ₹{predicted_price_inr:,.2f}"
    )