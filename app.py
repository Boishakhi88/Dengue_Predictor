import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Load model and scaler ---
model = tf.keras.models.load_model('lstm_model.h5', compile=False)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Define features ---
features = [
    'Region_Code', 'Temperature', 'Humidity', 'Rainfall', 'Sunshine',
    'Population Density', 'Urbanization Rate', 'Waste Management ',
    'Water Storage ', 'NDVI', 'Month', 'Week',
    'lag_1', 'lag_2', 'lag_3',
    'rolling_3', 'rolling_7', 'rolling_14',
    'Temp_Humidity', 'Rainfall_Sunshine'
]

# --- Title ---
st.title("🦟 Dengue Weekly Predictor: Using LSTM Model")
st.markdown("🔢 Input today's features below. The model will predict dengue cases for the next 7 days.")

# --- User Input ---
user_input = {}
for feat in features:
    default_val = 0.0 if feat != 'Region_Code' else 0
    user_input[feat] = st.number_input(f"{feat}", value=float(default_val))

# --- Forecast Button ---
if st.button("🔮 Predict Next 7 Days"):
    # Prepare initial day-0 input
    today_input = np.array([user_input[f] for f in features])

    # Scale input (excluding Region_Code)
    scaled = today_input.copy()
    scaled[1:] = scaler.transform(today_input[1:].reshape(1, -1))

    # Store initial history (7 days)
    sequence = [scaled] * 7  # start with today repeated 7x

    predictions = []

    for day in range(7):
        # Shape: (1, 7, features)
        input_seq = np.array(sequence[-7:]).reshape(1, 7, len(features))

        # Predict
        pred = model.predict(input_seq).flatten()[0]
        pred = max(0, round(pred))  # clip negative

        predictions.append(pred)

        # Prepare next day's features
        next_input = scaled.copy()

        # Update lag & rolling based on predictions
        prev_lags = [pred] + [sequence[-1][features.index(f)] for f in ['lag_1', 'lag_2']]
        next_input[features.index('lag_1')] = prev_lags[0]
        next_input[features.index('lag_2')] = prev_lags[1]
        next_input[features.index('lag_3')] = prev_lags[2]

        roll_hist = [pred] + [sequence[-i][features.index('lag_1')] for i in range(1, 7)]
        rolling_3 = np.mean(roll_hist[:3])
        rolling_7 = np.mean(roll_hist[:7])
        rolling_14 = np.mean(roll_hist + [0]*(14 - len(roll_hist)))

        next_input[features.index('rolling_3')] = rolling_3
        next_input[features.index('rolling_7')] = rolling_7
        next_input[features.index('rolling_14')] = rolling_14

        # Keep interaction terms same for now (or recalc if others updated)
        next_input[features.index('Temp_Humidity')] = next_input[features.index('Temperature')] * next_input[features.index('Humidity')]
        next_input[features.index('Rainfall_Sunshine')] = next_input[features.index('Rainfall')] * next_input[features.index('Sunshine')]

        # Append for next prediction step
        sequence.append(next_input)

    # --- Display Results ---
    st.success(f"📅 Total Predicted Cases (Next 7 Days): **{sum(predictions)}**")
    st.info(f"📈 Daily Predictions: {predictions}")

    # --- Plot ---
    fig, ax = plt.subplots()
    ax.bar(range(1, 8), predictions, color='orange')
    ax.set_xlabel("Next 7 Days")
    ax.set_ylabel("Predicted Cases")
    ax.set_title("📊 Dengue Forecast for Next 7 Days")
    ax.grid(True)
    st.pyplot(fig)
