import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
st.title("Sales Prediction App")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:", df.head())
    
    # Preprocessing
    df.dropna(inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    
    # Splitting Data
    X = df.drop(columns=["Item_Outlet_Sales"], axis=1)
    y = df["Item_Outlet_Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Display Metrics
    st.write("### Model Performance:")
    st.write("- MAE:", mean_absolute_error(y_test, y_pred))
    st.write("- MSE:", mean_squared_error(y_test, y_pred))
    st.write("- R2 Score:", r2_score(y_test, y_pred))
    
    # Visualization
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)
