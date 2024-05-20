import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# Load the classifier and data outside the button logic
model_xgb = joblib.load("model_xgb.pkl")
csv_path = 'data.csv'
df = pd.read_csv(csv_path)
sample_size = 100
indices = np.random.choice(df.index, size=sample_size, replace=False)
X_background = df.drop('Gangrenous', axis=1).loc[indices]

# Initialize the explainer with the background data
explainer = shap.Explainer(model_xgb.predict, X_background)

mean_values = {
    "WBC": 9.16,
    "NLR": 7.15,
    "D_dimer": 0.84,
    "Fibrinogen": 4.43,
    "Gallbladder width": 3.33,
    "Gallbladder wallness": 0.42
}

std_values = {
    "WBC": 4.71,
    "NLR": 8.77,
    "D_dimer": 0.99,
    "Fibrinogen": 1.69,
    "Gallbladder width": 0.99,
    "Gallbladder wallness": 0.16
}

st.header("Gangrenous Cholecystitis Prediction Model")
WBC = st.number_input("WBC")
NLR = st.number_input("NLR")
D_dimer = st.number_input("D-dimer")
Fibrinogen = st.number_input("Fibrinogen")
Gallbladder_width = st.number_input("Gallbladder width")
Gallbladder_wallness = st.number_input("Gallbladder wallness")
Hypokalemia_hyponatremia= st.number_input("Hypokalemia or hyponatremia")


if st.button("Submit"):
    WBC_std = (WBC - mean_values["WBC"]) / std_values["WBC"]
    NLR_std = (NLR - mean_values["NLR"]) / std_values["NLR"]
    D_dimer_std = (D_dimer - mean_values["D_dimer"]) / std_values["D_dimer"]
    Fibrinogen_std = (Fibrinogen - mean_values["Fibrinogen"]) / std_values["Fibrinogen"]
    Gallbladder_width_std = (Gallbladder_width - mean_values["Gallbladder width"]) / std_values["Gallbladder width"]
    Gallbladder_wallness_std = (Gallbladder_wallness - mean_values["Gallbladder wallness"]) / std_values["Gallbladder wallness"]
   
    X = pd.DataFrame([[WBC_std, NLR_std, D_dimer_std, Fibrinogen_std, Gallbladder_width_std, Gallbladder_wallness_std, Hypokalemia_hyponatremia]], 
                     columns=["WBC", "NLR", "D-dimer", "Fibrinogen", "Gallbladder width", "Gallbladder wallness", "Hypokalemia or hyponatremia"])

    
    shap_values = explainer(X)
    rounded_values = np.round(shap_values.values, 2)

    # Attempt to calculate an average base value (use cautiously)
    base_value = np.mean(model_xgb.predict_proba(X_background)[:, 1])  # For binary classification, adjust accordingly

    # Generate and display SHAP force plot
    shap.force_plot(base_value, rounded_values[0], X.iloc[0],
                    feature_names=X.columns.tolist(), matplotlib=True, show=False)
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.tight_layout()
    plt.savefig("shap_plot.png")
    st.image('shap_plot.png')
    
    pred = model_xgb.predict_proba(X)
    st.markdown("#### _Based on feature values, predicted possibility of Gangrenous Cholecystitis {:.2%}_".format(pred[0][1]))
    prediction = model_xgb.predict(X)[0]
    st.text(f"This instance is a {prediction}")
