import streamlit as st
import pandas as pd
import numpy as np
import subprocess
subprocess.check_call(["pip", "install", "-i", "https://mirrors.aliyun.com/pypi/simple/", "shap"])
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

st.header("Gangrenous Cholecystitis Prediction Model")
WBC = st.number_input("WBC")
NLR = st.number_input("NLR")
D_dimer = st.number_input("D-dimer")
Fibrinogen = st.number_input("Fibrinogen")
Gallbladder_width = st.number_input("Gallbladder width")
Gallbladder_wallness = st.number_input("Gallbladder wallness")
Hypokalemia_hyponatremia= st.number_input("Hypokalemia or hyponatremia")


if st.button("Submit"):
    X = pd.DataFrame([[WBC, NLR, D_dimer, Fibrinogen, Gallbladder_width, Gallbladder_wallness, Hypokalemia_hyponatremia]], 
                     columns=["WBC", "NLR", "D-dimer", "Fibrinogen", "Gallbladder width","Gallbladder wallness","Hypokalemia or hyponatremia"])
    
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
    st.markdown("#### _Based on feature values, predicted possibility of Gangrenous {:.2%}_".format(pred[0][1]))
    prediction = model_xgb.predict(X)[0]
    st.text(f"This instance is a {prediction}")
