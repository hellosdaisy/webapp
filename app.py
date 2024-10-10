import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('E:异位妊娠//大样本//网页计算器//svm_model.pkl')

# Define feature options
Vaginal_bleeding_options = {    
    0: 'No bleeding (0)',    
    1: 'Less than menstrual flow (1)',    
    2: 'Equivalent to menstrual flow (2)',    
}

HCG_options = {    
    1: 'HCG＜1000 (1)',    
    2: '1000≤HCG＜2000 (2)',    
    3: '2000≤HCG＜3000 (3)',
    4: '3000≤HCG＜4000 (4)',
    5: 'HCG≥5000 (5)'
}

# Define feature names
feature_names = [    
    "Gravidity", "History_of_pelvic_surgery", "History_of_cesarean_section", "Vaginal_bleeding", "Abdominal_tenderness",   
    "Homogeneous_adnexal_mass", "HCG", "Progesterone"
]

# Streamlit user interface
st.title("Predictive Model of Ectopic Pregnancy")

# Gravidity: numerical input
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=8, value=1)

# History_of_pelvic_surgery: categorical selection
History_of_pelvic_surgery = st.selectbox("History_of_pelvic_surgery (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# History_of_cesarean_section: categorical selection
History_of_cesarean_section = st.selectbox("History_of_cesarean_section (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Vaginal_bleeding: categorical selection
Vaginal_bleeding = st.selectbox("Vaginal_bleeding:", options=list(Vaginal_bleeding_options.keys()), format_func=lambda x: Vaginal_bleeding_options[x])

#Abdominal_tenderness: categorical selection
Abdominal_tenderness = st.selectbox("Abdominal_tenderness (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

#Homogeneous_adnexal_mass: categorical selection
Homogeneous_adnexal_mass = st.selectbox("Homogeneous_adnexal_mass (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# HCG: categorical selection
HCG = st.selectbox("HCG:", options=list(HCG_options.keys()), format_func=lambda x: HCG_options[x])

# Progesterone: numerical input
Progesterone = st.number_input("Progesterone:", min_value=0.5, max_value=58, value=1)

# Process inputs and make predictions
feature_values = [Gravidity, History_of_pelvic_surgery, History_of_cesarean_section, Vaginal_bleeding, Abdominal_tenderness, Homogeneous_adnexal_mass, HCG, Progesterone]
features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of ectopic pregnancy. "            
            f"The model predicts that your probability of having ectopic pregnancy is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "          
        )
    else:        
        advice = (            
            f"According to our model, you have a low risk of ectopic pregnancy. "           
              f"The model predicts that your probability of not having ectopic pregnancy is {probability:.1f}%. "                 
        )
    st.write(advice)

     # Calculate SHAP values and display force plot    
    explainer= shap.KernelExplainer(model)   
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")