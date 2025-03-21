url = 'https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_log_model.pkl'


response = requests.get(url)
with open('best_log_model.pkl', 'wb') as f:
    f.write(response.content)

best_log_model = joblib.load('best_log_model.pkl')

ur2 = 'https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_rf_model.pkl'

response = requests.get(ur2)
with open('best_rf_model.pkl', 'wb') as f:
    f.write(response.content)

# بارگذاری مدل
best_rf_model = joblib.load('best_rf_model.pkl')

ur3 = 'https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_gb_model.pkl'

response = requests.get(ur3)
with open('best_gb_model.pkl', 'wb') as f:
    f.write(response.content)

# بارگذاری مدل
best_gb_model = joblib.load('best_gb_model.pkl')



import streamlit as st
import numpy as np
import pandas as pd
import joblib  # برای بارگذاری مدل‌های ذخیره‌شده
from io import BytesIO



best_log_model = joblib.load('https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_log_model.pkl')
best_rf_model = joblib.load('https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_rf_model.pkl')
best_gb_model = joblib.load('https://github.com/behshad-ghasemi/blank-app/raw/main/models/best_gb_model.pkl')
scaler = joblib.load("https://github.com/behshad-ghasemi/blank-app/raw/main/models/scaler.pkl")

st.title('Calculating the probability of getting HFpEF!')
st.write("Please insert the input, you can see the probability of each model.\n"
"\n Have a good prediction, Behshad :)")

mtDNA_cn = st.number_input("mtDNA cn", min_value=0.0, max_value=1.0, step=0.01)
telomere = st.number_input("telomere", min_value=0.0, max_value=1.0, step=0.01)
miR_21 = st.number_input("miR-21", min_value=0.0, max_value=1.0, step=0.01)
miR_92 = st.number_input("miR-92", min_value=0.0, max_value=1.0, step=0.01)

if st.button('Etimation'):
    new_data = np.array([[mtDNA_cn, telomere, miR_21, miR_92]])
    new_data_scaled = scaler.transform(new_data)


    prob_log = best_log_model.predict_proba(new_data_scaled)[:, 1]
    prob_rf = best_rf_model.predict_proba(new_data)[:, 1]
    prob_gb = best_gb_model.predict_proba(new_data)[:, 1]

    

    st.write(f"🔹 **Logistic Regression:** probability of getting HFpEF:  {prob_log[0]:.4f}")
    st.write(f"🔹 **Random Forest:** probability of getting HFpEF : {prob_rf[0]:.4f}")
    st.write(f'🔹 **Gradient Boosting:** probability of getting HFpEF : {prob_gb[0]:.4f}')

    results = {
        "mtDNA_cn": [mtDNA_cn],
        "Telomere": [telomere],
        "miR_21": [miR_21],
        "miR_92": [miR_92],
        "Logistic Regression": [prob_log[0]],
        "Random Forest": [prob_rf[0]],
        "Gradient Boosting": [prob_gb[0]]
    }

    if prob_gb[0] > 0.6:
        st.write("🚨 😱 **This person is at high risk of HFpEF** 😱 ")
    else:
        st.write("🥳 **This person is at low risk of HFpEF** \n ^-^ I am happy!  ")
    import seaborn as sns
    import matplotlib.pyplot as plt

    # نمودار مقایسه‌ای احتمال مدل‌ها
    fig, ax = plt.subplots(figsize=(6, 6))
    models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    probabilities = [prob_log[0], prob_rf[0], prob_gb[0]]
    
    sns.barplot(x=models, y=probabilities, palette='viridis', ax=ax)
    ax.set_title("Comparison of Models' Probability for HFpEF")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

        # ذخیره نمودار به صورت PDF
    def save_plot_as_pdf(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format="pdf")
        buffer.seek(0)
        return buffer


    # نمایش ویژگی‌های مهم Gradient Boosting
    def plot_feature_importance(model):
        importance = model.feature_importances_
        features = ["mtDNA_cn", "Telomere", "miR_21", "miR_92"]
        importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        st.write("### Feature Importance - Gradient Boosting Model")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        st.pyplot(fig)

        return fig

    feature_importance_fig = plot_feature_importance(best_gb_model)


