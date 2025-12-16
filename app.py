import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🏥", layout="wide")

@st.cache_data
def load_and_train_model():
    df = pd.read_csv('diabetes_dataset.csv')
    df_clean = df.drop('PatientID', axis=1)
    df_clean['Gender_Encoded'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
    
    df_fe = df_clean.copy()
    df_fe['BMI_Category_Encoded'] = pd.cut(df_fe['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df_fe['Age_Group_Encoded'] = pd.cut(df_fe['Age'], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
    df_fe['Glucose_Level_Encoded'] = pd.cut(df_fe['Glucose'], bins=[0, 100, 125, 200], labels=[0, 1, 2]).astype(int)
    df_fe['BMI_Age_Interaction'] = df_fe['BMI'] * df_fe['Age']
    df_fe['Glucose_Insulin_Ratio'] = df_fe['Glucose'] / (df_fe['Insulin'] + 1)
    df_fe['BMI_Glucose_Product'] = df_fe['BMI'] * df_fe['Glucose']
    df_fe['High_Risk'] = ((df_fe['BMI'] > 30) & (df_fe['Glucose'] > 125)).astype(int)
    df_fe['Age_BMI_Risk'] = ((df_fe['Age'] > 45) & (df_fe['BMI'] > 30)).astype(int)
    
    features = ['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction', 
                'Gender_Encoded', 'BMI_Category_Encoded', 'Age_Group_Encoded', 'Glucose_Level_Encoded',
                'BMI_Age_Interaction', 'Glucose_Insulin_Ratio', 'BMI_Glucose_Product',
                'High_Risk', 'Age_BMI_Risk']
    
    X = df_fe[features]
    y = df_fe['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train both models
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, 
                                      min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    lr_model = LogisticRegression(random_state=42, max_iter=1000, C=0.1)
    lr_model.fit(X_train_scaled, y_train)
    
    # Use Logistic Regression as primary (more interpretable and stable)
    model = lr_model
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    return model, scaler, accuracy, roc_auc, df_clean

def calculate_features(age, gender, bmi, bp, insulin, glucose, dpf):
    gender_encoded = 1 if gender == 'Male' else 0
    
    if bmi <= 18.5:
        bmi_cat = 0
    elif bmi <= 25:
        bmi_cat = 1
    elif bmi <= 30:
        bmi_cat = 2
    else:
        bmi_cat = 3
    
    if age <= 30:
        age_group = 0
    elif age <= 45:
        age_group = 1
    elif age <= 60:
        age_group = 2
    else:
        age_group = 3
    
    if glucose <= 100:
        glucose_level = 0
    elif glucose <= 125:
        glucose_level = 1
    else:
        glucose_level = 2
    
    bmi_age_int = bmi * age
    glucose_insulin_ratio = glucose / (insulin + 1)
    bmi_glucose_prod = bmi * glucose
    high_risk = 1 if (bmi > 30 and glucose > 125) else 0
    age_bmi_risk = 1 if (age > 45 and bmi > 30) else 0
    
    return [age, bmi, bp, insulin, glucose, dpf, gender_encoded, 
            bmi_cat, age_group, glucose_level, bmi_age_int, 
            glucose_insulin_ratio, bmi_glucose_prod, high_risk, age_bmi_risk]

st.title("🏥 Aplikasi Prediksi Diabetes")
st.markdown("### Machine Learning untuk Deteksi Risiko Diabetes")
st.markdown("---")

model, scaler, accuracy, roc_auc, df_clean = load_and_train_model()

tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Analisis Dataset", "ℹ️ Info Model"])

with tab1:
    st.header("Masukkan Data Pasien")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Usia (tahun)", min_value=1, max_value=100, value=45, step=1)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.5, step=0.1)
        blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=40, max_value=200, value=80, step=1)
    
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=500, value=100, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=300, value=120, step=1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    
    if st.button("🔍 Prediksi Risiko Diabetes", type="primary"):
        features = calculate_features(age, gender, bmi, blood_pressure, insulin, glucose, dpf)
        features_scaled = scaler.transform([features])
        
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        st.markdown("---")
        st.subheader("📋 Hasil Prediksi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **RISIKO TINGGI**")
                st.metric("Status", "Diabetes")
            else:
                st.success("✅ **RISIKO RENDAH**")
                st.metric("Status", "Non-Diabetes")
        
        with col2:
            st.metric("Probabilitas Diabetes", f"{probability[1]*100:.1f}%")
        
        with col3:
            st.metric("Probabilitas Non-Diabetes", f"{probability[0]*100:.1f}%")
        
        st.markdown("---")
        st.subheader("📈 Interpretasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Faktor Risiko Anda:**")
            if bmi > 30:
                st.write("- ⚠️ BMI tinggi (Obesitas)")
            elif bmi > 25:
                st.write("- ⚠️ BMI sedikit tinggi (Overweight)")
            else:
                st.write("- ✅ BMI normal")
            
            if glucose > 125:
                st.write("- ⚠️ Glucose tinggi")
            elif glucose > 100:
                st.write("- ⚠️ Glucose sedikit tinggi (Prediabetes)")
            else:
                st.write("- ✅ Glucose normal")
            
            if age > 45:
                st.write("- ⚠️ Usia di atas 45 tahun")
            else:
                st.write("- ✅ Usia relatif muda")
        
        with col2:
            st.write("**Rekomendasi:**")
            if prediction == 1:
                st.write("- 🏥 Konsultasi dengan dokter segera")
                st.write("- 🍎 Terapkan pola makan sehat")
                st.write("- 🏃 Rutin berolahraga")
                st.write("- 📊 Monitor glucose secara berkala")
            else:
                st.write("- ✅ Pertahankan gaya hidup sehat")
                st.write("- 🏃 Olahraga teratur")
                st.write("- 🍎 Makan makanan bergizi")
                st.write("- 📊 Check-up rutin")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(['Non-Diabetes', 'Diabetes'], probability*100, color=['green', 'red'])
        ax.set_xlabel('Probabilitas (%)')
        ax.set_xlim([0, 100])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                   f'{probability[i]*100:.1f}%', va='center')
        st.pyplot(fig)

with tab2:
    st.header("📊 Analisis Dataset Diabetes")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df_clean))
    col2.metric("Diabetes", df_clean['Outcome'].sum())
    col3.metric("Non-Diabetes", len(df_clean) - df_clean['Outcome'].sum())
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Outcome")
        fig, ax = plt.subplots(figsize=(8, 5))
        outcome_counts = df_clean['Outcome'].value_counts()
        ax.pie(outcome_counts, labels=['Non-Diabetes', 'Diabetes'], autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'], startangle=90)
        ax.set_title('Distribusi Diabetes vs Non-Diabetes')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribusi Usia")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df_clean, x='Age', hue='Outcome', kde=True, ax=ax, bins=20)
        ax.set_title('Distribusi Usia berdasarkan Outcome')
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BMI vs Outcome")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_clean, x='Outcome', y='BMI', ax=ax)
        ax.set_xticklabels(['Non-Diabetes', 'Diabetes'])
        st.pyplot(fig)
    
    with col2:
        st.subheader("Glucose vs Outcome")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_clean, x='Outcome', y='Glucose', ax=ax)
        ax.set_xticklabels(['Non-Diabetes', 'Diabetes'])
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    numeric_cols = ['Age', 'BMI', 'BloodPressure', 'Insulin', 'Glucose', 'DiabetesPedigreeFunction', 'Outcome']
    correlation = df_clean[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("ℹ️ Informasi Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model", "Logistic Regression")
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
        st.metric("ROC-AUC Score", f"{roc_auc:.4f}")
    
    with col2:
        st.metric("Regularization (C)", "0.1")
        st.metric("Max Iterations", "1000")
        st.metric("Test Size", "20%")
    
    st.markdown("---")
    
    st.subheader("📝 Tentang Model")
    st.write("""
    Model ini menggunakan **Logistic Regression** dengan **Feature Engineering** 
    untuk memprediksi risiko diabetes pada pasien.
    
    Logistic Regression dipilih karena interpretable dan memberikan probability 
    yang lebih reliable untuk medical screening.
    
    **Features yang digunakan:**
    - Basic Features: Age, BMI, Blood Pressure, Insulin, Glucose, Diabetes Pedigree Function, Gender
    - Engineered Features: BMI Category, Age Group, Glucose Level, Interaction Features, Risk Indicators
    
    **Performance:**
    - Accuracy mencapai ~78-80%
    - ROC-AUC Score ~0.82-0.85
    - Model telah dilatih dengan 200 data pasien
    
    **Cara Kerja:**
    1. User memasukkan data pasien
    2. Model menghitung features tambahan secara otomatis
    3. Model memprediksi probabilitas diabetes
    4. Hasil ditampilkan dengan rekomendasi
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Disclaimer")
    st.warning("""
    Aplikasi ini adalah alat bantu skrining dan **BUKAN pengganti diagnosa medis profesional**.
    Selalu konsultasikan dengan dokter untuk diagnosa dan penanganan yang tepat.
    """)
    
    st.markdown("---")
    st.info("💡 **Tip:** Aplikasi ini dibuat sebagai tugas analisis data untuk Pak Yulizar")
