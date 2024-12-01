import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Analysis & ML App", layout="wide", page_icon="📊")

USER_CREDENTIALS = {"username": "admin", "password": "admin123"}

def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

def show_login_form():
    st.write("### Please log in to access the app")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
            st.session_state['logged_in'] = True  # Set login status to True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password. Please try again.")

def show_sidebar():
    st.sidebar.title("Navigasi")
    option = st.sidebar.radio(
        "Pilih Proses:",
        ["📂 Data Preparation", "📊 EDA", "📈 Modeling", "🤖 Prediction", "📊 Cross-validation"]
    )
    return option

def load_dataset(upload_key):
    uploaded_file = st.file_uploader(f"Upload Dataset untuk {upload_key} (CSV)", type=["csv"], key=upload_key)
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset berhasil dimuat!")
            return data
        except Exception as e:
            st.error(f"Gagal memuat dataset: {e}")
            return None
    else:
        st.info(f"Silakan upload dataset untuk {upload_key}.")
        return None

def handle_missing_values(data):
    imputer = SimpleImputer(strategy="median")
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    empty_columns = [col for col in numerical_cols if data[col].isnull().all()]
    
    if empty_columns:
        st.warning(f"Kolom kosong sepenuhnya akan dihapus: {empty_columns}")
        data = data.drop(columns=empty_columns)
        numerical_cols = [col for col in numerical_cols if col not in empty_columns]
    
    imputed_data = imputer.fit_transform(data[numerical_cols])
    
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols, index=data.index)
    
    data[numerical_cols] = imputed_df
    
    return data

# Main app logic
check_login()

if not st.session_state['logged_in']:
    show_login_form()  
else:
    option = show_sidebar()

    if option == "📂 Data Preparation":
        st.title("📂 Data Preparation")
        st.write("### 🛠️ Membersihkan Dataset Anda")
        data = load_dataset("Data Preparation")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            st.write("### ✅ Pembersihan Missing Values")
            data = handle_missing_values(data)
            st.dataframe(data.head())
            if st.button("💾 Simpan Dataset Bersih"):
                data.to_csv("Cleaned_Dataset.csv", index=False)
                st.success("Dataset bersih berhasil disimpan sebagai 'Cleaned_Dataset.csv'.")

    elif option == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")
        st.write("### 🔍 Analisis Data Anda")
        data = load_dataset("EDA")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            if len(numerical_cols) > 0:
                selected_col = st.selectbox("Pilih Kolom untuk Visualisasi Distribusi:", numerical_cols)
                if selected_col:
                    st.write(f"### 🔢 Distribusi Kolom: {selected_col}")
                    fig = px.histogram(data, x=selected_col, nbins=30, title=f"Distribusi {selected_col}")
                    st.plotly_chart(fig)
                if st.checkbox("Tampilkan Korelasi Heatmap"):
                    if len(numerical_cols) > 1:
                        st.write("### 🌡️ Korelasi Heatmap")
                        fig_corr, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(data[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                        st.pyplot(fig_corr)
                    else:
                        st.warning("Dataset tidak memiliki cukup kolom numerik untuk heatmap.")
            else:
                st.warning("Dataset tidak memiliki kolom numerik untuk analisis.")
            
            if st.checkbox("Tampilkan Statistik Deskriptif"):
                st.write("### 📊 Statistik Deskriptif")
                st.dataframe(data.describe())
                
            if st.checkbox("Tampilkan Scatter Matrix"):
                st.write("### 🔍 Scatter Matrix")
                fig_scatter_matrix = px.scatter_matrix(
                    data, 
                    dimensions=data.select_dtypes(include=['float64', 'int64']).columns,  # Select numeric columns
                    title="Scatter Matrix of Features"
                )
                st.plotly_chart(fig_scatter_matrix)

    elif option == "📈 Modeling":
        st.title("📈 Modeling")
        st.write("### 🧠 Latih Model Machine Learning Anda")
        data = load_dataset("Modeling")
        if data is not None:
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            target = st.selectbox("🎯 Pilih Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            st.success("Model berhasil dilatih!")
            y_pred = model.predict(X_test)
            st.write("### 📊 Evaluasi Model")
            st.metric("🎯 Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
            st.text("📋 Classification Report:")
            st.text(classification_report(y_test, y_pred))

            if st.checkbox("Tampilkan Feature Importance"):
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                feature_importance = rf_model.feature_importances_
                feature_names = X.columns
                fig = px.bar(x=feature_names, y=feature_importance, title="Feature Importance")
                st.plotly_chart(fig)

    elif option == "🤖 Prediction":
        st.title("🤖 Prediction")
        st.write("### 🔮 Prediksi dengan Dataset Baru")
        train_data = load_dataset("Prediction (Pelatihan)")
        if train_data is not None:
            st.write("### 📋 Dataset Pelatihan Overview")
            st.dataframe(train_data.head())
            target = st.selectbox("🎯 Pilih Target Variable (Pelatihan):", train_data.columns)
            features = train_data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = train_data[target]
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X, y)
            st.success("Model telah dilatih!")
            pred_data = load_dataset("Prediction (Prediksi)")
            if pred_data is not None:
                st.write("### 📋 Dataset Prediksi Overview")
                st.dataframe(pred_data.head())
                pred_data_imputed = SimpleImputer(strategy="median").fit_transform(pred_data.select_dtypes(include=['float64', 'int64']))
                pred_data_imputed_df = pd.DataFrame(pred_data_imputed, columns=X.columns, index=pred_data.index)

                predictions = model.predict(pred_data_imputed_df)
                pred_data['Prediction'] = predictions
                st.write("### 🔮 Hasil Prediksi")
                st.dataframe(pred_data)

                st.write("### 📈 Visualisasi Prediksi")
                fig_pred = px.bar(pred_data, x=pred_data.index, y='Prediction', title="Prediksi Hasil")
                st.plotly_chart(fig_pred)

    elif option == "📊 Cross-validation":
        st.title("📊 Cross-validation")
        st.write("### 🧪 Evaluasi Model dengan Cross-Validation")
        data = load_dataset("Cross-validation")
        if data is not None :
            st.write("### 📋 Dataset Overview")
            st.dataframe(data.head())
            target = st.selectbox("🎯 Pilih Target Variable:", data.columns)
            features = data.drop(columns=[target])
            X = features.select_dtypes(include=['float64', 'int64'])
            y = data[target]

            model = LogisticRegression(max_iter=1000, random_state=42)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cross_val_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

            st.write("### 📊 Cross-Validation Results")
            st.metric("🎯 Akurasi Rata-rata", f"{np.mean(cross_val_scores):.2f}")
            st.text(f"📋 Scores dari tiap fold: {cross_val_scores}")
            
            st.write("### 📊 Distribusi Skor Cross-Validation")
            fig_cv, ax = plt.subplots()
            ax.hist(cross_val_scores, bins=5, edgecolor='black')
            ax.set_title("Distribusi Skor Cross-Validation")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            st.pyplot(fig_cv)

