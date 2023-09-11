import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
st.set_page_config(page_title="Latihan Dashboard", layout="wide")
st.write('''
# Hallo Selamat datang di Dashboard saya      
perkenalkan nama saya [Yohn Fhayer](https: )
''')
add_select_item = st.sidebar.selectbox("Kamu mau analisa apa?",("Iris","HeartDisease!"))

def iris():
    st.write('''
    Ini merupakan aplikasi prediksi untuk iris
    ''')
    st.sidebar.header("User Input")
    upload = st.sidebar.file_uploader("Upload CSV", type=['CSV'])
    if upload is not None:
        input_df = pd.read_csv(upload)
    else:
        def user_manual():
            st.sidebar.header("Manual Input")
            sepalLengthCm = st.sidebar.slider("Sepal Length (cm)", 4.3, 10.0, 6.5)
            sepalWidthCm = st.sidebar.slider("Sepal Width (cm)", 1.0, 6.0, 3.0)
            petalWidthCm = st.sidebar.slider("Petal Width (cm)", 1.0, 6.0, 3.0)
            petalLengthCm = st.sidebar.slider("Petal Length (cm)", 4.3, 10.0,6.5)
            data = {"SepalLengthCm":sepalLengthCm,
                    "SepalWidthCm"  :sepalWidthCm,
                    "PetalLengthCm" :petalLengthCm,
                    "petalWidthCm" : petalWidthCm}
            df = pd.DataFrame(data, index = [0])
            return df
        input_df = user_manual()
    img_iris= Image.open(r'iris.jpg')
    st.image(img_iris, width = 100)

    if st.sidebar.button("Predict"):
        df = input_df
        st.write(df)
        with open(r"best_model_iris.pkl",'rb') as file:
            clf = pickle.load(file)
        prediction = clf.predict(df)
        result = ['Iris-Sentosa' if prediction == 0 else ("Iris-Versicolor" if prediction == 1 else "Iris-Virginica")]
        st.subheader("Prediction:")
        output = str(result[0])
        with st.spinner("Wait for it!"):
            time.sleep(5)
            st.success(f"Preiction adalah {output} sebagai tepilih")

def heart_disease():
    st.write('''
    Ini merupakan aplikasi prediksi untuk Heart Disease
    ''')
    st.sidebar.header("User Input")
    upload = st.sidebar.file_uploader("Upload CSV", type=['CSV'])
    if upload is not None:
        input_df = pd.read_csv(upload)
    else:
        def user_manual():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 1.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Excercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            sex = st.sidebar.selectbox("Jenis Kelamin", ("Perempuan","Pria"))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1
            age = st.sidebar.slider("Usia", 29, 77, 30)
            data = {"cp":cp,
                    "thalach":thalach,
                    "slope":slope,
                    "oldpeak":oldpeak,
                    "exang":exang,
                    "ca":ca,
                    "thal":thal,
                    "sex":sex,
                    "age":age}
            features = pd.DataFrame(data, index = [0])
            return features
        input_df = user_manual()
    img_hd= Image.open(r'heart_disease.jpg')
    st.image(img_hd, width = 100)

    if st.sidebar.button("Predict"):
        df = input_df
        st.write(df)
        with open(r"best_model_rf.pkl",'rb') as file:
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else "Yes Heart Disease"]
        st.subheader("Prediction:")
        output = str(result[0])
        with st.spinner("Wait for it!"):
            time.sleep(5)
            st.success(f"Preiction adalah {output} sebagai tepilih")

if add_select_item == "Iris":
    iris()
elif add_select_item == "HeartDisease!":
    heart_disease()




