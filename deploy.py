import streamlit as st
import joblib
import pandas as pd

# Load model, label encoder, and columns
model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
model_columns = joblib.load('model_columns.joblib')

# Judul aplikasi
st.title('Prediksi Tipe Obesitas')

col1, col2 = st.columns(2)

# Iput fitur
with col1:
    Gender = st.selectbox('Apa jenis kelamin Anda?', ['Male', 'Female'])
    Age = st.number_input('Berapa usia Anda?', min_value=0, max_value=100, value=25)
    Height = st.number_input('Berapa tinggi badan Anda (cm)?', min_value=50, max_value=250, value=170)
    Weight = st.number_input('Berapa berat badan Anda (kg)?', min_value=20, max_value=200, value=70)
    family_history_with_overweight = st.selectbox('Apakah ada riwayat keluarga dengan kelebihan berat badan?', ['yes', 'no'])
    FAVC = st.selectbox('Apakah Anda sering mengonsumsi makanan berkalori tinggi?', ['yes', 'no'])
    FCVC = st.slider('Seberapa sering Anda mengonsumsi sayuran (0-3)?', 0, 3, 2)
    NCP = st.slider('Berapa kali Anda makan utama dalam sehari (1-5)?', 1, 5, 3)

with col2:
    CAEC = st.selectbox('Apakah Anda suka ngemil?', ['no', 'Sometimes', 'Frequently', 'Always'])
    SMOKE = st.selectbox('Apakah Anda merokok?', ['yes', 'no'])
    SCC = st.selectbox('Apakah Anda memantau konsumsi kalori?', ['yes', 'no'])
    CALC = st.selectbox('Seberapa sering Anda mengonsumsi alkohol?', ['no', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox('Apa moda transportasi yang Anda gunakan?', ['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile'])
    CH2O = st.slider('Berapa liter air yang Anda minum per hari?', 0, 5, 2)
    FAF = st.slider('Berapa kali Anda berolahraga dalam seminggu?', 0, 7, 3)
    TUE = st.slider('Berapa jam Anda menggunakan perangkat teknologi per hari?', 0, 24, 5)

# Create dictionary with inputs
input_data = {
    'Gender': [Gender],
    'Age': [Age],
    'Height': [Height],
    'Weight': [Weight],
    'family_history_with_overweight': [family_history_with_overweight],
    'FAVC': [FAVC],
    'FCVC': [FCVC],
    'NCP': [NCP],
    'CAEC': [CAEC],
    'SMOKE': [SMOKE],
    'CH2O': [CH2O],
    'SCC': [SCC],
    'FAF': [FAF],
    'TUE': [TUE],
    'CALC': [CALC],
    'MTRANS': [MTRANS]
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame(input_data)

#Mengubah data kategorikal menjadi data numerik dengan one-hot encoding
input_data = pd.get_dummies(input_df)

# Pastikan kolom input_data sesuai dengan kolom fitur yang digunakan saat pelatihan model
input_data = input_data.reindex(columns=model_columns, fill_value=0)

def get_prediction_color(prediction):
    if prediction == 1:
        return '#4CAF50'  # Green
    elif prediction == 3:
        return '#FFC107'  # Amber
    elif prediction == 2:
        return '#F44336'  # Red
    else:
        return '#FFFFFF'  # White
    
# Predict button
if st.button('Prediksi'):
    # Make prediction
    prediction = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)

    color = get_prediction_color(prediction[0])

    st.markdown("""
        <div style="text-align: center;">
            <h2>Hasil Prediksi</h2>
            <div style="background-color: {}; padding: 10px; border-radius: 5px; color: white;">
                <h3>{}</h3>
            </div>
        </div>
    """.format(color, prediction_label[0]), unsafe_allow_html=True)
