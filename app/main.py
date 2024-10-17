import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib


def get_clean_data():
    data = pd.read_csv("data/diabetes.csv")  # Update to your diabetes dataset path
    return data


def add_sidebar():
    st.sidebar.header("Diabetes Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Pregnancies", "Pregnancies"),
        ("Glucose", "Glucose"),
        ("Blood Pressure", "BloodPressure"),
        ("Skin Thickness", "SkinThickness"),
        ("Insulin", "Insulin"),
        ("BMI", "BMI"),
        ("Diabetes Pedigree Function", "DiabetesPedigreeFunction"),
        ("Age", "Age"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['Outcome'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = [
        'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
        'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age'
    ]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['Pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
            input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
            input_data['DiabetesPedigreeFunction'], input_data['Age']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['Pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
            input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
            input_data['DiabetesPedigreeFunction'], input_data['Age']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['Pregnancies'], input_data['Glucose'], input_data['BloodPressure'],
            input_data['SkinThickness'], input_data['Insulin'], input_data['BMI'],
            input_data['DiabetesPedigreeFunction'], input_data['Age']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    


    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

def add_predictions(input_data):
    # Load the model and scaler correctly
    model = joblib.load(open("model/model.pkl", "rb"))  # Ensure you're using joblib here
    scaler = joblib.load(open("model/scaler.pkl", "rb"))

    # Convert input data to a DataFrame for consistent feature names
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_df = pd.DataFrame(input_array, columns=input_data.keys())

    # Scale the input DataFrame
    input_array_scaled = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(input_array_scaled)

    st.subheader("Diabetes Prediction")
    st.write("The prediction is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>The Person Has No Diabetes</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>The Person Has Diabetes</span>", unsafe_allow_html=True)

    st.write("Probability of No Diabetes: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of Diabetes: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis but should not be used as a substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title="Diabetes Predictor",
        page_icon=":apple:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
        
    input_data = add_sidebar()

    with st.container():
        st.title("Diabetes Risk Predictor")
        st.write("Welcome to the Diabetes Risk Predictor, a sophisticated tool designed to evaluate individual risk factors associated with diabetes. Leveraging advanced algorithms and a robust dataset, our predictor offers accurate assessments to aid in proactive health management.")

        st.title("Key Features:")
        st.write("""
    * User-Centric Interface: Intuitive input fields allow users to enter essential health metrics, including glucose levels, BMI, age, and more.
    * Immediate Risk Assessment: Users receive instant feedback regarding their diabetes risk based on the latest clinical guidelines.
    * Educational Resources: Access information on diabetes, risk factors, prevention strategies, and lifestyle modifications to enhance well-being.
    * Data Integrity: Built on a comprehensive dataset, our model ensures reliable and precise predictions.""")

        st.title("How It Works:")
        st.write("""
    * Data Input: Enter your health information in the designated fields.
    * Risk Calculation: The tool analyzes the data using established predictive models to assess your risk of developing diabetes.
    * Personalized Feedback: Receive tailored recommendations based on your results, designed to promote better health outcomes.""")

        st.title("Why Use Our Tool?")
        st.write("Understanding your diabetes risk is essential for informed health decisions. Our Diabetes Risk Predictor empowers users to take proactive steps towards prevention and management, fostering a healthier future.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)
        
    st.write("Disclaimer: This tool is intended for informational purposes only and is not a substitute for professional medical advice. Users are encouraged to consult with a healthcare provider for personalized assessments and recommendations.")


if __name__ == '__main__':
    main()
