import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r"./ML/titanic_pipeline.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure titanic_pipeline.pkl is in the same directory.")
        return None

# App title and description
st.title("🚢 Titanic Survival Predictor")
st.markdown("""
This app predicts whether a passenger would have survived the Titanic disaster
based on their characteristics.
""")

# Load model
model = load_model()

if model is not None:
    # Create input form
    st.sidebar.header("Passenger Information")

    # Input fields
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3],
                                 help="1st = Upper, 2nd = Middle, 3rd = Lower")

    sex = st.sidebar.selectbox("Sex", ["male", "female"])

    age = st.sidebar.slider("Age", 0, 100, 25,
                           help="Age in years")

    fare = st.sidebar.slider("Fare", 0.0, 500.0, 32.0,
                            help="Ticket price in pounds")

    embarked = st.sidebar.selectbox("Port of Embarkation",
                                   ["S", "C", "Q"],
                                   help="S = Southampton, C = Cherbourg, Q = Queenstown")

    # Display current inputs
    st.subheader("Current Passenger Profile")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Class:** {pclass}")
        st.write(f"**Sex:** {sex}")
        st.write(f"**Age:** {age} years")

    with col2:
        st.write(f"**Fare:** £{fare:.2f}")
        st.write(f"**Embarked:** {embarked}")

    # Prediction button
    if st.button("🔮 Predict Survival", type="primary"):
        # Create input DataFrame
        input_data = pd.DataFrame([[pclass, sex, age, fare, embarked]],
                                 columns=["Pclass", "Sex", "Age", "Fare", "Embarked"])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display results
        st.subheader("Prediction Results")

        if prediction == 1:
            st.success("🎉 **SURVIVED** - This passenger likely would have survived!")
            st.write(f"Survival Probability: **{prediction_proba[1]:.1%}**")
        else:
            st.error("💔 **DID NOT SURVIVE** - This passenger likely would not have survived.")
            st.write(f"Survival Probability: **{prediction_proba[1]:.1%}**")

        # Probability bar chart
        prob_data = pd.DataFrame({
            'Outcome': ['Did Not Survive', 'Survived'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })

        st.bar_chart(prob_data.set_index('Outcome'))

        # Historical context
        st.subheader("📊 Historical Context")
        st.info(f"""
        - Overall survival rate on Titanic: ~38%
        - Your predicted survival probability: {prediction_proba[1]:.1%}
        - Women and children had higher survival rates
        - First-class passengers had better survival chances
        """)

    # Model information
    with st.expander("ℹ️ About This Model"):
        st.write("""
        This prediction model was trained on the famous Titanic dataset using:
        - **Algorithm**: Logistic Regression with preprocessing pipeline
        - **Features**: Passenger class, sex, age, fare, and port of embarkation
        - **Accuracy**: ~79% on test data
        - **Preprocessing**: Handles missing values and scales numerical features

        **Note**: This is a simplified model for educational purposes.
        Real survival depended on many more factors including location on ship,
        time of rescue, etc.
        """)

else:
    st.error("Unable to load the model. Please check if the model file exists.")


