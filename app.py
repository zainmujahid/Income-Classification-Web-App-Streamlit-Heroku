import joblib
import pandas as pd
import warnings
import streamlit as st
warnings.filterwarnings("ignore")


def main():
    st.title("Income Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Income Classification Web App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.text_input("Age", "17 - 90")
    workclass = st.selectbox("Workclass", ('Private', 'Local-gov', 'Self-emp-not-inc', 'Federal-gov', 'State-gov',
                                           'Self-emp-inc', 'Without-pay', 'Never-worked'))
    fnlwgt = st.text_input("Final Weight (fnlwgt)", "12285 - 1490400")
    education = st.selectbox("Education", ('11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school',
                                           '7th-8th', 'Bachelors', 'Masters', 'Doctorate', '5th-6th', 'Assoc-voc', '9th',
                                           '12th', '1st-4th', 'Preschool'))
    educational_num = st.text_input("Educational Years", "1 - 16")
    marital_status = st.selectbox("Martial Status", ('Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced', 'Separated',
                                                     'Married-spouse-absent', 'Married-AF-spouse'))
    occupation = st.selectbox("Occupation", ('Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Prof-specialty',
                                             'Other-service', 'Craft-repair', 'Adm-clerical', 'Exec-managerial',
                                             'Tech-support', 'Sales', 'Priv-house-serv', 'Transport-moving',
                                             'Handlers-cleaners', 'Armed-Forces'))
    relationship = st.selectbox(
        "Relationship", ('Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative'))
    race = st.selectbox(
        "Race", ('Black', 'White', 'Asian-Pac-Islander', 'Other', 'Amer-Indian-Eskimo'))
    gender = st.selectbox("Gender", ('Male', 'Female'))
    capital_gain = st.text_input("Capital Gain", "0 - 99999")
    capital_loss = st.text_input("Capital Loss", "0 - 4353")
    hours_per_week = st.text_input("Hours Per Week", "40 - 99")
    native_country = st.selectbox("Native Country", ('United-States', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic',
                                                     'Ireland', 'Germany', 'Philippines', 'Thailand', 'Haiti', 'El-Salvador',
                                                     'Puerto-Rico', 'Vietnam', 'South', 'Columbia', 'Japan', 'India', 'Cambodia',
                                                     'Poland', 'Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal',
                                                     'China', 'Nicaragua', 'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador',
                                                     'Yugoslavia', 'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago',
                                                     'Outlying-US(Guam-USVI-etc)', 'France', 'Holand-Netherlands'))
    column_names = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status", "occupation",
                    "relationship", "race", "gender", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    df = pd.DataFrame(columns=column_names)

    if st.button("Predict"):
        df.loc[0] = [age, workclass, fnlwgt, education, educational_num, marital_status, occupation,
                     relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country]
        # Mapping Categorical Features to Numbers
        categorical = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'gender', 'native-country']
        for feature in categorical:
            x_train_labels[feature] = label_ecnoder.fit_transform(
                x_train_labels[feature])
            df[feature] = label_ecnoder.transform(df[feature])
        # Scaling the input with pretrained scalar
        scaled_inpuit = scalar.transform(df)
        prediction = model.predict(scaled_inpuit)[0]
        st.success("Prediction: {}".format(prediction))


if __name__ == '__main__':
    model = joblib.load("model.pkl")  # Importing the pre-trained model
    scalar = joblib.load("s_scaler.pkl")  # Importing the standard scaler
    # Importing the label label_ecnoderoder
    label_ecnoder = joblib.load("label_ecnoder.pkl")
    # Importing the train file
    x_train_labels = joblib.load("train_labels.pkl")
    main()
