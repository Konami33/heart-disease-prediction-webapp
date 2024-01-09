import pyrebase
import streamlit as st
from datetime import datetime
from temp import home, predict, blog, about

firebaseConfig = {
  'apiKey': "AIzaSyByvsc3esXR7wkLmuLdMtdEAFd1wBpXYzc",
  'authDomain': "heart-disease-prediction-8b96b.firebaseapp.com",
  'databaseURL': "https://heart-disease-prediction-8b96b-default-rtdb.firebaseio.com/",
  'projectId': "heart-disease-prediction-8b96b",
  'storageBucket': "heart-disease-prediction-8b96b.appspot.com",
  'messagingSenderId': "213918288033",
  'appId': "1:213918288033:web:7801b15607e386b3c348e0",
  'measurementId': "G-6GC1X9ZF9Y"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

st.sidebar.title("My App")


#Authentication
choice = st.sidebar.selectbox('Login/Register', ["Login", "Register"])
email = st.sidebar.text_input("Enter your email")
password = st.sidebar.text_input("Enter your password", type='password')


if choice == "Register":
    handle = st.sidebar.text_input("Enter your username")
    submit = st.sidebar.button("Register")

    if submit:
        try:
            user = auth.create_user_with_email_and_password(email, password)
            st.success("Account created succesfully!")
            st.balloons()

            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("ID").set(user["localId"])
            st.write("Please login to your account.")
        except Exception as e:
            st.error(f"Login failed: {e}")


if choice == "Login":
    login = st.sidebar.checkbox("Login")
    st.write('<style>'
             'div.row-widget.stRadio > div {'
                 'flex-direction: row;'
                 'align-items: stretch;'
             '}'
            
            'div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]  {'
                'background-color: #565656;'
                'padding: 6px 10px;'
                'margin: 5px;'
            '}'
             
             '</style>', unsafe_allow_html=True)

    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        handle = db.child(user['localId']).child("Handle").get().val()
        st.title("Hello " + str(handle))

        st.write('<style>'
                 'div.row-widget.stRadio > div {'
                 'margin: 0 160px'
                 '}'
                 '</style>', unsafe_allow_html=True)
        bio = st.radio("", ['Home', 'Predict', 'Blog', 'About'])
        if bio == 'Home':
            home()

        if bio == 'Predict':
            predict()

        if bio == 'Blog':
            blog()

        if bio == 'About':
            about()


