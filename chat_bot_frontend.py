mport streamlit as st
import pandas as pd
import openai
import os
from langchain.llms import OpenAI
import pyttsx3
import requests
from bs4 import BeautifulSoup
import pyttsx3
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import speech_recognition as sr
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
import threading
import time



os.environ["secret_key"]="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"
api_key=os.getenv("secret_key")
llm=OpenAI(api_key=api_key,temperature=0.3)



placeholder=st.empty()


llm2=OpenAI(api_key=api_key,temperature=0.3)
llm3=OpenAI(api_key=api_key,temperature=0.3)
llm4=OpenAI(api_key=api_key,temperature=0.3)

df = pd.read_csv("Training.csv.xls")
X_train=df.drop("prognosis",axis=1)
y_train=df["prognosis"]
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)
col=df.columns.values.tolist()
col.pop()

def Med():
    i=False
    time1=False
    dict={}
    with st.chat_message("assistant"):
        st.write("Hi friend!")
        sel=st.selectbox("How are you feeling today",["select","Good","Bad"])
        if sel:
            if sel=="Good":
                st.write("Good to hear this!!!")
            elif sel=="Bad":
                me=st.multiselect("choose a symptoms how you feel",col)
                for symptom in col:
                    if symptom in me:
                        dict[symptom] = 1
                    else:
                        dict[symptom] = 0
                nd = pd.DataFrame([dict])
                predicted_disease = model.predict(nd)
                if me:
                    st.markdown("The disease might be")
                    st.info(predicted_disease[0])
                    i=True
                    prompt1="generate remedies for the disease :"+predicted_disease[0]
                    st.write(llm4.predict(prompt1))
        
    

with st.sidebar:
    pages_select=st.selectbox("Navigation",["Home","Login","chatbot","symptoms to disease","Interactive bot"])

user_data=pd.read_csv("users.csv")

def authenticate_user(username,password,user_data):
    if username in user_data["user_id"].values:
        stored_password = user_data.loc[user_data["user_id"] == username, "passwords"].values[0]

        if password == stored_password:
            st.success("Login successful!")
        else:
            st.error("Incorrect password. Please try again.")
    else:
        st.error("Username not found. Please create a new account.")
        st.write("New to this app?:face_with_rolling_eyes: Create a new  account :smile_cat:")

def create_new_account(new_userid,username, password, user_data,name,email,dob,height,weight,gender,blood):
    if len(password) < 8:
        st.warning("Password must be at least 8 characters long.")
    elif username in user_data["users"].values:
        st.error("Username already exists. Please choose a different username.")
    else:
        val={"user_id": [new_userid],"user_name":[username], "passwords": [password],
             "name":[name],"email":[email], "dob":[dob],"height":[height],"weight":[weight],
             "gender":[gender],"blood":[blood_group]}
        new_account = pd.DataFrame(val)
        user_data = pd.concat([user_data, new_account], ignore_index=True)
        user_data.to_csv("hi.csv", index=False)
        st.success("Account created successfully!")

    
def main():
    st.title(":blue[Login]") 
    with st.form(key="login_form"):
        userid=st.text_input("User Id:")
        password=st.text_input("Enter password:",type="password")
        submit_button=st.form_submit_button("Login")
        if submit_button:
            authenticate_user(userid,password,user_data)
    with st.form(key="signup_form"):
        new_userid=st.text_input("Enter new user id")
        name=st.text_input("Enter your name")
        email=st.text_input("Enter your email id")
        new_password = st.text_input("New Password:", type="password")
        dob=st.date_input("Enter your date of birth")
        height=st.number_input("Enter height(in cm):",0,350)
        weight=st.number_input("Enter weight(in kgs):",0,200)
        gender=st.radio("Select gender:",["Male","Female","Other"])
        blood_group=st.selectbox("Select your blood group:",["A+","B+","A-","B-","O+","O-","AB+","AB-"]) 
        create_account_button = st.form_submit_button("create Account")

        if create_account_button:
            create_new_account(new_userid,name, new_password, user_data,name,email,dob,height,weight,gender,blood_group)

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    




@st.cache_resource
def chat_page(chat):
    st.title(":blue[HEALTH HELPER BOT]:dizzy:")
    with st.chat_message("assistant"):
        st.write("Hi friend:wave:,Ask any querries related to health")
    if chat:
        prompt2=chat+" in kannada language"
        with st.chat_message("user"):
            st.write(chat)
        with st.chat_message("assistant"):
            st.write(llm.predict(chat))
            st.info("kannada")
            st.write(llm3.predict(prompt2))

def recognize_and_speak(target_language='en', max_duration=10):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    with sr.Microphone() as source:
        st.write("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=max_duration)
        except sr.WaitTimeoutError:
            st.write("Speech recognition timed out. Maximum duration reached.")
            return

    try:
        text = recognizer.recognize_google(audio)
        st.write(f"You said: {text}")

        translated_text = translate_text(text, target_language)
        st.write(f"Translated Text: {translated_text}")

        
        speak(translated_text)

    except sr.UnknownValueError:
        st.write("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")

def translate_text(text, target_language='en'):
    return text

stop_speech=False
def speak(text):
    global stop_speech
    engine = pyttsx3.init()
    engine.say(llm3.predict(text))
    engine.runAndWait()
    while not stop_speech:
        time.sleep(0.1)
        
            
def home_page():
    st.title(":blue[Cure AI]:sunglasses:")
    

if pages_select=="Home":
    with placeholder:
        with st.container():
            home_page()
        
elif pages_select=="Login":
    with placeholder:
        with st.container():
            main()

elif pages_select=="chatbot":
    chat=st.chat_input("say something")
    
    with placeholder:
        with st.container():
            chat_page(chat)

    

elif pages_select=="symptoms to disease":
    with placeholder:
        with st.container():
            Med()
elif pages_select=="Interactive bot":
    text=st.chat_input("say something")
    with placeholder:
        with st.container():
            toggle1=st.toggle("Stop speech")
            recognize_and_speak(target_language='en', max_duration=10)
            speech_thread=threading.Thread(target=speak,args=(text,))
            if toggle1:
                stop_speech=True
                speech_thread.join()

            
else:
    with placeholder:
        with st.container():
            home_page()
