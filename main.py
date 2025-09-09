import streamlit as st
import numpy as np    
from data_1 import Data

page_bg = """
<style>
/*      */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: white;
}

/*   */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/*   */
.block-container {
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}

/*  */
h1, h2, h3 {
    color: #FFD700;  /* ذهبي راقي */
    text-shadow: 2px 2px 5px #000;
}

/*  */
p, label, div {
    font-size: 16px !important;
    color: white !important;
}

/*  */
.stButton>button {
    background: linear-gradient(45deg, #00b09b, #96c93d);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.1);
    box-shadow: 0 0 20px rgba(255,255,255,0.6);
}

/*   */
.stTextInput>div>div>input, .stNumberInput input, .stSelectbox div, .stMultiSelect div {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 12px !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.title("💭 Depression Screening App 😔✨")    
st.subheader("🧑‍⚕️ Personal Information")

name=st.text_input("✍️ Enter your name")
age=st.number_input("🎂 Enter your age")
gendar=st.selectbox("🚻 Enter your gender",options=["Male","Female"])
marital_status=st.selectbox("💍 Enter your marital status",options=["Single","Married","Divorced"])
education_level=st.selectbox("🎓 What is your education level",options=["Primary","Secondry","University"])
occupation=st.selectbox("💼 Are you ..." ,options=["Student","Employed","Unemployed"])   

st.subheader("❤️ General Health Information")
chronic_diseases=st.multiselect("🩺 If you have chronic disease tell us ", options=["Diabetes","Hypertension","Heart Problems",None])
family_history=st.multiselect("👨‍👩‍👧 Family History",options=["Diabetes","Hypertension","Heart Problems","Depression",None])

st.subheader("🏃 Life Style")
sleep=st.number_input("😴 Average of sleeping hours")
sleep_quality=st.selectbox("🛌 Quality of Sleep",options=["Regular","Disturbed"])
physical_activity=st.selectbox("🏋️ physical activity",options=["Active","Sedentary"])
diet_type=st.selectbox("🥗 Diet Type ",options=["Balanced","Unbalanced"])
smoke=st.selectbox("🚬 Are you smoking",options=["Yes","No"])
drink=st.selectbox("🍷 Are you drinking Alcohol",options=["Yes","No"])

st.subheader("📋 PHQ-9 Questionnaire")
st.markdown("⚡ Answer each question on a scale from 0️⃣ to 3️⃣")

qs1=st.number_input("1️⃣ Little interest or pleasure in doing things",min_value=0,max_value=3)
qs2=st.number_input("2️⃣ Feeling down, depressed, or hopeless",min_value=0,max_value=3)
qs3=st.number_input("3️⃣ Trouble falling or staying asleep or sleeping too much",min_value=0,max_value=3)
qs4=st.number_input("4️⃣ Feeling tired or having little energy",min_value=0,max_value=3)
qs5=st.number_input("5️⃣ Poor appetite or overeating",min_value=0,max_value=3)
qs6=st.number_input("6️⃣ Feeling bad about yourself",min_value=0,max_value=3)
qs7=st.number_input("7️⃣ Trouble concentrating",min_value=0,max_value=3)
qs8=st.number_input("8️⃣ Moving or speaking slowly / restless",min_value=0,max_value=3)
qs9=st.number_input("9️⃣ Thoughts of self-harm",min_value=0,max_value=3)

st.subheader("📊 Description of Scoring")
st.markdown("""
🌈 **Interpretation of total score:**
- 0-4 👉 Minimal or None  
- 5-9 👉 Mild Depression  
- 10-14 👉 Moderate Depression  
- 15-19 👉 Moderately Severe Depression  
- 20-27 👉 Severe Depression  
""")

with st.spinner("WAIT >>>>>>"):
    st.write("🧾 Result")
    ob=Data(q1=qs1,q2=qs2,q3=qs3,q4=qs4,q5=qs5,q6=qs6,q7=qs7,q8=qs8,q9=qs9,age=age,gen=gendar)
    ob.special_evaluate()
    st.write(ob.present_test())

    st.write("📈 Accuracy of the model")
    st.write(ob.local_evaluate())
    st.plotly_chart(ob.plot_predictions())
    st.plotly_chart(ob.plot_accuracy())
