import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
st.set_page_config(page_title="DIABETES")
from sklearn.model_selection import train_test_split

reg=RandomForestClassifier()
dtc=DecisionTreeClassifier()

df=pd.read_csv("diabetes_data.csv")
df_copy=df.copy()
df_copy['Glucose']=df_copy['Glucose'].replace(0,np.mean(df_copy['Glucose']))
df_copy['BloodPressure']=df_copy['BloodPressure'].replace(0,np.mean(df_copy['BloodPressure']))
df_copy['SkinThickness']=df_copy['SkinThickness'].replace(0,np.mean(df_copy['SkinThickness']))
df_copy['Insulin']=df_copy['Insulin'].replace(0,np.mean(df_copy['Insulin']))
df_copy['BMI']=df_copy['BMI'].replace(0,np.mean(df_copy['BMI']))

X=df_copy.drop("Outcome",axis=1)
y=df["Outcome"]

#front_end
hide_st_style ='''
<style>
footer {visibility: hidden; } I
</style>
'''
st.markdown (hide_st_style, unsafe_allow_html=True)
page_icon=":stethoscope:"
pensiveface="😔"
smiley="😊"


selected=option_menu(
menu_title=None,
options=["Introduction","App","Data"],
icons=["house","file-spreadsheet","database"],
orientation="horizontal",
menu_icon="cast",
default_index=0,
    )

if selected=="Introduction":
    st.title("Health Prediction"+" "+ page_icon)

    st.markdown("<h2 style='font-family: Arial; font-size: 27px;'>Diabetes Prediction App powered by YAJ organisation.</h2>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size: 22px;'>Our app, a testament to AI's potential, empowers individuals by offering personalized diabetes risk predictions based on a comprehensive analysis of multifaceted health variables..</h2>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size: 22px;'>This AI model does not possess a 100% accuracy rate. We will provide you with the accuracy rate below your prediction.</h2>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-size: 28px;'>Hope you Like it...</h2>", unsafe_allow_html=True)
    st.button('Source Code','https://theyajat.streamlit.app/','Source Code',10)


if selected=="App":
    st.markdown("<h2 style='font-size: 38px;'>Start your Prediction...</h2>", unsafe_allow_html=True)
    preg=st.number_input("Enter your Pregnancy: ",min_value=0, max_value=10, value=1, step=1)
    glu=st.number_input("Enter your Glucose Level: ",min_value=0, max_value=200, value=70, step=1)
    bpl=st.number_input("Enter your Blood Pressure Level: ",min_value=60,max_value=130,value=90,step=1)
    stl=st.number_input("Enter your Skin Thickness: ",min_value=0,max_value=100,value=90,step=1)
    insulin=st.number_input("Enter your Insulin Level: ",min_value=0,max_value=900,value=0,step=1)
    bmi=st.number_input("Enter your BMI: " ,min_value=10,max_value=40,value=24)
    dpf=st.number_input("Enter your Diabetes Pedigree Function: ",min_value=0,max_value=3)
    age=st.number_input("Enter your Age: ",min_value=1,max_value=100,value=1)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    reg.fit(X_train,y_train)
    accuracy_level=reg.score(X_test,y_test)
    predicted_val=reg.predict([[preg,glu,bpl,stl,insulin,bmi,dpf,age]])
    if st.button("Predict"):
        if predicted_val==1:
            st.error("The person has Diabetes !!! Take Care"+ " "+pensiveface)
        else:
            st.success("You Don't have any Disease :)"+ " "+smiley)

        st.success(f"The accuracy of Prediction is {int(accuracy_level*100)}%")

if selected=="Data":
    st.subheader("You can download now :)")
    data=df
    csv_file=data.to_csv(index=False)
    st.download_button(label="Download CSV",data=csv_file,file_name="salary_dataset",mime="test/csv",)
    st.table(df.head(21))
