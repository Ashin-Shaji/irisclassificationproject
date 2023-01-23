import streamlit as st
import pandas as p
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
st.title('Iris Prediction App')
st.markdown("""
### *This app predicts the iris flower type*:hibiscus:""")
st.sidebar.header('User inputs')
def userinput():
    sepall=st.sidebar.slider('sepal length',4.2,7.7,5.2)
    sepalw=st.sidebar.slider('sepal width',2.0,4.4,3.4)
    petall=st.sidebar.slider('petal length',1.0,6.9,1.5)
    petalw=st.sidebar.slider('sepal width',0.1,2.5,0.3)
    data={'sepall':sepall,'sepalw':sepalw,'petall':petall,
    'petalw':petalw}
    features=p.DataFrame(data,index=[0])
    return features
df=userinput()
st.subheader('User Inputs')
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target
clf=RandomForestClassifier()
clf.fit(x,y)
pred=clf.predict(df)#here there is no test data,so we use user inputs to predict
predprob=clf.predict_proba(df)#predict_proba-used to find the probability among the three(setosa,versicolor,virginica)
st.subheader("""Class Labels and their Corresponding Index""")
st.write(iris.target_names)
st.subheader(' Prediction')
st.write(iris.target_names[pred])
# st.write(pred)
st.subheader('Prediction Probability')
st.write(predprob)


