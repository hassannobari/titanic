import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TitanicReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7,16)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(16,1)
        self.sigmoid= nn.Sigmoid()
    def forward(self,x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

@st.cache_resource
def load_model():
    model = TitanicReal()
    model.load_state_dict(torch.load('titanic_real_model.pth'))
    model.eval()
    return model
model = load_model()
#################################################UI####
st.title("پیشبینی زنده ماندن در تایتانیک")
st.write("لطفا اطلاعات مسافر را وارد کنید تا هوش مصنوعی پیش بینی کند:  ")
######input
pclass = st.selectbox("کلاس بلیط(pclass)",[1,2,3])
sex = st.selectbox("جنسیت را وارد کنید ",['men','women'])
age = st.slider('سن را انتخاب کنید : ',0,100,25)
sibsp = st.number_input('تعداد همراهان :',0,10,0)
parch = st.number_input('تعداد وادین/فرزندان :',0,10,0)
fare = st.number_input('قیمت بلیط :',0.0,500.0,20.0)
embarked = st.selectbox('محل سوار شدن :',['england','france','ireland'])
################button###############
if st.button('شروع پیشبینی'):
    sex_val = 1 if sex == 'women' else 0
    embarked_val = {'england':0 ,'france':1,'ireland':2}[embarked]
    #############
    input_df = pd.DataFrame([[pclass,sex_val,age,sibsp,parch,fare,embarked_val]],
                            columns=['Pclass','Sex','Age','Sibsp','Parch','Fare','Embarked'])

    scaler = StandardScaler()
    scaler.mean_ = np.array([2.308,0.352,29.699,0.523,0.381,32.204,0.361])
    scaler.scale_= np.array([0.836,0.478,14.526,1.102,0.806,49.693,0.676])
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled,dtype=torch.float32)
#############
    with torch.no_grad():
        prediction = model(input_tensor)

    prob = prediction.item()
    st.write(f"live chance: **{prob *100:.2f}%**")
    if prob < 0.5:
        st.write("your opponent is dead")
    else:
        st.write("this time you are so clucky \n 'you are survived'")