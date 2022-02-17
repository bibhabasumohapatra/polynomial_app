import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import lightgbm as lgb
import joblib
import streamlit as st
import  nltk
nltk.download('punkt')

st.header("Internship at polynomial drive")

##########  USER GIVES TEXT INPUT

######### I convert it into pandas dataframe


text = st.text_area("input text")
test_dict = {
    "review": [text]
}


tfidf_vec = joblib.load('tfidf_lgbm_1.pkl')

test_df = pd.DataFrame.from_dict(test_dict)
model_path_lgb ='lgbmclf_1.pkl'
# model_path_lgb = 'https://github.com/bibhabasumohapatra/Internship-at-polynomial-Drive-2022/blob/main/models_saves/lgbmclf_better_accuracy.bin'
model_inference = joblib.load(model_path_lgb)
xtest = tfidf_vec.transform(test_df.review)
value_in_int = model_inference.predict(xtest)

#### show the following values
if value_in_int == 1 or value_in_int == 0:
    #### show its negetive (very negetive)
    st.markdown('## its negetive review')

if value_in_int == 2:
    #### show its negetive (marginally negetive)
    st.markdown('## its negetive review(marginally negetive)')

if value_in_int == 3:
    #### show its nreutral
    st.markdown('## its neutral review')

if value_in_int == 4:
    #### show its positive
    st.markdown('## its positive review(marginally positive)')

if value_in_int == 5:
    #### show its positive
    st.markdown('## its positive review(very positive)')
