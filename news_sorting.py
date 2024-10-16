import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Initialize session state
if 'news_data' not in st.session_state:
    st.session_state.news_data = pd.DataFrame(columns=['title', 'content', 'category'])

# Function to train the model
def train_model(data):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(data['content'], data['category'])
    return pipeline

# Streamlit app
st.title('News Sorting App')

# Input form
with st.form('news_input'):
    title = st.text_input('News Title')
    content = st.text_area('News Content')
    category = st.selectbox('Category', ['Politics', 'Technology', 'Sports', 'Entertainment'])
    submitted = st.form_submit_button('Submit')

    if submitted:
        new_data = pd.DataFrame({'title': [title], 'content': [content], 'category': [category]})
        st.session_state.news_data = pd.concat([st.session_state.news_data, new_data], ignore_index=True)
        st.success('News article added successfully!')

# Display news data
st.subheader('News Articles')
st.dataframe(st.session_state.news_data)

# Train model and make predictions
if len(st.session_state.news_data) > 0:
    model = train_model(st.session_state.news_data)
    
    st.subheader('Predict Category')
    predict_content = st.text_area('Enter news content to predict its category')
    if st.button('Predict'):
        prediction = model.predict([predict_content])[0]
        st.write(f'Predicted category: {prediction}')

# Download data
if len(st.session_state.news_data) > 0:
    csv = st.session_state.news_data.to_csv(index=False)
    st.download_button(
        label="Download news data as CSV",
        data=csv,
        file_name="news_data.csv",
        mime="text/csv",
    )

