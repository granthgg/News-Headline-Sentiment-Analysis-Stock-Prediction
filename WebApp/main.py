from config import api_key

import streamlit as st
import pickle
import sklearn
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Set the page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to get the latest top 10 world news headlines
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key)

top_headlines = newsapi.get_top_headlines(category='general', language='en', country='in')

st.sidebar.title("Today's News Headlines")
for i, article in enumerate(top_headlines['articles'][:10], 1):
    st.sidebar.markdown(f"{i}. {article['title']}")

# Function to load the necessary data and models
def load_data_and_models():
    with open('WebApp/train_corpus.pkl', 'rb') as file:
        train_corpus = pickle.load(file)

    with open('WebApp/cv.pkl', 'rb') as file:
        cv = pickle.load(file)

    return train_corpus, cv


# Function to set the title and an introduction
def set_title_and_intro():
    st.title('Stock Sentiment Analysis')
    st.markdown("""
    This application predicts the movement of stock prices based on the sentiment of the news headlines. 
    Enter the news headline and get the prediction!
    """)


# Function to select the model for prediction
def select_model():
    model = st.radio(
        "Select the Model for Prediction:",
        ["Logistic Regression", "Naive Bayes Classifier", "Random Forest Classifier"])

    if model == 'Logistic Regression':
        with open('WebApp/lr_classifier.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        st.write('You Selected Logistic Regression')

    elif model == 'Naive Bayes Classifier':
        with open('WebApp/nb_classifier.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        st.write('You Selected Naive Bayes Classifier')

    elif model == 'Random Forest Classifier':
        with open('WebApp/rf_classifier.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        st.write('You Selected Random Forest Classifier')

    return loaded_model


# Function to process the news input
def process_news_input(news_input):
    news_input = re.sub(pattern='[^a-zA-Z]', repl=' ', string=news_input)
    news_input = news_input.lower()
    news_input_words = news_input.split()
    news_input_words = [word for word in news_input_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_news = [ps.stem(word) for word in news_input_words]
    final_news = ' '.join(final_news)

    return final_news


# Function to make the prediction
def make_prediction(final_news, cv, loaded_model):
    temp = cv.transform([final_news]).toarray()
    stock_prediction = loaded_model.predict(temp)

    return stock_prediction


# Function to display the prediction
def display_prediction(stock_prediction):
    if stock_prediction == 1:
        st.success("Prediction: The stock price will remain the same or will go DOWN. :chart_with_downwards_trend:")
    else:
        st.success('Prediction: The stock price will go UP! :chart_with_upwards_trend:')


# Main function to run the app
def main():
    train_corpus, cv = load_data_and_models()
    set_title_and_intro()
    loaded_model = select_model()
    news_input = st.text_input("Enter the News:")

    if news_input:  #make a prediction if there is news input
        final_news = process_news_input(news_input)
        stock_prediction = make_prediction(final_news, cv, loaded_model)
        display_prediction(stock_prediction)


if __name__ == "__main__":
    main()
