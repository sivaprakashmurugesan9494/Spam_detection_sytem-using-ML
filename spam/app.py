import streamlit as st
import pandas as pd
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Load NLTK stopwords
nltk.download('stopwords')

# Load the model and CountVectorizer
model_filename = "spam/MNB.pkl"
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

cv_filename = "spam/countvectorizer.pkl"
with open(cv_filename, 'rb') as file:
    cv = pickle.load(file)

ps = PorterStemmer()

# Preprocess Text Function
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

# Sidebar for Navigation and Settings
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to:", ["Home", "Single Prediction", "Batch Prediction", "Model Evaluation"])
st.sidebar.write("### About the App")
st.sidebar.info(
    "This Spam Detection System uses a trained Multinomial Naïve Bayes model to predict whether a message is Spam or Not Spam."
)
st.sidebar.write("### Model Information")
st.sidebar.text("Model: MultinomialNB")
st.sidebar.text("Feature Extraction: CountVectorizer")

# Main Streamlit App
st.title("📢 Spam Detection System")
st.write(
    "This app detects whether a given message is **Spam** or **Not Spam** using the Multinomial Naïve Bayes model."
)

# Navigation Logic
if options == "Home":
    st.header("Welcome to the Spam Detection App! 🌟")
    st.write(
        "Navigate to the left sidebar to test single messages, upload batch files, or view model performance metrics."
    )
    st.image("https://www.medianama.com/wp-content/uploads/2023/06/spam-g742071dd1_1280.jpg", use_column_width=True)

elif options == "Single Prediction":
    st.header("📧 Single Message Prediction")
    user_input = st.text_area("Enter your message here:")
    if st.button("Predict Message Type"):
        if user_input:
            processed_text = preprocess_text(user_input)
            transformed_text = cv.transform([processed_text]).toarray()
            prediction = model.predict(transformed_text)
            st.success("**Prediction:** " + ("Spam" if prediction[0] == 1 else "Not Spam"))
        else:
            st.warning("Please enter a message to predict.")

elif options == "Batch Prediction":
    st.header("📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file containing messages:", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(data.head())

        if 'message' not in data.columns:
            st.error("The uploaded file must contain a 'message' column.")
        else:
            corpus = [preprocess_text(msg) for msg in data['message']]
            X = cv.transform(corpus).toarray()
            predictions = model.predict(X)
            data['Prediction'] = ['Spam' if pred == 1 else 'Not Spam' for pred in predictions]
            st.write("Prediction Results:")
            st.write(data)

            # Download Results
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("🔗 Download Results", csv, "spam_predictions.csv", "text/csv")

elif options == "Model Evaluation":
    st.header("📊 Model Evaluation")
    if st.button("Show Model Performance"):
        spam = pd.read_csv("/content/spam.csv", encoding='ISO-8859-1')
        spam = spam[['v1', 'v2']]
        spam.columns = ['label', 'message']

        corpus = [preprocess_text(msg) for msg in spam['message']]
        X = cv.transform(corpus).toarray()
        Y = pd.get_dummies(spam['label'], drop_first=True).values.ravel()

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        st.write("**Confusion Matrix:**")
        st.write(confusion_matrix(Y_test, Y_pred))
        st.write("**Accuracy Score:**", accuracy_score(Y_test, Y_pred))

st.sidebar.write("---")
st.sidebar.write("**Developed with Streamlit & Scikit-Learn 📚**")
