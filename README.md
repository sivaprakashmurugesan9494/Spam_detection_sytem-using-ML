📧 Spam Detection System Using Machine Learning
This project is a Spam Detection System built using Multinomial Naïve Bayes and deployed as an interactive web application using Streamlit. The app allows users to predict whether a given message is Spam or Not Spam and provides batch processing for bulk message predictions.

🌟 Features
Single Message Prediction: Enter a message and get an instant prediction.
Batch Prediction: Upload a CSV file of messages and get predictions for all.
Model Evaluation: View the model's performance metrics, including accuracy and a confusion matrix.
Interactive UI: User-friendly interface with clear navigation.
🚀 Live Demo

Access the app here.

https://spamdetection0.streamlit.app/

📊 How It Works
Text Preprocessing:

Removes non-alphabetic characters.
Converts text to lowercase.
Removes stopwords (e.g., "the", "and").
Stems words to their base form using PorterStemmer.
Feature Extraction:

Uses CountVectorizer to create a Bag of Words representation.
Prediction:

Employs a trained Multinomial Naïve Bayes model to classify messages.
Batch Processing:

Allows CSV uploads for batch prediction.
🔍 Example Usage
Single Message Prediction
Navigate to the Single Prediction section.
Enter a message:
"Win a free iPhone! Click here to claim your prize."
Click "Predict Message Type."
The app will display the prediction: Spam.
Batch Prediction
Navigate to the Batch Prediction section.
Upload a CSV file with a column named message.
View the predictions and download the results.



If you have any issues or questions, feel free to open an issue on the repository or contact me directly.

Let me know if you'd like further tweaks! 😊

This Spam Detection System uses a trained Multinomial Naïve Bayes model to predict whether a message is Spam or Not Spam.

