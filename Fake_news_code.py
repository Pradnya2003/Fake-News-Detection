# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import re
import string
import matplotlib.pyplot as plt

# Set the path for the dataset files
true_path = 'True.csv'
fake_path = 'Fake.csv'

# Read the data
true_news = pd.read_csv(true_path)
fake_news = pd.read_csv(fake_path)

# Add 'class' column to both datasets
true_news['class'] = 1
fake_news['class'] = 0

# Remove 10 entries from the end of each dataset for manual testing
true_manual_testing = true_news.tail(10)
fake_manual_testing = fake_news.tail(10)
true_news = true_news.iloc[:-10]
fake_news = fake_news.iloc[:-10]

data_manual_testing = pd.concat([true_manual_testing, fake_manual_testing])

# Concatenate the remaining entries of both datasets
data_merge = pd.concat([true_news, fake_news])

# Print the count of articles per subject and plot them in a bar chart
subject_count = data_merge['subject'].value_counts()
print(subject_count)
subject_count.plot(kind='bar')
plt.show()

# Print the count of fake and true news articles and plot them in a pie chart
class_count = data_merge['class'].value_counts()
print(class_count)
class_count.plot(kind='pie', autopct='%1.1f%%')
plt.show()

# Drop unnecessary columns
data = data_merge.drop(columns=['title', 'subject', 'date'])

# Apply data shuffling
data = data.sample(frac=1).reset_index(drop=True)

# Check for missing values
print(data.isnull().sum())

# Define the filtering function
def filtering(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    return text

# Apply filtering function to 'data'
data['text'] = data['text'].apply(filtering)

# Split the data into text and class labels
x = data['text']
y = data['class']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert text data into numerical vectors using TfidfVectorizer
vector = TfidfVectorizer()
x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)

# Train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred_log_reg = log_reg.predict(x_test)

# Print the classification report for Logistic Regression model
print("Logistic Regression Classification Report:\n")
print(classification_report(y_test, y_pred_log_reg))

# Train a Decision Tree Classifier model
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
y_pred_dec_tree = dec_tree.predict(x_test)

# Print the classification report for Decision Tree Classifier model
print("Decision Tree Classifier Classification Report:\n")
print(classification_report(y_test, y_pred_dec_tree))

# Define a function to convert numeric class labels to text labels
def output_label(n):
    return "True News" if n == 1 else "Fake News"

# Define a function for manual testing
def manual_testing(news):
    news = filtering(news)
    news_vector = vector.transform([news])
    pred_log_reg = log_reg.predict(news_vector)
    pred_dec_tree = dec_tree.predict(news_vector)
    return f"Logistic Regression Prediction: {output_label(pred_log_reg[0])}\nDecision Tree Prediction: {output_label(pred_dec_tree[0])}"

# Take an input news text from the user and call 'manual_testing' function
input_news = input("Enter the news text to test: ")
print(manual_testing(input_news))
