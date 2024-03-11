""" This program performs sentiment analysis
 on a dataset of Amazon product reviews. 
"""

# Load the packages that the program will use.
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
import random
from textblob import TextBlob

# Load the english model.
nlp = spacy.load('en_core_web_md')

# Read the csv file that contains the dataset.
df = pd.read_csv('amazon_product_reviews.csv')

# Select column that contains the data we need for the analysis.
reviews = df['reviews.text']

# Clean the data from any missing values.
clean_df = df.dropna(subset=['reviews.text'])


def process_data(text):

    """ This function process and clean the text of the reviews."""

    # Lowercasing the text to have consistency between the words.
    text = text.lower()     

    # Removing special characters and punctuation from the text.
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenization of each word in the text.
    doc = nlp(text)     

    # Removing stop words and doing lemmatization to the tokens.
    tokens = [token.lemma_ for token in doc if token not in STOP_WORDS]

    # Joining tokens into a string.
    processed_text = ' '.join(tokens)    

    return processed_text


def analyze_sentiment(text):

    """ This function will do the sentiment analysis to check the
    polarity, applying the TextBlob model on the processed text and
    will classify it depending on the score. 
    """
    
    analysis = TextBlob(text)       

    # Create a conditional statement to check the polarity score.
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"
    

""" Testing the program with a hardcoded string"""

text = "I do not like this tablet"        

# Applying the function to the text.
cleaned_text = process_data(text)       

# Analyzing the sentiment of the text.
analized_text = analyze_sentiment(cleaned_text)     

# Displaying the results.
print(f"Testing: {cleaned_text}")
print(f"- Has a {analized_text} sentiment.\n")

""" Running the model on random samples of data"""

# Create variables to store random indexex.
random_index = random.randint(0, len(df) - 1)
random_index_2 = random.randint(0, len(df) - 1)

# Select random rows from the data.
chosen_review_A = df.loc[random_index, 'reviews.text']
chosen_review_B = df.loc[random_index_2, 'reviews.text']

# Apply the function to clean the text of the chosen reviews.
cleaned_text_A = process_data(chosen_review_A)
cleaned_text_B = process_data(chosen_review_B)

# Call the variable with cleaned text to check the result.
analized_text = analyze_sentiment(cleaned_text_A)

# Displaying the results.
print(f"- The review number {random_index}: \n {cleaned_text_A}")
print(f"- Has a {analized_text} sentiment.\n")

""" Checking the similarity between two reviews. """

doc_A = nlp(cleaned_text_A)
doc_B = nlp(cleaned_text_B)

similarity = doc_A.similarity(doc_B)

# Variable to store the conclusion after the similarity score.
assumption = ''     

# Create a conditional statement to check the similarity score.
if similarity <= 0.5:
    assumption = "similar"
else:
    assumption = "NOT similar"

# Displaying the results.
print(f"- The review number {random_index}: \n {cleaned_text_A}")
print(f"- And the review number {random_index_2}: \n {cleaned_text_B}")
print(f"- Have a Similarity score of: {round(similarity, 2)}"
f" and we can assume that the two chosen reviews are {assumption}.")