from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def tokenize(text):
    """
    Tokenizes text data
    Args:
    text str: Messages as text data
    Returns:
    words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Normalize text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text to create tokens
    words = word_tokenize(text)

    # remove stop words using nltk package
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    # extract root form of words using lemmatize
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words