import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')

'''
def clean(text):
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() if word not in stopwords.words('english')])
    return review
'''

def tokenize_column(df, col_name):
    tokens = [word.lower() for text in df[col_name] for word in word_tokenize(text)]
    return tokens

def common_words(token, num = 10):
    fdist = FreqDist(token)
    return fdist.most_common(num)

def stop_word_ratio(token):
    stops = set(stopwords.words('english'))
    count_stops = sum(1 for word in token if word in stops)
    return count_stops/len(token)


# Load the dataset into pandas dataframe
print('Loading Data...')
file_name = './dataset/GPT-wiki-intro.csv'
data = pd.read_csv(file_name)
print('Data Loading Complete')

# Sample data in the dataset
print(f'Columns in the dataset:\n{data.columns}')
print(f'Sample Data\n{data.head()}')
print(f'Description:\n{data.info()}')

# Tokenizing relevant columns
tokens_human = tokenize_column(data, 'wiki_intro')
tokens_ai = tokenize_column(data, 'generated_intro')


num = 50
print('Printing most common words')
print(f'Humans: {common_words(tokens_human, num)}')
print(f'AI: {common_words(tokens_ai, num)}')
y_hu = [p[1] for p in common_words(tokens_human, num)]
y_ai = [p[1] for p in common_words(tokens_ai, num)]
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(range(1, num+1,1), y_hu, label='Human Text')
ax.plot(range(1, num+1,1), y_ai, label = 'AI Text')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Frequency')
ax.set_xlabel('Rank')
ax.set_title(f'Frequency of {num} most common words')
ax.legend(loc = 'best')
plt.show()


print('Printing Stop word Ratio')
print(f'Humans: {stop_word_ratio(tokens_human):.3f}')
print(f'AI: {stop_word_ratio(tokens_ai):.3f}')

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data['wiki_intro'])
features = vectorizer.get_feature_names_out()
df_h = pd.DataFrame(matrix.toarray(), columns =features)

vectorizer2 = TfidfVectorizer()
matrix2 = vectorizer2.fit_transform(data['generated_intro'])
features2 = vectorizer2.get_feature_names_out()
df_ai = pd.DataFrame(matrix2.toarray(), columns =features2)

#print(np.max(matrix.toarray()))
fig, ax = plt.subplots(figsize=(15,8), nrows = 2, ncols = 1)
sns.heatmap(np.log1p(df_h), cmap='viridis', ax = ax[0])
sns.heatmap(np.log1p(df_ai), cmap='viridis', ax = ax[1])
plt.tight_layout()
plt.show()


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        return ' '.join(tokens)

    def preprocess_all(self, texts):
        return [self.preprocess(text) for text in texts]
    
    def clean(self, text):
        #remove stopwords
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        return text.lower()