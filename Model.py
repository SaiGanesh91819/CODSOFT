import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')                  # IMPORT ALL REQUIRED LIBRARIES
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
import pickle

#-----------------x-----------------------#

train_data=pd.read_csv('sentimentdataset.csv')
text_data=list(train_data['Text'])               # LOAD DATA
sentiment_data = train_data['Sentiment']

#-----------------x-----------------------#          

i=0
for w in text_data :
    text_data[i]=str(re.sub(r'\W+', ' ', w)).lower()   # REMOVE SPECIAL CHARACTERS
    i+=1

#----------------x------------------------#  

wt=list()
for x in text_data:
    words=word_tokenize(x,'english')
    wt.append(words)                             # TOKENIZE
    #print(words)

text_data = wt
del wt

#----------------x------------------------#

stop_words=set(stopwords.words('english'))

for s in text_data:                              # REMOVE STOPWORDS
    for x in s:
        if x in stop_words:
            s.remove(x)

#---------------x------------------------#

lemmatizer = WordNetLemmatizer()
for s in text_data:
    i=0                                          # LEMMATIZE
    for x in s:
        s[i]=lemmatizer.lemmatize(x,'v')
        i+=1
preprocessed_documents_str = [' '.join(tokens) for tokens in text_data]

#---------------x------------------------#

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(preprocessed_documents_str, sentiment_data, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = SVC(kernel='linear', random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

input_text = input("Enter a text string to analyze its sentiment: ")

def preprocess_text(text):
    
    text = re.sub(r'\W+', ' ', text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word, 'v') for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

preprocessed_input_text = preprocess_text(input_text)
input_text_tfidf = tfidf_vectorizer.transform([preprocessed_input_text])
predicted_sentiment = model.predict(input_text_tfidf)[0]
print("The sentiment of the input text is:",predicted_sentiment)

with open('model_pickle','wb') as f:
    pickle.dump(model,f)
with open('tfidf','wb') as f:
    pickle.dump(tfidf_vectorizer,f)

