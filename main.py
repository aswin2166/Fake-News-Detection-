
from flask import Flask, request, render_template
import pandas as pd
import nltk
import pickle
from nltk.corpus import wordnet,stopwords
from nltk.stem import WordNetLemmatizer
lemmatize=WordNetLemmatizer()
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


app = Flask(__name__)
fake=pd.read_csv('Fake.csv')
fake['label']='fake'
real=pd.read_csv('True.csv')
real['label']='real'
fake1=fake[:2000]
real1=real[:2000]
df1=pd.concat([fake1,real1],ignore_index=True)

# load_model=pickle.load(open('model.pkl','rb'))
x=df1[:2000]
y=df1[:2000]


def pos_tagger(nltk_tag):
  nltk_tag=str(nltk_tag)
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None

def vectorize(x):
    text = []
    for i in df1.text:
        tk = nltk.sent_tokenize(i)
        text.append(tk)
    a1=[]
    for i in range(len(text)):
        a=str(text[i])
        a=re.sub('[^a-zA-Z]',' ',a)
        pos_tagged_text=nltk.pos_tag(nltk.word_tokenize(a))
        a1.append(pos_tagged_text)

    wordnet_tagged=list(map(lambda x: (x[0],pos_tagger(x[1])),a1))

    lemmatized_sentence=[]
    for word,tag in wordnet_tagged:
        if word not in set(stopwords.words('english')):
            if tag is None:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(lemmatize.lemmatize(word))
        else:
            pass


    tfidf_vectorizer = TfidfVectorizer()
    vector = tfidf_vectorizer.fit_transform([str.join(' ', x) for x in lemmatized_sentence])
    feature_name_tfidf = tfidf_vectorizer.get_feature_names_out()
    array_tfidf = vector.toarray()
    x_tfidf = pd.DataFrame(array_tfidf)
    return x_tfidf
x=vectorize(x)
x_train_tf,x_test_tf,y_train_tf,y_test_tf=train_test_split(x,y,test_size=.2,random_state=0)
model = DecisionTreeClassifier()
model.fit(x_train_tf,y_train_tf)

def fake_new_detector(text):
  a1=[]
  for i in range(len(text)):
    a=str(text[i])
    a=re.sub('[^a-zA-Z]',' ',a)
    pos_tagged_text=nltk.pos_tag(nltk.word_tokenize(a))
    a1.append(pos_tagged_text)

  wordnet_tagged=list(map(lambda x: (x[0],pos_tagger(x[1])),a1))

  lemmatized_sentence=[]
  for word,tag in wordnet_tagged:
    if word not in set(stopwords.words('english')):
      if tag is None:
        lemmatized_sentence.append(word)
      else:
        lemmatized_sentence.append(lemmatize.lemmatize(word))
    else:
      pass

  vector=tfidf_vectorizer.transform([str.join(' ', x) for x in lemmatized_sentence])
  array_pred=vector.toarray()
  x_pred=pd.DataFrame(array_pred)

  prediction=model.predict(x_pred)
  return prediction


@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('textarea')
        pred=fake_new_detector(text)
        print(pred)
        return render_template('index.html',prediction=pred)

    else:
        return render_template('index.html',prediction="something went wrong")

if __name__ == '__main__':
    app.run()