from flask import Flask, request, jsonify
import json
import os
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import pickle
from newspaper import Article
from keras import backend as K

# Create an instance of Flask
app = Flask(__name__, static_url_path='')
#Get PORT environment variable
port_from_env = os.getenv('PORT')
#Set the port
port = int(port_from_env) if port_from_env is not None else 5000

max_words = 20000
def vectorize_sequences(sequences, dimensions = max_words):
    #   One hot encode
    results = np.zeros((len(sequences), dimensions))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results

@app.route('/', methods=['POST'])
def index():
  #print(json.loads(request.get_data()))
  return jsonify(
    status=200
  )

@app.route('/detect_fake_news', methods=['GET', 'POST'])
def detect_fake_news():
    data = json.loads(request.get_data())
    url = data['url']
    model = load_model('data/fake_news.h5')
    tokenizer = Tokenizer()
    with open('data/tokenizer.pickel', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #url = 'http://nationalreport.net/flint-tap-water-rated-more-trustworthy-than-hillary-clinton/'

    #Parse the url and get title, author and text from URL
    article = Article(url)
    article.download()
    article.parse()
    authors = article.authors
    title = article.title
    text = article.text

    authors_text = ' '.join(authors)
    text_all =  [title+' ' +authors_text+' '+text]
    sequences = tokenizer.texts_to_sequences(text_all)
    sample_data = np.array(sequences)
    X_sample = vectorize_sequences(sample_data)
    y_pred = model.predict(X_sample).ravel()
    K.clear_session()
    if y_pred[0]>0.5:
        result = "Fake"
    else:
        result = "Real"
    return jsonify({"result":result, "Title":title, "Author":authors, "Text":text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
