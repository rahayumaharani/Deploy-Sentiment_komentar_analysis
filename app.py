from flask import Flask, request, render_template
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
from joblib import load
import pickle
from nltk.corpus import stopwords
from flask import Markup

app = Flask(__name__, static_url_path="/static")

stopwords_Comment_Instagram = stopwords.words("indonesian")
more_stopword = ['username', 'hadeww', 'lg', 'ATT', 'bgt', 'skrg',
                 'd', 'elah', 'krn', 'rt', 'dr', 'pd','ber', 'Ckck',
                 'Mna', 'mna', 'eneg', 'yg', 'pny', 'jd', 'aj', 'dg',
                 'sgj','Mrsk', 'pny', 'g', 'mua', 'ttp', 'ny', 'tp',
                 'gt', 'jg', 'ni', 'haltis', 'M', 'lbh', 'wes', 'org',
                 'la', 'curh', 'am', 'gw', 'dr', 'az', 'd', 'k', 'KD',
                 'or', 'n', 'an', 'bc', 'nmx', 'KL', 'sm', 'ky', 'G',
                 'la', 'bs', '...',]
stopwords_ind = stopwords_Comment_Instagram + more_stopword

dataset_komentar_instagram_cyberbullying = pd.read_csv(
    "app/dataset_komentar_instagram_cyberbullying.csv"
)

vocab = list(pickle.load(open("app/kbest_feature.pickle", "rb")))
model = load("app/model_sentiment_analisis.model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def my_form_post():
    if request.method == "POST":
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # convert to lowercase and remove numbers
        text1 = request.form["text1"].lower()
        text_final = re.sub(r"\d+", "", text1)

        # remove punctuation
        text_final = re.sub(r"[^\w\s]", "", text_final)

        # remove stop words
        processed_doc1 = " ".join(
            [word for word in text_final.split() if word not in stopwords_ind]
        )

        # stemming
        processed_doc1 = stemmer.stem(processed_doc1)

        # sentiment analysis
        X = []
        for word in processed_doc1.split():
            if word in vocab:
                X.append(vocab.index(word))

        if len(X) > 0:
            X_test = [0]*1000
            for index in X:
                X_test[index] = 1
            y_pred = model.predict([X_test])
            pos_prob = round(model.predict_proba([X_test])[0][1], 2)
            neg_prob = round(model.predict_proba([X_test])[0][0], 2)

            if len(y_pred) > 0:
                if y_pred[0] == 0:
                    sentimen = "Negatif"
                else:
                    sentimen = "Positif"

                return render_template(
                    "index.html",
                    final=sentimen,
                    text1=text_final,
                    positive_prob=pos_prob,
                    negative_prob=neg_prob,
                )
            else:
                return render_template(
                    "index.html", final="Sentimen netral", positive_prob=0, negative_prob=0)
    
    # add a return statement here to ensure the function always returns a response
    return render_template("index.html", final="Sentimen netral", positive_prob=0, negative_prob=0)


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
