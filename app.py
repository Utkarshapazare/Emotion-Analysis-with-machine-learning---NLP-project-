from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Ensure stopwords are downloaded before the app runs
nltk.download('stopwords', quiet=True)

stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Instantiate the stemmer once globally to speed up requests
porter = PorterStemmer()

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('clf.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    
    # Clean text and append emojis with proper spacing
    text = re.sub(r'[\W+]', ' ', text.lower()) + ' ' + ' '.join(emojis).replace('-', '')
    
    # Stem words and remove stopwords
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment ('joy', 'sadness', 'anger', 'fear', or 'neutral')
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
