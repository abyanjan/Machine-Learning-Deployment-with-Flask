from flask import Flask, render_template, url_for,redirect
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# creating a function to calssify a text as postive or negative based on the sentiment scores
def sentiment_label (text):
  sid = SentimentIntensityAnalyzer()
  senti_scores = sid.polarity_scores(text)
  compound_score = senti_scores['compound']

  if compound_score >= 0.05:
    return 'Positive'
  elif compound_score > -0.05 and compound_score < 0.05:
    return 'Neutral'
  else:
    return 'Negative'


# text input form
class Text_Input(FlaskForm):
    text = TextAreaField('Text', validators = [DataRequired()])
    submit = SubmitField('Submit')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

@app.route('/',methods = ['GET','POST'])
def home():
    form = Text_Input()

    if form.validate_on_submit():
        review = str(form.text.data)
        result = sentiment_label(review)
        return render_template('predict.html', form=form, result = result)
    else:
        return render_template('home.html', form = form)



if __name__ == '__main__':
    app.run(debug=False)
