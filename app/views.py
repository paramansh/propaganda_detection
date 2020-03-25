from flask import render_template, request, redirect
from app import app
from flask_wtf import FlaskForm
from wtforms import TextAreaField
from wtforms.validators import DataRequired

from src import predict

class MyForm(FlaskForm):
  text = TextAreaField('Enter Propganda Text', validators=[DataRequired()])

@app.route('/')
def index():
  form = MyForm()
  return render_template("index.html", form=form)

@app.route('/about')
def about():
  return render_template("about.html")

@app.route('/get_spans', methods=['POST', 'GET'])

def get_spans():
  if request.method == 'POST':
    text = request.form['text']
    text_spans_indices = predict.get_predictions(text)
    text_spans = []
    start_index = 0
    for sp in text_spans_indices:
      text_normal = text[start_index: sp[0]]
      text_propaganda = text[sp[0]: sp[1]]
      start_index = sp[1]
      obj_normal = {
        'text': text_normal,
        'color': 'blue'
      }
      obj_propaganda = {
        'text': text_propaganda,
        'color': 'red'
      }
      text_spans.append(obj_normal)
      text_spans.append(obj_propaganda)

    text_normal = text[start_index: ]
    obj_normal = {
      'text': text_normal,
      'color': 'blue'
    }
    text_spans.append(obj_normal)

    return render_template("spans_output.html", text_spans=text_spans)
  # else:
  #   text = "My name is paramansh singh from patiala"
  #   text_spans_indices = [[5,15], [20, 25], [29, 35]]
  #   text_spans = []
  #   start_index = 0
  #   for sp in text_spans_indices:
  #     text_normal = text[start_index: sp[0]]
  #     text_propaganda = text[sp[0]: sp[1]]
  #     start_index = sp[1]
  #     obj_normal = {
  #       'text': text_normal,
  #       'color': 'blue'
  #     }
  #     obj_propaganda = {
  #       'text': text_propaganda,
  #       'color': 'red'
  #     }
  #     text_spans.append(obj_normal)
  #     text_spans.append(obj_propaganda)

  #   text_normal = text[start_index: ]
  #   obj_normal = {
  #     'text': text_normal,
  #     'color': 'blue'
  #   }
  #   text_spans.append(obj_normal)

  #   return render_template("spans_output.html", text_spans=text_spans)
  redirect('/')