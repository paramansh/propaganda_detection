from flask import render_template, request, redirect
from app import app
from flask_wtf import FlaskForm
from wtforms import TextAreaField
from wtforms.validators import DataRequired

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
    return 'got' + str(request.form['text'])
  redirect('/')