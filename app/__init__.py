from flask import Flask

app = Flask(__name__, instance_relative_config=True)

from app import views

app.config.from_object('config')
import os
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY