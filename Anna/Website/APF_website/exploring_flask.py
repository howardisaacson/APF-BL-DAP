# Following the Flask Quickstart tutorial and the tutorial here: https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3
# NOTE: Must first ssh -L 5000:127.0.0.1:5000 blpc0

# to run:
# (1) cd to this directory in command line (currently /mnt_home/azuckerman/BL_APF_DAP/APF_website)
# (2) In command line run: export FLASK_APP=exploring_flask.py
# (3) In command line run: flask run                        --> only accessible locally
#                          flask run --host=0.0.0.0         --> Careful! This is externally visible
# (4) In command line run: ssh -L 5000:127.0.0.1:5000 blpc0
# (5) Go to URL http://127.0.0.1:5000/


from flask import Flask
from flask import url_for
from markupsafe import escape
from flask import request
from flask import Flask, render_template
import sqlite3
from werkzeug.exceptions import abort
import tablib
import os

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_post(post_id):
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?', (post_id,)).fetchone()
    conn.close()
    if post is None:
        abort(404)
    return post

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/<name>")
def hello(name):
    #a = input('a = ') --> this will do it in the command line not on the webpage
    #b = input('b = ')
    #c = a + b
    #print('a + b = ' + str(c))
    return f"Hello, {escape(name)}!"


@app.route('/index')
#def index():
#    conn = get_db_connection()
#    entries = conn.execute('SELECT * FROM test_table').fetchall()
#    conn.close()
#    return render_template('index.html', posts=entries)
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)

#@app.route('/')
#def index():
#    return 'Index Page'

#@app.route('/hello') --> this one doesn't work...
#def hello():
#    return 'Hello, World'

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

@app.route('/login')
def login():
    return 'login'

@app.route('/user/<username>')
def profile(username):
    return f'{username}\'s profile'

with app.test_request_context():
    print(url_for('index'))
    print(url_for('login'))
    print(url_for('login', next='/'))
    print(url_for('profile', username='John Doe'))
    
    
@app.route('/<int:post_id>')
def post(post_id):
    post = get_post(post_id)
    return render_template('post.html', post=post)

logfile_table = tablib.Dataset()
with open('/mnt_home/azuckerman/BL_APF_DAP/apf_log_full_16Aug2021.csv') as f:
    logfile_table.csv = f.read()

@app.route("/APF_Logfile")
def disp_logfile():
    return logfile_table.html

results_table = tablib.Dataset()
with open('/mnt_home/azuckerman/BL_APF_DAP/SM_steller_properties/specmatch_results_all_apf.csv') as f:
    results_table.csv = f.read()

@app.route("/APF_Results")
def disp_results():
    return results_table.html