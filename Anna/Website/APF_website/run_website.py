# Basic webpage to display APF data

# to run:
# (1) cd to this directory in command line (currently /mnt_home/azuckerman/BL_APF_DAP/APF_website)
# (2) In command line run: export FLASK_APP=run_website.py
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

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/Home')
#def index():
#    conn = get_db_connection()
#    entries = conn.execute('SELECT * FROM test_table').fetchall()
#    conn.close()
#    return render_template('index.html', posts=entries)
def index():
    #conn = get_db_connection()
    #posts = conn.execute('SELECT * FROM posts').fetchall()
    #conn.close()
    return render_template('homepage.html')

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

gaia_table = tablib.Dataset()
with open('/mnt_home/azuckerman/BL_APF_DAP/gaia_properties.csv') as f:
    gaia_table.csv = f.read()

@app.route("/Gaia_Values")
def disp_gaia():
    return gaia_table.html



