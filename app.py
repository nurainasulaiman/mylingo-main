from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/parallax.html')
def other_page():
    return render_template('parallax.html')

@app.route('/choosepath.html')
def choosepath():
    return render_template('choosepath.html')

@app.route('/memory.html')
def memory():
    return render_template('memory.html')

@app.route('/translate.html')
def translate():
    return render_template('translate.html')

@app.route('/signlogin.html')
def signlogin():
    return render_template('signlogin.html')

@app.route('/basics.html')
def basic():
    return render_template('basics.html')

@app.route('/lesson.html')
def lesson():
    return render_template('lesson.html')

@app.route('/ch1.html')
def ch1():
    return render_template('ch1.html')

@app.route('/ch2.html')
def ch2():
    return render_template('ch2.html')

# @app.route('/temp.html')
# def temp():
#     return render_template('temp.html')

if __name__ == '__main__':
    app.run(debug=True)