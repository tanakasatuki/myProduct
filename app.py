import os
from flask import (
     Flask, 
     request, 
     render_template)

from predict import predict

UPLOAD_FOLDER='./static/' # 画像のアップロード先のディレクトリ

app = Flask(__name__, static_folder='./static') # FlaskでAPIを書くときのおまじない

@app.route('/')
def page1():
    return render_template('page1.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/page3')
def page3():
    return render_template('page3.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/result', methods=['GET', 'POST'])
def upload_user_files():
    if request.method == 'POST':
        upload_file = request.files['upload_file']
        img_path = os.path.join(UPLOAD_FOLDER,upload_file.filename)
        upload_file.save(img_path)
        results = predict(img_path) 
        name = results[1]
        url = results[2]
        time = results[3]
        image = results[4]
        return render_template('results.html', name=name, url=url, time=time, image=image)

if __name__ == "__main__":
    app.run(debug=True)