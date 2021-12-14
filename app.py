from flask import Flask, render_template, request, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = "secret key"
extension = ['png', 'jpg', 'jpeg', 'gif', 'jfif']
dic = {0: 'Afghan_hound', 1: 'African_hunting_dog', 2: 'Beagle',3: 'Bulldog', 4: 'Chihuahua',5: 'Dalmation',
        6: 'Doberman', 7: 'German_shepherd',
        8: 'Golden_retriever', 9: 'Great_Dane', 10: 'Japanese_spaniel', 11: 'Labrador_retriever', 12: 'Pomeranian',
        13: 'Pug', 14: 'Rottweiler',
        15: 'Saint_Bernard', 16: 'Shih-Tzu', 17: 'Siberian_husky', 18: 'Tibetan_mastiff', 19: 'Poodle'}

model = load_model('dogmodel.h5')


def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predictions = np.argmax(model.predict(x), axis=1)
    return dic[predictions[0]]


@app.route('/', )
def main():
    return render_template('home.html')


@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img.filename == '':
            pred = flash('No image selected for uploading')
            return render_template("index.html", prediction=pred)
        else:
            ext = img.filename.rsplit(".",1)[1].lower()
            if ext not in extension:
                pred = flash('Allowed image types are - png, jpg, jpeg')
                return render_template("index.html", prediction=pred)
            else:
                img_path = "static/" + img.filename
                img.save(img_path)
                pred = predict_label(img_path)
                flash(f'Image successfully detected and displayed below...')
                return render_template("index.html", prediction=pred, img_path=img_path)
    return None


if __name__ == '__main__':
    app.run(debug=True)