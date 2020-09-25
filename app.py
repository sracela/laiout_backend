import base64
import os
import random
import tempfile

from flask import request
from flask import Flask
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/convert', methods=['POST'])
def convert():
    try:
        image_base64 = request.form['image']
    except KeyError:
        return 'Error', 400

    mimetype, image_base64 = image_base64.split(';')
    ext = mimetype.split('/')[-1]

    header = "base64,"
    # if image_base64.startswith(header):
    #     image_base64 = image_base64[len(header):]

    # imgdata = image_base64.decode("base64")
    image_base64 = image_base64.split(',')[1]
    imgdata = base64.b64decode(image_base64) 
    f_name = "input-{}".format(random.randint(1, 1E6))
    # fname = "/tmp/input-{}.{}".format(random.randint(1, 1E6), ext)
    with open("input/{}.{}".format(f_name, ext), 'wb') as f:
        f.write(imgdata)

    dir_path = os.path.dirname(__file__)
    # os.system("{}/script.py {}".format(dir_path, f_name))
    os.system("python /var/local/data/projects/coding/python/pix2code/webapp/webapp_aws.py {}".format(f_name))

    gui = "This is DSL"
    xml = "This is XML"
    with open("output/{}.gui".format(f_name), 'r') as f:
        gui = f.read()
    with open("output/{}.xml".format(f_name), 'r') as f:
        xml = f.read()
    return {
        'gui': gui,
        'xml': xml,
        'image': request.form['image']
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)
