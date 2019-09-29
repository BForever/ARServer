from flask import Flask
import time
import sys
from flask import g
from flask import current_app
from flask import request
import logging
import darknet.python.darknet as darknet

app = Flask(__name__)
logger = app.logger
logger.setLevel(logging.DEBUG)
file_log_handler = logging.FileHandler("log")
file_log_handler.setLevel(logging.INFO)
file_log_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s -%(pathname)s- %(filename)s - %(funcName)s - %(lineno)s - %(message)s'))
logger.addHandler(file_log_handler)
start = time.time()
if len(sys.argv) <= 1:
    net = darknet.load_net("darknet/cfg/yolov3-tiny.cfg".encode('utf-8'),
                           "darknet/yolov3-tiny.weights".encode('utf-8'), 0)
else:
    net = darknet.load_net("darknet/cfg/yolov3.cfg".encode('utf-8'),
                           "darknet/yolov3.weights".encode('utf-8'), 0)

meta = darknet.load_meta("darknet/cfg/coco.data".encode('utf-8'))
end = time.time()
logger.info("Time used for loading model: {}s".format(end - start))

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    with open("temp.jpg", 'wb')as image:
        image.write(request.stream.read())
    end = time.time()
    logger.info(
        "Time used for writing image: {}ms".format((end - start) * 1000))
    start = time.time()
    r = darknet.detect(net, meta, "temp.jpg".encode('utf-8'))
    end = time.time()
    logger.info(
        "Time used for detect: {}ms, result: {}".format((end - start) * 1000,
                                                        str(r)))
    return str(r)


if __name__ == '__main__':
    app.run("0.0.0.0", 5000, True, use_reloader=False)
