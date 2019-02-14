from flask import Flask, request
import pickle
import mxnet as mx
from mxnet import nd
import csv
import numpy as np

net = pickle.load(open("model_ccfraud1.Sequential" , "rb"))
url = "creditcard.csv"
app = Flask(__name__)

@app.route('/')
def return_table():
        testing = list()
        reader = csv.reader(open(url))
        idx = 0
        for row in reader:
            testing.append(list(row))
            idx += 1
            if idx == 1000:
                break
        input = testing[1:]
        input = np.asarray(input)
        input = nd.array(input[:, 1:-1], ctx=mx.cpu())
        output = net(input)
        tab = "<tr><th>Time</th><th>V1</th><th>V29</th><th>Amount</th></tr>\n"
        for idx in range(len(output)):
            if (output[idx] > 0.3) == True:
                tab += "<tr style='background-color: red'><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (testing[idx][0], testing[idx][1], testing[idx][-3], testing[idx][-2])
            else:
                tab += "<tr style='background-color: green'><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>\n" % (testing[idx][0], testing[idx][1], testing[idx][-3], testing[idx][-2])
        return "<table>%s</table>" % tab

app.run(host="127.0.0.1", port="100")