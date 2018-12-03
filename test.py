import score
import shutil
from urllib.request import urlopen
from IPython.display import Image, HTML, display
import requests
import json
import numpy as np
from base64 import b64encode

image_uri = 'https://dm2301files.storage.live.com/y4m6wHcpjt7I-MnRZfL_bgxcr15ESNq_w0i3I6UTJ60hfk-BqL38oMVm1CsM2M09JkvG5zJYTC1onV3Dlrjuz2dbBGgtCObcjM0d8J08EfTAXGHYzWNzHYcYkFJWE5jiDKxRnS7XjhIULL9NkrXH5P4rqeNFWrmyDu8ZGSuTJ3wdpgT-hGEG_63A1BjGzJY2yY4uz2VZbf8pDQLlOY3C7RwOA/coco_dog.jpg?psid=1&width=1024&height=1024&cropMode=center'
service_uri = 'http://23.96.116.25:80/score'

with urlopen(image_uri) as response:
    with open('temp.jpg', 'bw+') as f:
        shutil.copyfileobj(response, f)

def image_to_json(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    base64_bytes = b64encode(content)
    base64_string = base64_bytes.decode('utf-8')
    raw_data = {'image': base64_string}
    return json.dumps(raw_data, indent=2)

# Turn image into json and send an HTTP request to the prediction web service
input_data = image_to_json('temp.jpg')
headers = {'Content-Type':'application/json'}
resp = requests.post(service_uri, input_data, headers=headers)

# Extract predication results from the HTTP response
result = resp.text.strip("}\"").split("[")
predications = result[1].split(",")
print ("Predication results:")
for temp in predications:
    print (temp.strip("]").replace('\\','').strip().strip("\"").strip("}"))
Image('temp.jpg')