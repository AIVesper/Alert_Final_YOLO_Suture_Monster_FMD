import cv2.cv2 as cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import numpy
import base64

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'chatbot-108aea001-296006.appspot.com'})

def storeImage(image, uuid):#frame,id
    print('storeImage')
    #img = cv2.cvtColor(numpy.asarray(integrationModel.frame["image"]),cv2.COLOR_RGB2BGR)
    img = image
    img = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    bucket = storage.bucket()
    img = base64.b64decode(img)
    blob = bucket.blob('image/'  +"nomask-"+uuid+'.png')
    blob.upload_from_string(img)
    frameUrl =  blob.public_url
    blob.make_public()
    print(frameUrl)
    return frameUrl