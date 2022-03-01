import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml("mnist_784",version=1,return_X_y=True)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state=9,train_size=7500,test_size=2500)

xtrainscale=xtrain/255.0
xtestscale=xtest/255.0
model=LogisticRegression(solver="saga",multi_class='multinomial').fit(xtrainscale,ytrain)

def getprediction(image):
    im_pil=Image.open(image)
    imagebw=im_pil.convert('L')
    imagebwresize=imagebw.resize((28,28),Image.ANTIALIAS)
    pixelfilter=20
    minpix=np.percentile(imagebwresize,pixelfilter)
    imagebwresizeinvertedscale=np.clip(imagebwresize-minpix,0,255)
    maxpix=np.max(imagebwresize)
    imagebwresizeinvertedscale=np.asarray(imagebwresizeinvertedscale)/maxpix
    testsample=np.array(imagebwresizeinvertedscale).reshape(1,784)
    testpred=model.predict(testsample)
    return testpred[0]
    
