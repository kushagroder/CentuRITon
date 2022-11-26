import cv2
import torch.hub
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import model
from PIL import Image
from torchvision import transforms
from grad_cam import BackPropagation
import time 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
import vlc
from os.path import dirname, join
from twilio.rest import Client
import urllib.request
import smbus
import json
import Jetson.GPIO as gpio

# Setting the pin which will control the arduino
led_pin = 7

current_dir = dirname(__file__)

# Alarm sound file
file = 'alarm.mp3'
# Sound player start
p = vlc.MediaPlayer(os.getcwd()+'/'+file)

timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()

face_cascade = cv2.CascadeClassifier(os.getcwd()+'/haar_models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(os.getcwd()+'/haar_models/haarcascade_eye.xml') 
MyModel="BlinkModel.t7"

shape = (24,24)
classes = [
    'Close',
    'Open',
]

eyess=[]
cface=0

sens=30
def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])    
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface=1
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = image[y:y+h, x:x+w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        sensi=40
        eyes = eye_cascade.detectMultiScale(face,1.3, sensi) 
        i=0
        for (ex,ey,ew,eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye = face[y:y + h, x:x + w]
            eye = cv2.resize(eye, shape)
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye, cv2.resize(face, (48,48))])
            i=i+1
    cv2.imwrite(os.getcwd()+'/temp-images/display.jpg',image) 
    

def eye_status(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    #print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]

def func(imag,modl):
    drow(images=[{'path': imag, 'eye': (0,0,0,0)}],model_name=modl)

def drow(images, model_name):
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    net = model.Model(num_classes=len(classes))
    #
    checkpoint = torch.load(os.path.join(os.getcwd()+'/model', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    flag =1
    status=""
    for i, image in enumerate(images):
        if(flag):
            preprocess(image)
            flag=0
        if cface==0:
            image = cv2.imread(os.getcwd()+"/temp-images/display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.getcwd()+'/temp-images/display.jpg',image)

            #Stop the application of brakes and seat vibrations
            gpio.output(led_pin, gpio.LOW)

            timebasedrow= time.time()
            timebasedis= time.time()
            timerundrow= time.time()
            timerundis= time.time()
        else:
            timerundrow= time.time()
            if((timerundrow-timebasedrow)>3):
                if(not(p.is_playing())):
                    p.play()
                image = cv2.imread(os.getcwd()+'/temp-images/display.jpg')
                image = cv2.putText(image, 'Drowsy', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imwrite(os.getcwd()+'/temp-images/display.jpg',image)

                #Start the application of brakes and seat vibrations
                gpio.output(led_pin, gpio.HIGH)


        

def main():
    

    gpio.setmode(gpio.BOARD)
    gpio.setup(led_pin, gpio.OUT, initial=gpio.LOW)

    frame_rate = 1
    prev = 0 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while True:
        global eyess
        global cface    
        eyess=[]
        cface=0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)        
        time_elasped = time.time() - prev
        ret, img = cap.read()
      
        if(time_elasped > 1./frame_rate):
        	prev = time.time() 
        	if(ret):
        		cropped = img[211:599, 76:480]
        		flippity = cv2.flip(cropped, 1)                        
        		cv2.imwrite(os.getcwd()+'/temp-images/img.jpg',flippity) 
        		func(os.getcwd()+'/temp-images/img.jpg',MyModel)
        	else:
        		print('Ret is false')
        		exit()
        	
def disp():
    while 1:
        try:
            img = cv2.imread(os.getcwd()+'/temp-images/display.jpg')
            cv2.imshow(os.getcwd()+'/temp-images/image',img)
            k = cv2.waitKey(30) & 0xff
        except:
            ...
        
if os.path.exists('/dev/video0'):
	cap = cv2.VideoCapture(0)
elif os.path.exists('/dev/video1'):
	cap = cv2.VideoCapture(1)
else:
	cap = cv2.VideoCapture(2)

#cap.set(3, 1920)
#cap.set(4, 1080)
cv2.waitKey(0)
cap.set(10, 16)#brightness
cap.set(11, 25)#contrast
cap.set(12, 50)#saturation
cap.set(17, 4777)#white balance

timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()
        
myTime1 = time.time()
d = threading.Thread(target=disp, name='disp')
m = threading.Thread(target=main, name='main')
d.start()
m.start()
