from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
import imutils

import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# --------------------

import cv2
import numpy as np
import imutils
import time
# import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.models import model_from_json
import numpy as np
# import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import csv
import serial

# --------face
# Load the cascade

global isvideorun
isvideorun = False
global camera
global exitflag
exitflag = False
global isCamrun
isCamrun = False
global exitflagCam
exitflagCam = False
# //-------------------------------
json_file = open('quality.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("quality.h5")
print("Loaded model from disk")

Classes = ["freshapples", "freshbanana", "freshoranges",
           "rottenapples", "rottenbanana", "rottenoranges"]
# ----------------------------------------
global l

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('FruitSegri_UI_arduino.ui', self)
        
        self.ImageButton.clicked.connect(self.loadImage)
        self.Cambutton.clicked.connect(self.LiveProcess)
        self.ConnectButton.clicked.connect(self.Connect)

        self.PortScanButton.clicked.connect(self.ScanPorts)
        self.ConnectionLabel.setText("Not Connected")

        self.AppleLabel.setText(" ")
        self.BananaLabel.setText(" ")
        self.OrangeLabel.setText(" ")

        ports = ['COM%s' % (i + 1) for i in range(256)]
        ports_available = []
        if ports:
            for port in ports:
                try:
                    s = serial.Serial(port)
                    s.close()
                    ports_available.append(port)
                    self.comboBox.addItem(port)
                except (OSError, serial.SerialException):
                    pass
            print('ports_available :', ports_available)
            
            # self.COM_Port.setText(ports_available[len(ports_available)-1])
        else:
            # self.COM_Port.setText('Not Found')
            self.comboBox.addItem('Not Found')

        self.show()  

    def ScanPorts(self):
        ports = ['COM%s' % (i + 1) for i in range(256)]
        ports_available = []
        if ports:
            for port in ports:
                try:
                    s = serial.Serial(port)
                    s.close()
                    ports_available.append(port)
                    self.comboBox.addItem(port)
                except (OSError, serial.SerialException):
                    pass
            print('ports_available :', ports_available)
            
            # self.COM_Port.setText(ports_available[len(ports_available)-1])
        else:
            # self.COM_Port.setText('Not Found')
            self.comboBox.addItem('Not Found')
        

        
    def Connect(self):
        global sensor_port

        
        if self.ConnectButton.text()== 'Connect':
            COM_port_value =self.comboBox.currentText()
            sensor_port = serial.Serial(COM_port_value, 115200)
            sensor_port.close()
            sensor_port.open()
            self.ConnectButton.setText('DisConnect')
            self.ConnectionLabel.setText("Connected")
            print('Connected')
        else:
            sensor_port.close()
            self.ConnectButton.setText('Connect')
            self.ConnectionLabel.setText("Not Connected")
            print('Dis Connected')

 

    def LiveProcess(self):
        global l
        webcam = cv2.VideoCapture(1)
        if not webcam.isOpened():
            print("Could not open webcam")
            exit()
        # loop through frames
        InData=0
        while webcam.isOpened():

            # read frame from webcam 
            status, frame = webcam.read()           
            try:
                InData=int(sensor_port.read_all().decode())
                print('InData ',InData)
            # self.update()
            # cv2.waitKey(1)
            except:
                pass
            # if InData.decode()

            if status and InData==1:
                # self.update()
                # cv2.waitKey(1)
                print('got command')
                bbox, label, conf = cv.detect_common_objects(frame, model="yolov3") #in BGR fromat
        
                l=0
                for b in bbox:
                    if label[l] == "apple" or label[l] == "orange" or label[l] == "banana":
                        c = [b]
                        d = [label[l]]
                        # x, y, w, h = b
                        # d = [label[l] + ' - ' + output]
                        frame = draw_bbox(frame, c, d, conf)
                        # img_crop = output_image[y:y+h, x:x+w]
                        # img_crop=cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                        # cv2.imwrite('cropped.jpg',img_crop)

                        # get the coordinates of the bounding box
                        x1, y1, x2, y2 = b
                        # extract the region of interest (ROI) from the image
                        roi = frame[y1:y2, x1:x2]
                        
                        # cv2.imwrite('cropped1.jpg',roi)
                        roi=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                                # global filename
                        test_image = cv2.resize(roi, (56, 56))
                        # test_image = image.load_img('cropped1.jpg', target_size = (56, 56))
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(test_image, axis = 0)
                        result = loaded_model.predict(test_image)
                        Class=Classes[result.argmax()]
                        
                        
           
                        print(label[l] + " : " + Class)
                        
                        InData=0
                    
                        if label[l] == "apple":
                            # self.AppleLabel.setText("Apple : "+Class)
                            
                            if(Class=='rottenapples'): # "rottenapples", "rottenbanana", "rottenoranges"
                                tosend='0'
                                self.OrangeLabel.setText("Apple : "+  "Rotten")
                                sensor_port.write(tosend.encode())
                            else:
                                tosend='1'
                                self.OrangeLabel.setText("Apple : "+  "Fresh")
                                sensor_port.write(tosend.encode())
                                

                            
                        if label[l] == "banana":
                  
                            if(Class=='rottenbanana'): # "rottenapples", "rottenbanana", "rottenoranges"
                                tosend='0'
                                self.OrangeLabel.setText("Banana : "+  "Rotten")
                                sensor_port.write(tosend.encode())
                            else:
                                tosend='1'
                                self.OrangeLabel.setText("Banana : "+  "Fresh")
                                sensor_port.write(tosend.encode())
                            
                        if label[l] == "orange":
                            if(Class=='rottenoranges'): # "rottenapples", "rottenbanana", "rottenoranges"
                                tosend='0'
                                self.OrangeLabel.setText("Orange : "+  "Rotten")
                                sensor_port.write(tosend.encode())
                            else:
                                tosend='1'
                                self.OrangeLabel.setText("Orange : "+  "Fresh")
                                sensor_port.write(tosend.encode())

                        
                            

                        l = l+1

            
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Qimage = QImage(
                        frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.ImageToShow.setPixmap(QtGui.QPixmap.fromImage(Qimage))
                self.ImageToShow.setScaledContents(True)     
                self.update()
                if l>0:
                    self.clear_lables()
                cv2.waitKey(1)
            

                        # self.setPhoto(output_image)
            
            elif status:

                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Qimage = QImage(
                        frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.ImageToShow.setPixmap(QtGui.QPixmap.fromImage(Qimage))
                self.ImageToShow.setScaledContents(True)   
                self.update()
                # if l>0:
                #     self.clear_lables()
                cv2.waitKey(1)

            

    def loadImage(self):
        global l
        self.AppleLabel.setText(" ")
        self.BananaLabel.setText(" ")
        self.OrangeLabel.setText(" ")

        ConnectionStatus=self.ConnectionLabel.text()
        if ConnectionStatus=='Connected':
            print('Connetion status = Connected')
        else:
            print('Connetion status = disConnected')
            

        filename, _ = QFileDialog.getOpenFileName(self)
        print('filename ', filename)
        # self.photo.setPixmap(QtGui.QPixmap(filename))
        frame=cv2.imread(filename)
        
        bbox, label, conf = cv.detect_common_objects(frame, model="yolov3") #IN BGR format
        
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # to show on Ui and process
        
        Qimage = QImage(
            frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ImageToShow.setPixmap(QtGui.QPixmap.fromImage(Qimage))

        self.ImageToShow.setScaledContents(True)        
        self.update()
        cv2.waitKey(1)


   
        l=0
        for b in bbox:
            if label[l] == "apple" or label[l] == "orange" or label[l] == "banana":
                c = [b]
                d = [label[l]]
                x, y, w, h = b
                # d = [label[l] + ' - ' + output]
                frame = draw_bbox(frame, c, d, conf)
                
                x1, y1, x2, y2 = b
                # extract the region of interest (ROI) from the image
                img_crop = frame[y1:y2, x1:x2]
                # img_crop = frame[y:y+h, x:x+w]
                # img_crop=cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                # cv2.imwrite('cropped.jpg',img_crop)


                        # global filename
                test_image = cv2.resize(img_crop, (56, 56))

                # test_image = image.load_img('cropped.jpg', target_size = (56, 56))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                result = loaded_model.predict(test_image)
                Class=Classes[result.argmax()]
                print('Class : ',Class)
                
              
                if label[l] == "apple":
                    # self.AppleLabel.setText("Apple : "+Class)
                    
                    if(Class=='rottenapples'): # "rottenapples", "rottenbanana", "rottenoranges"
                        tosend='0'
                        self.OrangeLabel.setText("Apple : "+  "Rotten")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())
                    else:
                        tosend='1'
                        self.OrangeLabel.setText("Apple : "+  "Fresh")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())                       

                    
                if label[l] == "banana":
            
                    if(Class=='rottenbanana'): # "rottenapples", "rottenbanana", "rottenoranges"
                        tosend='0'
                        self.OrangeLabel.setText("Banana : "+  "Rotten")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())
                    else:
                        tosend='1'
                        self.OrangeLabel.setText("Banana : "+  "Fresh")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())
                    
                if label[l] == "orange":
                    if(Class=='rottenoranges'): # "rottenapples", "rottenbanana", "rottenoranges"
                        tosend='0'
                        self.OrangeLabel.setText("Orange : "+  "Rotten")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())
                    else:
                        tosend='1'
                        self.OrangeLabel.setText("Orange : "+  "Fresh")
                        if ConnectionStatus=='Connected':
                            sensor_port.write(tosend.encode())
                    

            l = l+1


        Qimage = QImage(
            frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ImageToShow.setPixmap(QtGui.QPixmap.fromImage(Qimage))

     
        
        self.ImageToShow.setScaledContents(True)   

        if l>0:
            self.clear_lables()
        self.update()

    def clear_lables(self):
        # self.update()
        # time.sleep(1)
        # self.update()
        
        
        print('show result time')
        # cv2.waitKey(10)
        # time.sleep(10)
        # self.AppleLabel.setText(" ")
        # self.BananaLabel.setText(" ")
        # self.OrangeLabel.setText(" ")
        

  

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
