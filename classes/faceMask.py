'''
Team no: 5
Project no: 5
Author: 1. Mohit - prepData() function
        2. Chaitanya - createArchitecture() and trainModel() functions
        3. Prashant - detectMask() function
Description : Class containing functions for data preprocessing, model training and detection
'''

''''''''''''''''''''''''Importing required libraries''''''''''''''''''''''''''


import cv2,os
import numpy as np
import imutils
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



''''''''''''''''''''''''facemask detect class''''''''''''''''''''''''''''''
class FaceMask:


    
    '''
    Constructor
    Arguments : path to dataset, path to prototxt file, path to caffe file, path to detection model
    '''
    def __init__(self, path_to_dataset, path_to_prototxt, path_to_caffe, path_to_model, confidence):
        
        # initialize some variables
        self.data_path = path_to_dataset
        self.prototxt_path = path_to_prototxt
        self.caffe_path = path_to_caffe
        self.conf = confidence
        self.model_path= path_to_model
        self.img_size = 224
        
        # creating label dictionary for classification
        self.categories=os.listdir(self.data_path)
        labels=[i for i in range(len(self.categories))]
        label_dict=dict(zip(self.categories,labels)) 
        print("[INFO] Label dictionary: " + str(label_dict))




        
    '''
    Function for preparing data for training
    Returns : data and target lists
    '''
    def prepData(self):
        print("[INFO] Preparing data...")
        # lists for storing the prepared images
        data=[]
        target=[]
        
        for category in self.categories:
            folder_path=os.path.join(self.data_path,category)
            img_names=os.listdir(folder_path)

            for img_name in img_names:
                img_path=os.path.join(folder_path,img_name)
                img=cv2.imread(img_path)

                try:          
                    resized=cv2.resize(img,(self.img_size,self.img_size))
                    #resizing the colored into 224*224, since we need a fixed common size for all the images in the dataset
                    data.append(resized)
                    target.append(label_dict[category])
                    #appending the image and the label(categorized) into the list (dataset)
                    
                    data=np.array(data)/255.0
                    data=np.reshape(data,(data.shape[0],img_size,img_size,3))
                    target=np.array(target)
                    new_target=np_utils.to_categorical(target)
                    
                    print("[INFO] Data prepared...")
                    return data, new_target
                except Exception as e:
                    print('Exception:',e)
                    #if any exception rasied, the exception will be printed here. And pass to the next image





    '''
    Function to create model architecture
    '''
    def createArchitecture(self):
        print("[INFO] Creating architecture...")
        self.model=Sequential()

        self.model.add(Conv2D(8,(3,3),input_shape=data.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        #The first CNN layer followed by Relu and MaxPooling layers

        self.model.add(Conv2D(16,(3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        #The second convolution layer followed by Relu and MaxPooling layers

        self.model.add(Conv2D(8, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        #The third convolution layer followed by Relu and MaxPooling layers

        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        #Flatten layer to stack the output convolutions from second convolution layer

        self.model.add(Dense(64,activation='relu'))
        #Dense layer of 64 neurons
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(2,activation='softmax'))
        #The Final layer with two outputs for two categories
        print("[INFO] Architecture created...")
        self.model.summary()




        
    '''
    Funciton to train the model
    Arguments : data and target array
    '''
    def trainModel(self, data, target):
        print("[INFO] Training model...")
        # compile the model
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        # split the data into training and testing set
        train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
        # create checkpoint and fit the model
        checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=False,mode='auto')
        history= self.model.fit(train_data,train_target,epochs=20,callbacks=[checkpoint],validation_split=0.2)
        # save the model
        print("[INFO] Model Trained...")
        self.model.save('model.model')
        print("[INFO] Model save as 'trained_model.model'... ")




        
    '''
    Function to run detection
    '''
    def detectMask(self):
        try:
            face_detector = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffe_path)
        except Exception as e:
            print("[INFO] Error loading prototxt and caffe file...")
            print("[Error]" + str(e))

        mask_detector = load_model(self.model_path)

        print("[INFO] Starting webcam. Press 'q' to quit...")
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))

            face_detector.setInput(blob)
            detections = face_detector.forward()

            faces = []
            bbox = []
            results = []

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.conf:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    faces.append(face)
                    bbox.append((startX, startY, endX, endY))

            if len(faces) > 0:
                results = mask_detector.predict(faces)
            elif len(faces) == 0:
                cv2.putText(frame,'No Face Found',(40,20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),2)

            for (face_box, result) in zip(bbox, results):
                (startX, startY, endX, endY) = face_box
                (mask, withoutMask) = result

                label = ""
                if mask > withoutMask:
                    label = "Mask"
                    color = (0, 255, 0)
                else:
                    label = "No Mask"
                    color = (0, 0, 255)

                cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam stopped...")
