#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Activation,Dropout
from keras.utils import normalize, to_categorical
from keras import backend as K
import numpy as np
from keras.preprocessing import image


img_width ,img_height=100,100

train_data_dir=r'C:\Users\Divyansh\Desktop\datasets\casting_data\casting_data\train'
validation_data_dir=r'C:\Users\Divyansh\Desktop\datasets\casting_data\casting_data\test'
nb_train_samples=6633
nb_validation_samples=715
epochs=1
batch_size=40

if K.image_data_format()=="channels_first":
	input_shape=(3,img_width,img_height)
else:
	input_shape=(img_width,img_height,3)

train_datagen =ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

train_genrator=train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='binary')

validation_genrator=test_datagen.flow_from_directory(
	validation_data_dir,
	target_size=(img_width,img_height),
	batch_size=batch_size,
	class_mode='binary')





# In[ ]:


model = Sequential()


model.add(Conv2D(64,(3,3),input_shape=input_shape))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(train_genrator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_genrator,
	validation_steps=nb_validation_samples//batch_size)












# In[ ]:


import os
import time
#logic of script
data={}
stack=[]


def predictionSystem(currentFile):
    img_pred=image.load_img(currentFile,target_size=(100,100))
    img_pred=image.img_to_array(img_pred)
    img_pred=np.expand_dims(img_pred,axis=0)



    ################################################################################################################

    rslt=model.predict(img_pred)
    print(rslt)
    if rslt[0][0]==1:
        print("damaged item")
    else:
        print("ok item")




start_time = time.time()
seconds = 4
img_dir=r"C:\Users\Divyansh\Desktop\ppppp"
while(True):
    current_time = time.time()
    if((current_time - start_time)%5==4):
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                currentFile=os.path.join(root, file)
                if file.endswith(".jpg"):
                    if currentFile in stack:
                        continue
                    else:
                        print(currentFile)
                        stack.append(currentFile)
                        predictionSystem(currentFile)
            
            
        
        


# In[ ]:





# In[ ]:




