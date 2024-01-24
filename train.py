import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET

from tensorflow.keras import layers, models

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def parse_xml_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []

    for image in root.findall('image'):
        name = image.get('name')
        width=float(image.get('width'))
        height=float(image.get('height'))
        for box in image:
            xtl= float(box.get('xtl'))
            ytl= float(box.get('ytl'))
            xbr=float(box.get('xbr'))
            ybr=float(box.get('ybr'))
            annotations.append([name,xtl,ytl,xbr,ybr,width,height])

    return annotations

batch_size = 1
train_directory = "images/"
annotation_file = "annotations.xml"
annotations = parse_xml_annotations(annotation_file)
np.random.shuffle(annotations)
x_train=[]
y_train=[]


for annotation in annotations:
                #image_filename, x, y, width, height = annotation.strip().split(',')
                image_filename = annotation[0]
                xtl=annotation[1]
                ytl =annotation[2]
                xbr=annotation[3]
                ybr=annotation[4]
                width=annotation[5]
                height=annotation[6]
                img = load_img(image_filename, target_size=(256, 256))
                #plt.imshow(img)
                img_array = img_to_array(img) / 255.0

                label = [xtl,ytl,xbr,ybr,width,height]
                
                x_train.append(img_array)
                y_train.append(label)
print(x_train)
print(y_train)
X_train=np.array(x_train,dtype=float)
Y_train=np.array(y_train,dtype=float)




# Example usage

#print('hi!!!')
#train_datagen = custom_generators(train_directory, annotation_file, batch_size)
#print(train_datagen)

# Create and compile your model
# ...

# Train the mo
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6))  # Output layer with 2 neurons for x and y coordinates

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

class CustomTerminationCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Add your custom termination condition here
        loss= logs['val_loss']
        print('epoch::{epoch}, loss::{loss}')
        if logs['val_loss'] < 3:
            self.model.stop_training = True

custom_termination_callback = CustomTerminationCallback()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train,Y_train, validation_data=(X_train,Y_train), epochs=10, callbacks=[early_stopping])

print(model.predict(X_train))
print('actual value:{Y_train}')