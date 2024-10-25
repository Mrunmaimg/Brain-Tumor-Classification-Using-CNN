import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('C:\\Mrunmai\\ai project dataset\\archive (2)\\pred\\pred45.jpg')

img = Image.fromarray(image)
img=img.resize((64,64))

img = np.array(img)

input_img= np.expand_dims(img, axis=0)

#print(img)
#result = model.predict_classes(input_img)

#print(result)
# Make predictions
predictions = model.predict(input_img)

# Get the predicted class label
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
print(predicted_class)
