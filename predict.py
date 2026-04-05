import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("model/waste_classifier_model.h5")

img_path = "test_image.jpg"
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

classes = ['plastic', 'paper', 'metal', 'organic']
prediction = classes[np.argmax(model.predict(img_array))]

print("Predicted Waste Type:", prediction)