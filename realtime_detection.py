import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/waste_classifier_model.h5")
classes = ['plastic', 'paper', 'metal', 'organic']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (128,128)) / 255.0
    img = np.reshape(img, (1,128,128,3))

    prediction = model.predict(img)
    label = classes[np.argmax(prediction)]

    cv2.putText(frame, label, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Waste Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()