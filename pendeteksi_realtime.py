from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

#untuk memuat model
classifier =load_model('C:\\Users\\ikhram\\PycharmProjects\\PendeteksiHelm\\model.h5')

#pembuatan label, input camera dan output filevideo
class_labels = ['Helm ON','NO Helm']
cap = cv2.VideoCapture(0)

outputFile = "hasil_ujicoba.mp4"
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
vid_writer = cv2.VideoWriter(outputFile, fourcc, 30.0, (1280,720))

start_point = (15, 15)
end_point = (300, 80) 
thickness = -1

while True:
    # untuk mengambil 1 frame dari kamera yang sedang menangkap gambar secara real-time
    ret, frame = cap.read()
    labels = []
    
    gray = cv2.resize(frame,(224,224))
    roi = gray.astype('float')/255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)

    # membuat prediksi dengan menggunakan model

    preds = classifier.predict(roi)[0]
    label=class_labels[preds.argmax()]

    
    if(label=='NO Helm'):
        image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    if(label=='Helm ON'):
        image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
    cv2.imshow('Pendeteksi Helm',frame)
    img = cv2.resize(frame, (1280,720))
    vid_writer.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid_writer.release()
cap.release()
cv2.destroyAllWindows()


























