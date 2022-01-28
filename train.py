
# Command yang digunakan untuk running file train.py
# python train.py -d dataset -p plot.png -m model.h5

# import package yang diperlukan
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# buat parser argumen dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="graph.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="model.h5",
	help="path to save output model")
args = vars(ap.parse_args())

#-----------------------------------------------------------------------

#penentuan jumlah proses batching data
INIT_LR = 1e-3
EPOCHS = 20
BS = 4

# untuk mengambil list gambar di directory dataset dan kemudian di inisialisasi
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# untuk loop ulang image paths
for imagePath in imagePaths:
	# mengextract class label dari filename
	label = imagePath.split(os.path.sep)[-2]

	# untuk load gambar, mengubah tipe warna, dan mengubah ukuran (scaling)
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	labels.append(label)

# Ubah data dan labels menjadi NumPy arrays saat scaling pixel (labeling)
data = np.array(data) / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(labels)

# mempartisi data menjadi training dan testing dan menggunakan 80% dari
# data untuk training dan sisa 20% untuk testing (masih termasuk proses labeling)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# inisialisasi augmentasi data
trainAug = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# implementasi Arsitektur model
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# pembangunan head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

#----------------------------------------------------------------------

# compile model yang sudah kita bangun
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training ...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#-----------------------------------------------------------------------


# plot training loss dan accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Helmet Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])


print("[INFO] saving helmet detector model...")
model.save(args["model"], save_format="h5")
