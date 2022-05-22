from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from playsound import playsound



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-rn", "--recognizern", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-ln", "--len", required=True,
	help="path to label encoder")
ap.add_argument("-cn", "--confidencen", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())
recognizern = pickle.loads(open(args["recognizern"], "rb").read())
len = pickle.loads(open(args["len"], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

print("[INFO] starting night video stream...")
vsn = VideoStream(src=1).start()
time.sleep(2.0)

fps = FPS().start()

while True:

	frame = vs.read()

	framen = vsn.read()

	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	framen = imutils.resize(framen, width=600)
	(h, w) = framen.shape[:2]

	n1="name"
	n2="namen"

	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlob)
	detections = detector.forward()



	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue


			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			n1=name

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	fps.update()

	imageBlobn = cv2.dnn.blobFromImage(
		cv2.resize(framen, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlobn)
	detectionsn = detector.forward()

	for i in range(0, detectionsn.shape[2]):

		confidencen = detectionsn[0, 0, i, 2]

		if confidencen > args["confidencen"]:

			boxn = detectionsn[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = boxn.astype("int")

			facen = framen[startY:endY, startX:endX]
			(fH, fW) = facen.shape[:2]

			if fW < 20 or fH < 20:
				continue


			faceBlobn = cv2.dnn.blobFromImage(facen, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlobn)
			vecn = embedder.forward()
			predsn = recognizern.predict_proba(vecn)[0]
			jn = np.argmax(predsn)
			proban = predsn[jn]
			namen = len.classes_[jn]
			n2=namen

			textn = "{}: {:.2f}%".format(namen, proban * 100)
			yn = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(framen, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(framen, textn, (startX, yn),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	fps.update()



	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	cv2.imshow("Frame_night", framen)
	key = cv2.waitKey(1) & 0xFF

	if (n1==n2=="Unkown"):
		playsound("intruder.mp3")
	elif (n1=="Unkown"):
		playsound("sus.mp3")
	elif (n2=="Unkown") :
		playsound("alert.mp3")


	if key == ord("q"):
		break



fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()