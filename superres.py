import os
import cv2

sourcePath = 'source'
resultPath = 'result'

files = [f for f in os.listdir(sourcePath) if os.path.isfile(os.path.join(sourcePath, f))]
numOfFiles = len(files)
index = 0

sr = cv2.dnn_superres.DnnSuperResImpl_create()
superresModelPath = "FSRCNN_x3.pb"
sr.readModel(superresModelPath)
sr.setModel("fsrcnn", 3)

for f in files:
	print(f + " ..." + str(index) + "/" + str(numOfFiles))
	index = index + 1
	
	sourceImg = cv2.imread(os.path.join(sourcePath, f))
	resImg = sr.upsample(sourceImg)
	cv2.imwrite(os.path.join(resultPath, f), resImg)