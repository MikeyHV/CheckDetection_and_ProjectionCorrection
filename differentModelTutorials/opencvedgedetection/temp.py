import cv2

img = cv2.imread("check1.jpg")
(H, W) = img.shape[:2]
cv2.imshow("Input", img)
cv2.waitKey(0)
blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),swapRB=False, crop=False)
print("1")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel", "caffe.TEST")
print("2")
net.setInput(blob)
print("3")
hed = net.forward()
print("4")
hed = cv2.resize(hed[0, 0], (W, H))

hed = (255 * hed).astype("uint8")

cv2.imshow("HED", hed)
cv2.waitKey(0)
