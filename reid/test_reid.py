
import cv2
from reid.reid_model import ReID
from reid.matcher import Matcher

reid = ReID()
matcher = Matcher()

img1 = cv2.imread("data/market1501/query/1485_c3s3_096319_00.jpg")
img2 = cv2.imread("data/market1501/query/1485_c5s3_096462_00.jpg")

feat1 = reid.extract_features(img1)
feat2 = reid.extract_features(img2)

gid1 = matcher.match(feat1)
gid2 = matcher.match(feat2)

print("Image1 ID:", gid1)
print("Image2 ID:", gid2)

