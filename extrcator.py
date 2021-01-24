import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class Extractor:
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(queryDescriptors=des, trainDescriptors=self.last['des'], k=2)
            # matches is a list of tuples where each tuple contains
            # ((key_pt, nearest_neighbour), (key_py, second_nearest_neighbour))
            # where key pt is in the current/query frame and neighbour is in
            # the last/train frame.
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
        if len(ret) > 0:
            ret = np.array(ret)
            # Fundamental matrix would be used to see if the match of a point
            # is along the epi-line in the last image.
            # Fundamental matrix reduces the search space for finding the
            # neighbour in the last/train image.
            model, inliers = ransac((ret[:, 0], ret[:, 1]), FundamentalMatrixTransform, min_samples=8,
                                     residual_threshold=1, max_trials=100)
            ret = ret[inliers]

        # return
        self.last = {'kps': kps, 'des': des}
        return ret
