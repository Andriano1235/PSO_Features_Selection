from _features.fos import FirstOrderStatistic
from _features.glcm import glcm_features
from _features.glcm_four_dir import glcm_four_dir_features
from _features.glds import glds_features
from _features.glrlm import glrlm_features


class Features():
    # structure of the solution
    def __init__(self):
        self.FOS = []
        self.GLCM_mean = []
        self.GLCM_four_dir = []
        self.GLDS = []
        self.GLRLM = []
        self.Labels_FOS = []
        self.Labels_GLCM_mean = []
        self.Labels_GLCM_four_dir = []
        self.Labels_GLDS = []
        self.Labels_GLRLM = []


def extract_texture(images, mask=None):
    FOS = []
    GLCM_mean = []
    GLCM_four_dir = []
    GLDS = []
    GLRLM = []

    for f in images:
        # 1. FOS
        features_FOS, labels_FOS = FirstOrderStatistic(f, mask)
        FOS.append(features_FOS)
        # 2. GLCM
        features_GLCM_mean, labels_mean = glcm_features(f, ignore_zeros=True)
        GLCM_mean.append(features_GLCM_mean)

        features_GLCM_four_dir, labels_four_dir = glcm_four_dir_features(
            f, ignore_zeros=True)
        GLCM_four_dir.append(features_GLCM_four_dir)
        # 3. GLDS
        features_glds, labels_GLDS = glds_features(
            f, mask, Dx=[0, 1, 1, 1], Dy=[1, 1, 0, -1])
        GLDS.append(features_glds)
        # 4. GLRLM
        features_GLRLM, labels_GLRLM = glrlm_features(f, mask, Ng=256)
        GLRLM.append(features_GLRLM)

    # Panggil Class Features untuk membuat dan mengisi kamus features
    features = Features()
    features.FOS = FOS
    features.GLCM_mean = GLCM_mean
    features.GLCM_four_dir = GLCM_four_dir
    features.GLDS = GLDS
    features.GLRLM = GLRLM

    features.Labels_FOS = labels_FOS
    features.Labels_GLCM_mean = labels_mean
    features.Labels_GLCM_four_dir = labels_four_dir
    features.Labels_GLDS = labels_GLDS
    features.Labels_GLRLM = labels_GLRLM

    return features
