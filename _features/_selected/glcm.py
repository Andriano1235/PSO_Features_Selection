import numpy as np
from _mahotas.features.texture import haralick, haralick_labels


def glcm_features(f, ignore_zeros=True):

    # 1) Labels
    labels = haralick_labels
    labels_mean = [label + "_Mean" for label in labels]

    # 2) Parameters
    f = f.astype(np.uint8)

    # 3) Calculate Features: Mean and Range
    features = haralick(f,
                        ignore_zeros=ignore_zeros,
                        compute_14th_feature=True,
                        return_mean=True)

    features_mean = features[0:8]
    # features_range = features[14:]

    return features_mean, labels_mean,
