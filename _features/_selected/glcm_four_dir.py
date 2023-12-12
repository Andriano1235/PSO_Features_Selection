import numpy as np
from _mahotas.features.texture_four_dir import haralick


def glcm_four_dir_features(f, ignore_zeros=True):

    # 1) Labels
    labels_0 = ["GLCM_ASM_0",
                "GLCM_Contrast_0",
                "GLCM_InverseDifferenceMoment_0",
                "GLCM_DifferenceVariance_0",
                "GLCM_DifferenceEntropy_0",
                "GLCM_Information2"]
    labels_45 = ["GLCM_SumOfSquaresVariance_45",
                 "GLCM_SumAverage_45",
                 "GLCM_DifferenceVariance_45",
                 "GLCM_MaximalCorrelationCoefficient_45"]
    labels_90 = ["GLCM_Correlation_90",
                 "GLCM_SumAverage_90",
                 "GLCM_SumVariance_90",
                 "GLCM_DifferenceVariance_90",
                 "GLCM_Information1_90"]
    labels_135 = ["GLCM_SumVariance_135",
                  "GLCM_SumEntropy_135",
                  "GLCM_Information1_135",
                  "GLCM_MaximalCorrelationCoefficient_135"]

    labels_four_dir = []
    labels_four_dir.extend(labels_0)
    labels_four_dir.extend(labels_45)
    labels_four_dir.extend(labels_90)
    labels_four_dir.extend(labels_135)

    # 2) Parameters
    f = f.astype(np.uint8)

    # 3) Calculate Features: Mean and Range
    features = haralick(f,
                        ignore_zeros=ignore_zeros,
                        compute_14th_feature=True,
                        return_mean_ptp=False)

    features_four_dir = np.hstack(np.hstack(features))
    features_four_dir = [i for i in features_four_dir if i != 0]

    return features_four_dir, labels_four_dir
