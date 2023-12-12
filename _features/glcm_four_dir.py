import numpy as np
import mahotas


def glcm_four_dir_features(f, ignore_zeros=True):

    # 1) Labels
    labels = ["GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
              "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
              "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
              "GLCM_Entropy", "GLCM_DifferenceVariance",
              "GLCM_DifferenceEntropy", "GLCM_Information1",
              "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"]
    angles = ["0", "45", "90", "135"]

    labels_four_dir = [(label+"_"+angle)
                       for angle in angles for label in labels]

    # 2) Parameters
    f = f.astype(np.uint8)

    # 3) Calculate Features: Mean and Range
    features = mahotas.features.haralick(f,
                                         ignore_zeros=ignore_zeros,
                                         compute_14th_feature=True,
                                         return_mean_ptp=False)
    features_four_dir = np.hstack(np.hstack(features))

    return features_four_dir, labels_four_dir
