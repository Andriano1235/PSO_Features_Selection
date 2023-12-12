import numpy as np

# First Order Statistic


def FirstOrderStatistic(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    features : numpy ndarray
        1)Mean, 2)Variance, 3)Median (50-Percentile), 4)Mode, 
        5)Skewness, 6)Kurtosis, 7)Energy, 8)Entropy, 
        9)Minimal Gray Level, 10)Maximal Gray Level, 
        11)Coefficient of Variation, 12,13,14,15)10,25,75,90-
        Percentile, 16)Histogram width
    labels : list
        Labels of features.
    '''
    if mask is None:
        mask = np.ones(f.shape)

    # 1) Labels
    labels = ["FOS_Mean", "FOS_Mode", "FOS_Median", "FOS_Variance", "FOS_StandardDeviation", "FOS_Skewness",
              "FOS_Kurtosis", "FOS_Energy", "FOS_Entropy", "FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel", "FOS_CoefficientOfVariation",
              "FOS_10Percentile", "FOS_25Percentile", "FOS_75Percentile",
              "FOS_90Percentile", "FOS_HistogramWidth", "FOS_Smoothness"]

    # 2) Parameters
    f = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255
    Ng = (level_max - level_min) + 1
    bins = Ng

    # 3) Calculate Histogram H inside ROI
    f_ravel = f.ravel()
    mask_ravel = mask.ravel()
    roi = f_ravel[mask_ravel.astype(bool)]
    H = np.histogram(roi, bins=bins, range=[
                     level_min, level_max], density=True)[0]

    # 4) Calculate Features
    features = np.zeros(18, np.double)
    i = np.arange(0, bins)
    features[0] = np.dot(i, H)  # mean
    features[1] = np.argmax(H)  # mode
    features[2] = np.percentile(roi, 50)  # median
    features[3] = sum(np.multiply(((i-features[0])**2), H))  # variance
    features[4] = np.sqrt(features[3])  # std dev
    features[5] = sum(np.multiply(((i-features[0])**3), H)) / \
        (features[4] ** 3)  # skewness
    features[6] = sum(np.multiply(((i-features[0])**4), H)) / \
        (features[4] ** 4)  # kurtosis
    features[7] = sum(np.multiply(H, H))  # energy
    features[8] = -sum(np.multiply(H, np.log(H+1e-16)))  # entropy
    features[9] = min(roi)  # min
    features[10] = max(roi)  # max
    features[11] = features[4] / features[0]  # coef var
    features[12] = np.percentile(roi, 10)
    features[13] = np.percentile(roi, 25)
    features[14] = np.percentile(roi, 75)
    features[15] = np.percentile(roi, 90)
    features[16] = features[15] - features[11]  # hist width
    features[17] = 1-(1/(1+features[3]))  # smoothness

    return features, labels
