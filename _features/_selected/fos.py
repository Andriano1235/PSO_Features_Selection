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
    labels = ["FOS_StandardDeviation",
              "FOS_Skewness",
              "FOS_MinimalGrayLevel",
              "FOS_10Percentile",
              "FOS_Smoothness"]

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
    features = np.zeros(5, np.double)
    other_features = np.zeros(2, np.double)
    i = np.arange(0, bins)

    other_features[0] = np.dot(i, H)  # mean

    # features[1] = np.argmax(H)  # mode
    # features[2] = np.percentile(roi, 50)  # median

    other_features[1] = sum(np.multiply(
        ((i-other_features[0])**2), H))  # variance

    features[0] = np.sqrt(other_features[1])  # std dev
    features[1] = sum(np.multiply(((i-other_features[0])**3), H)) / \
        (other_features[1] ** 3)  # skewness
    # features[6] = sum(np.multiply(((i-other_features[0])**4), H)) / \
    #     (features[1] ** 4)  # kurtosis
    # features[7] = sum(np.multiply(H, H))  # energy
    # features[3] = -sum(np.multiply(H, np.log(H+1e-16)))  # entropy
    features[2] = min(roi)  # min
    # features[5] = max(roi)  # max
    # features[11] = features[4] / other_features[0]  # coef var
    features[3] = np.percentile(roi, 10)
    # features[6] = np.percentile(roi, 25)
    # features[14] = np.percentile(roi, 75)
    # features[7] = np.percentile(roi, 90)
    # features[16] = features[15] - features[11]  # hist width
    features[4] = 1-(1/(1+other_features[1]))  # smoothness

    return features, labels
