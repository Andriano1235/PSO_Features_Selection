import pandas as pd


class Dataframes():
    # structure of the solution
    def __init__(self):
        self.fos = None
        self.GLCM_mean = None
        self.GLCM_four_dir = None
        self.GLDS = None
        self.GLRLM = None
        self.Class = None


def convert2dataframe(features, labels=None):
    if labels != None:
        df_class = pd.DataFrame(labels, columns=["Class"])
        dataframe.Class = df_class

    df_fos = pd.DataFrame(features.FOS, columns=features.Labels_FOS)
    df_GLCM_mean = pd.DataFrame(
        features.GLCM_mean, columns=features.Labels_GLCM_mean)
    df_GLCM_four_dir = pd.DataFrame(
        features.GLCM_four_dir, columns=features.Labels_GLCM_four_dir)
    df_GLDS = pd.DataFrame(features.GLDS, columns=features.Labels_GLDS)
    df_GLRLM = pd.DataFrame(features.GLRLM, columns=features.Labels_GLRLM)

    dataframe = Dataframes()
    dataframe.fos = df_fos
    dataframe.GLCM_mean = df_GLCM_mean
    dataframe.GLCM_four_dir = df_GLCM_four_dir
    dataframe.GLDS = df_GLDS
    dataframe.GLRLM = df_GLRLM

    return dataframe
