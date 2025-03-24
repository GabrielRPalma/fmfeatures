from mclustpy import mclustpy


def get_clusters(data, G, model_name):
    """Obtain Gaussian Mixture Model (GMM) clusters

    This function uses the python implementation
    of the mclust package, originally writen in
    R programming language. The function returns
    the clusters based on multiple financial time
    series, including explanatory and target
    series

    Parameters
    ----------
    data : numpy.ndarray
        Numpy array containing the explanatory series
        for a given financial time series. `data` must
        not contain the target series.
        
    G : int, optional
     The number of clusters used by the GMM.
        
    model_name : str
              The name of the variance covariance structure
              used by the GMM.
        

    Returns
    -------
    dict
        A dictionary containing the output values 
        returned by the mclust function, with keys 
        specified by OUT_NAMES.
        
    """
    res = mclustpy(data, G, modelNames=model_name, random_seed=2020)
    return res
