import os


def find_cache_base_dir(cache_base_dir=None):
    """
    Finds the base cache directory for caching operations

    Arguments
    ---------
    cache_base_dir: string, optional
        Defaults to None. If None, then the cache directory is searched in the
        environement variable 'KYMATIO_CACHE'. If the latter does not
        exist (so returns None), then the default base cache directory is:
        "~/kymatio_cache"

    Returns
    -------
    cache_base_dir: string
        The path to the cache base directory.
    """
    if cache_base_dir is None:
        kymatio_cache = os.environ.get('KYMATIO_CACHE')
        if kymatio_cache is None:
            return os.path.join(os.path.expanduser("~"), "kymatio_cache")
        else:
            return kymatio_cache
    else:
        return cache_base_dir


def get_cache_dir(name="", cache_base_dir=None, create=True):
    """
    Get the path to a cache directory of given name, possibly created if
    not existing before.

    Arguments
    ---------
    name: string, optional
        Name of the cache directory. For instance, "mnist" or "fsdd".
        Defaults to empty string.
    cache_base_dir: string, optional
        Name of the base directory. Passed to find_cache_base_dir.
        Defaults to None.
    create: boolean, optional
        Provides the authorization to create non-existing directories

    Returns
    -------
    path: string
        The path to the caching directory
    """
    path = os.path.join(
        find_cache_base_dir(cache_base_dir=cache_base_dir), name)
    if os.path.exists(path):
        return path
    else:
        if create:
            os.makedirs(path)
            return path
        else:
            raise ValueError(
                'The cache directory does not exist,' +
                'but I cannot create it: {}'.format(path))
