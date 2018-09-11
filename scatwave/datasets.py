import os
import subprocess
from .caching import get_cache_dir


def find_datasets_base_dir(datasets_base_dir=None):
    """
    Finds the base cache directory for caching operations

    Arguments
    ---------
    datasets_base_dir: string, optional
        Defaults to None. If None, then the datasets directory is searched in the
        environement variable 'SCATTERING_DATASETS'. If the latter does not
        exist the default base cache directory is: "~/scattering_datasets"

    Returns
    -------
    datasets_base_dir: string
        The path to the datasets base directory.


    Notes
    -----
    Set the environment variable SCATTERING_DATASETS to direct dataset
    downloads to a desired download location.
    """

    
    if datasets_base_dir is None:
        datasets_base_dir = os.environ.get('SCATTERING_DATASETS',
                                os.path.expanduser("~/scattering_datasets"))
    return datasets_base_dir


def get_dataset_dir(dataset_name, datasets_base_dir=None, create=True):
    """
    Get the path to a dataset directory of given name, possibly create it if
    it doesn't exist.

    Arguments
    ---------
    dataset_name: string
        Name of the dataset. For instance, "mnist" or "fsdd".
    datasets_base_dir: string, optional
        Name of the base directory. Passed to find_cache_base_dir.
        Defaults to None, resulting in choice of default dataset directory.
    create: boolean, optional
        Provides the authorization to create non-existing directories

    Returns
    -------
    path: string
        The path to the dataset directory
    """

    base_dir = find_datasets_base_dir(datasets_base_dir)
    full_path = os.path.join(base_dir, dataset_name)
    if os.path.exists(full_path):
        return full_path
    elif create:
        os.makedirs(full_path)
        return full_path
    else:
        raise ValueError("Could not find dataset dir {}".format(full_path))


fsdd_url= "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
def fetch_fsdd(verbose=False):
    """
    Fetches the Free Spoken Digit Dataset (FSDD).
    If the dataset is not present in the caching directory named "fsdd", it
    is downloaded via git.

    Arguments
    ---------
    base_dir: string, optional
        Name of the base directory for the caching. Will be rooted at the root
        dataset directory given by functions in caching.py. Defaults to 'fsdd'
    url: string, optional
        url for the github repository containing the dataset.
        Defaults to
        "https://github.com/Jakobovski/free-spoken-digit-dataset.git"
    verbose: boolean, optional
        Whether to display indications of the operations undertaken.
        Defaults to False

    Returns
    -------
    dictionary: dictionary
        A dictionary containing the keys 'path_dataset', with value the
        absolute path to the recordings
        (should be base_dir/free-spoken-digit-dataset/recordings),
        and 'files', with value the list of the files in path_dataset
        ending with .wav
    """
    path = get_dataset_dir("fsdd")
    # check if there is already the free sound dataset within this directory
    name_git = 'free-spoken-digit-dataset'
    downloaded = name_git in os.listdir(path)
    # download the git if not existing:
    if not(downloaded):
        if verbose:
            print('Cloning git repository at ', fsdd_url)
        instruction = "git clone " + fsdd_url + ' ' + str(
            os.path.join(path, name_git))
        status, msg = subprocess.getstatusoutput(instruction)
        if status != 0:
            raise RuntimeError(msg)
    # now that it is downloaded, look at the recordings
    repo = os.path.join(path, name_git, 'recordings')
    files = [f for f in os.listdir(repo) if f.endswith('.wav')]
    dictionary = {'path_dataset': repo, 'files': files}
    return dictionary


