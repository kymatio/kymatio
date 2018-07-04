import os
import subprocess
from .caching import get_cache_dir


def fetch_fsdd(base_dir='fsdd',
               url="https://github.com/Jakobovski/free-spoken-digit-dataset.git",
               verbose=False):
    """

    Fetches the Free Spoken Digit Dataset (FSDD).
    If the dataset is not present in the caching directory named "fsdd", it
    is downloaded via git.

    Arguments
    ---------
    base_dir: string, optional
        Name of the base directory for the caching. Will be rooted at the root
        cache directory given by functions in caching.py. Defaults to 'fsdd'
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
    path = get_cache_dir(name=base_dir)
    # check if there is already the free sound dataset within this directory
    name_git = 'free-spoken-digit-dataset'
    downloaded = name_git in os.listdir(path)
    # download the git if not existing:
    if not(downloaded):
        if verbose:
            print('Cloning git repository at ', url)
        instruction = "git clone " + url + ' ' + str(
            os.path.join(path, name_git))
        status, msg = subprocess.getstatusoutput(instruction)
        if status != 0:
            raise RuntimeError(msg)
    # now that it is downloaded, look at the recordings
    repo = os.path.join(path, name_git, 'recordings')
    files = [f for f in os.listdir(repo) if f.endswith('.wav')]
    dictionary = {'path_dataset': repo, 'files': files}
    return dictionary
