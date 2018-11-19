import os
import subprocess
import numpy as np
from scipy.io import loadmat
from .caching import get_cache_dir
try:
    from urllib.request import urlopen
    from urllib.error import HTTPError, URLError
except:
    from urllib2 import urlopen, HTTPError, URLError

def find_datasets_base_dir(datasets_base_dir=None):
    """
    Finds the base cache directory for caching operations

    Arguments
    ---------
    datasets_base_dir: string, optional
        Defaults to None. If None, then the datasets directory is searched in the
        environement variable 'KYMATIO_DATASETS'. If the latter does not
        exist the default base cache directory is: "~/kymatio_datasets"

    Returns
    -------
    datasets_base_dir: string
        The path to the datasets base directory.


    Notes
    -----
    Set the environment variable KYMATIO_DATASETS to direct dataset
    downloads to a desired download location.
    """


    if datasets_base_dir is None:
        datasets_base_dir = os.environ.get('KYMATIO_DATASETS',
                                os.path.expanduser("~/kymatio_datasets"))
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



def _download(url, filename):

    try:
        f = urlopen(url)
        with open(filename, 'wb') as local_file:
            local_file.write(f.read())
    except URLError as e:
        raise
    except HTTPError as e:
        raise






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


atom_charges=dict(H=1, C=6, O=8, N=7, S=16)

def read_xyz(filename):
    """Reads xyz files that are used for storing molecule configurations.

    Parameters
    ==========

    filename: str
        Filename of the xyz file

    Returns
    =======
    dictionary containing molecule details

    Notes
    =====
    The file format is #atoms\\nenergy\\nrepeat: atom type\\tx\\ty\\tz"""

    energies = []
    charges = []
    positions = []
    n_atoms = []

    with open(filename, "r") as f:
        content = f.read()

    raw_molecule_txts = content.split("\n\n")
    for raw_molecule_txt in raw_molecule_txts:
        s = raw_molecule_txt.split("\n")
        n_atoms.append(int(s[0]))
        energies.append(float(s[1]))
        atom_positions = []
        molecule_charges = []
        charges.append(molecule_charges)
        positions.append(atom_positions)
        for i, row in zip(range(n_atoms[-1]), s[2:]):
            atom_type, *str_position = [x for x in row.split(" ") if x]
            molecule_charges.append(atom_charges[atom_type])
            pos = np.array(list(map(float, str_position)))
            atom_positions.append(pos)

    arr_positions = np.zeros((len(n_atoms), max(n_atoms), 3), dtype='float32')
    for arr_pos, pos, n in zip(arr_positions, positions, n_atoms):
        arr_pos[:n] = np.array(pos)

    arr_charges = np.zeros_like(arr_positions[..., 0], dtype='int')
    for arr_charge, molecule_charges, n in zip(arr_charges, charges, n_atoms):
        arr_charge[:n] = molecule_charges

    return dict(positions=arr_positions,
                energies=np.array(energies, dtype='float32'),
                charges=arr_charges)


def _pca_align_positions(positions, masks, inplace=False):
    """Rotate molecules so that longest axis is x"""
    if not inplace:
        output = np.zeros_like(positions)
    else:
        output = positions

    masks = masks.astype('bool')
    for pos, mask, out in zip(positions, masks, output):
        masked_pos = pos[mask]
        masked_pos -= masked_pos.mean(0)
        cov = masked_pos.T.dot(masked_pos.copy())
        v, V = np.linalg.eigh(cov)
        aligned = masked_pos.dot(V[:, ::-1])  # largest to smallest
        out[mask] = aligned

    if not inplace:
        return output



qm7_url = "https://qmml.org/Datasets/gdb7-12.zip"
def fetch_qm7(align=True, cache=True):
    """Fetches the GDB7-12 dataset"""

    if cache:
        cache_path = get_cache_dir("qm7")
        if align:
            aligned_filename = os.path.join(cache_path, "qm7_aligned.npz")
            if os.path.exists(aligned_filename):
                f = np.load(aligned_filename)
                return dict(**f)

        # load unaligned if existent, align if required
        unaligned_filename = os.path.join(cache_path, "qm7.npz")
        if os.path.exists(unaligned_filename):
            f = np.load(unaligned_filename)
            if align:
                _pca_align_positions(f['positions'], f['charges'], inplace=True)
                np.savez(aligned_filename, **f)
            return dict(**f)

    path = get_dataset_dir("qm7")
    qm7_file = os.path.join(path, "dsgdb7ae.xyz")
    if not os.path.exists(qm7_file):
        qm7_zipfile = os.path.join(path, "gdb7-12.zip")
        if not os.path.exists(qm7_zipfile):
            _download(qm7_url, qm7_zipfile)
            import zipfile
            with zipfile.ZipFile(qm7_zipfile, "r") as zipref:
                zipref.extractall(path)

    qm7 = read_xyz(qm7_file)
    if cache:
        np.savez(unaligned_filename, **qm7)

    if align:
        _pca_align_positions(qm7['positions'], qm7['charges'], inplace=True)
        if cache:
            np.savez(aligned_filename, **qm7)

    return qm7
