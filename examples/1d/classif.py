"""
Self-contained classification example on the free-spoken digit data recordings
This dataset is automatically downloaded and preprocessed from
https://github.com/Jakobovski/free-spoken-digit-dataset.git

Downloading and precomputing scattering coefficients should take about 5 mns.
Running the gradient descent takes about 1 mn.

Results:
Training accuracy = 99.6%
Testing accuracy = 95.3%
"""
import torch
from torch.nn import Linear, NLLLoss, LogSoftmax, Sequential
from torch.optim import Adam
from torch.autograd import Variable
from scattering import Scattering1D
from scattering.datasets import fetch_fsdd
from scattering.caching import get_cache_dir
import numpy as np
from scipy.io import wavfile
import os
import json
import time


def loadfile(path_file):
    sr, x = wavfile.read(path_file)
    x = np.asarray(x, dtype='float')
    # make it mono
    if x.ndim > 1:
        smallest_axis = np.argmin(x.shape)
        x = x.mean(axis=smallest_axis)
    x = np.asarray(x, dtype='float')
    x /= np.max(np.abs(x))
    return sr, x


def get_fsdd_scattering_coefs(T, J, Q, use_cuda=True, force_compute=False,
                              cache_name='fsdd', is_train=True,
                              log_transform=True, average_time=True,
                              verbose=True):
    """
    Downloads, preprocesses and caches the scattering coefficients of the
    Free Spoken Digit Dataset

    This function is the main entrance function to access the fsdd dataset
    in an automated fashion.

    Arguments
    ---------
    T: int > 0
        Support of the signals to use (otherwise, signals will be cut or
        padded to achieve this value)
    J: int > 0
        Scale parameter for the scattering transform
    Q: int > 0
        Quality factor for the scattering filterbank.
    use_cuda: boolean, optional
        Allows to use cuda to speed up operations. Defaults to True
    force_compute: boolean, optional
        Whether to force to recompute scattering vectors, even if existing.
        Defaults to False.
    cache_name: string, optional
        Name of the caching directory for this file. Defaults to "fsdd"
    is_train: boolean, optional
        Whether to provide files related to the training or testing set.
        Defaults to True
    log_transform: boolean, optional
        Whether to take the logarithm of the scattering coefficients, which
        should be more equally distributed. This removes the zero-order
        coefficients (averages). Defaults to True.
    average_time: boolean, optional
        Whether to take the average of the scatterings along time, to reduce
        the dimensionality of the problem. Defaults to True
    verbose: boolean, optional
        Whether to display information about what is performed.
        Defaults to True.

    Returns
    -------
    x: torch FloatTensor with 2 axis
        The input vectors (lines are samples, columns features)
        The inputs are processed according to the log and averaging operations,
        if necessary.
    y: torch LongTensor with 1 axis
        The output classes (integers from 0 to 9)
    """
    # Get the aggregated file in the adequate folder
    x, y = get_agg_scattering_coefs(
        T, J, Q, use_cuda=use_cuda, force_compute=force_compute,
        cache_name=cache_name, is_train=is_train, verbose=verbose)
    if log_transform:
        x = torch.log(torch.abs(x)[:, 1:])  # remove order 0
    if average_time:
        x = torch.mean(x, dim=-1)
    else:
        x = x.view(x.shape(0), -1).contiguous()
    return x, y


def get_agg_scattering_coefs(T, J, Q, use_cuda=True, force_compute=False,
                             cache_name='fsdd', is_train=True,
                             verbose=False):
    """
    Help function for get_fsdd_dataset.
    Computes and caches the aggregated dataset for train and test (aggregated
    along all files, which is possible since the dataset is very small).

    Arguments
    ---------
    T: int > 0
        Support of the signals to use (otherwise, signals will be cut or
        padded to achieve this value)
    J: int > 0
        Scale parameter for the scattering transform
    Q: int > 0
        Quality factor for the scattering filterbank.
    use_cuda: boolean, optional
        Allows to use cuda to speed up operations. Defaults to True
    force_compute: boolean, optional
        Whether to force to recompute scattering vectors, even if existing.
        Defaults to False.
    cache_name: string, optional
        Name of the caching directory for this file. Defaults to "fsdd"
    is_train: boolean, optional
        Whether to provide files related to the training or testing set.
        Defaults to True
    verbose: boolean, optional
        Whether to display information about what is performed.
        Defaults to True.

    Returns
    -------
    x: torch FloatTensor with 2 axis
        The input vectors (lines are samples, columns features), un-processed.
    y: torch LongTensor with 1 axis
        The output classes (integers from 0 to 9)
    """
    # Check if the aggregated files for train or test already exist
    suffix = 'train' if is_train else 'test'
    path_data = get_cache_dir(os.path.join(cache_name, 'scattering'))
    input_path = os.path.join(path_data, 'input_' + suffix + '.th')
    output_path = os.path.join(path_data, 'output_' + suffix + '.th')
    exists_input = os.path.exists(input_path)
    exists_output = os.path.exists(output_path)
    if not(exists_input) or not(exists_output) or force_compute:
        # get the list of all the scattering coefficients
        # (they may be recomputed along the way if necessary
        files = get_detail_scattering_coefs(
            T, J, Q, use_cuda=use_cuda, force_compute=force_compute,
            cache_name=cache_name, is_train=is_train, verbose=verbose)
        if verbose:
            print('Aggregating individual scattering vectors for', suffix)
        # Aggregate the files
        path_dataset = os.path.join(path_data, suffix)
        s_acc = []
        y_acc = []
        for f in files:
            s = torch.load(os.path.join(path_dataset, f))
            label = int(f.split('_')[0])
            y = torch.LongTensor([label])
            s_acc.append(s.unsqueeze(0))
            y_acc.append(y.unsqueeze(0))
        s_acc = torch.cat(s_acc, dim=0)
        y_acc = torch.squeeze(torch.cat(y_acc, dim=0))
        # save them
        torch.save(s_acc, input_path)
        torch.save(y_acc, output_path)
    else:
        s_acc = torch.load(input_path)
        y_acc = torch.load(output_path)
    # return the result
    return s_acc, y_acc


def get_detail_scattering_coefs(T, J, Q, use_cuda=True,
                                force_compute=False,
                                cache_name='fsdd', is_train=True,
                                verbose=False):
    """
    Get the list of all the class_name_idex.th files for the train/test class
    If the folder cache_name/scattering/train or test does not exist or
    if force_compute is True, all the scattering coefficients are recomputed
    along the way (both for train and test)

    Arguments
    ---------
    T: int > 0
        Support of the signals to use (otherwise, signals will be cut or
        padded to achieve this value)
    J: int > 0
        Scale parameter for the scattering transform
    Q: int > 0
        Quality factor for the scattering filterbank.
    use_cuda: boolean, optional
        Allows to use cuda to speed up operations. Defaults to True
    force_compute: boolean, optional
        Whether to force to recompute scattering vectors, even if existing.
        Defaults to False.
    cache_name: string, optional
        Name of the caching directory for this file. Defaults to "fsdd"
    is_train: boolean, optional
        Whether to provide files related to the training or testing set.
        Defaults to True
    verbose: boolean, optional
        Whether to display information about what is performed.
        Defaults to True.

    Returns
    -------
    list_train (or list_test): list
        The list of the files present in the correct caching directory to
        look after when performing the aggregation. Files are sorted to
        ensure reproducibility.
    """
    suffix = 'train' if is_train else 'test'
    path_dataset = get_cache_dir(
        name=os.path.join(cache_name, 'scattering', suffix))
    list_files = sorted(
        [f for f in os.listdir(path_dataset) if f.endswith('.th')])
    if len(list_files) > 0 and not(force_compute):
        return list_files
    else:
        # we recompute all scattering coefficients
        list_train, list_test = compute_scattering_coefs(
            T, J, Q, use_cuda=use_cuda, cache_name=cache_name, verbose=verbose)
        if is_train:
            return list_train
        else:
            return list_test


def compute_scattering_coefs(T, J, Q, use_cuda=True, cache_name='fsdd',
                             verbose=False):
    """
    Get the list of individual computed scattering coefficients, both for train
    and test.
    If not existing, computes and caches them.

    Arguments
    ---------
    T: int > 0
        Support of the signals to use (otherwise, signals will be cut or
        padded to achieve this value)
    J: int > 0
        Scale parameter for the scattering transform
    Q: int > 0
        Quality factor for the scattering filterbank.
    use_cuda: boolean, optional
        Allows to use cuda to speed up operations. Defaults to True
    cache_name: string, optional
        Name of the caching directory for this file. Defaults to "fsdd"
    verbose: boolean, optional
        Whether to display information about what is performed.
        Defaults to True.

    Returns
    -------
    list_train, list_test:
        The list of the files present in the correct caching directory to
        look after when performing the aggregation. Files are sorted to
        ensure reproducibility.
    """
    # 1) Download the dataset (if not existing)
    info_data = fetch_fsdd(verbose=verbose)
    files = sorted(info_data['files'])
    path_data = info_data['path_dataset']
    # 2) Preprocess with the scattering
    if verbose:
        print("Computing scattering coefficients")
    scattering = Scattering1D(T, J, Q)
    if use_cuda:
        scattering = scattering.cuda()
    # build the caching directory
    path_temp = {
        suffix: get_cache_dir(
            name=os.path.join(cache_name, 'scattering', suffix))
        for suffix in ['train', 'test']}

    n = len(files)
    t0 = time.time()
    for count, f in enumerate(files):
        t1 = time.time()
        print("{}/{} - Time elapsed: {:0.2f}s".format(count, n, t1 - t0),
              end="\r")
        # load the file
        rate, x = loadfile(os.path.join(path_data, f))
        if x.size <= T:
            # pad it with zeros:
            missing = T - x.size
            x_padded = np.zeros(T, dtype='float32')
            left_pad = (T - len(x)) // 2
            x_padded[left_pad:left_pad + len(x)] = x
            x = Variable(torch.from_numpy(x_padded[np.newaxis, np.newaxis]))
            if use_cuda:
                x = x.cuda()
            # compute the scattering
            s = scattering.forward(x).data[0]
            if use_cuda:
                x = x.cpu()
                s = s.cpu()
        else:
            s_acc = []
            # split it
            num_chunks = -(-x.size // T)  # negative integer divison for ceil
            for i in range(num_chunks):
                if i == num_chunks - 1:
                    data = torch.from_numpy(x[-T:]).float()
                else:
                    data = torch.from_numpy(x[i * T: (i + 1) * T]).float()
                data = Variable(data.unsqueeze(0).unsqueeze(0))
                if use_cuda:
                    data = data.cuda()
                s = scattering.forward(data).data
                if use_cuda:
                    data = data.cpu()
                    s = s.cpu()
                s_acc.append(s)
            # average them
            s_acc = torch.cat(s_acc, dim=0)
            s = s_acc.mean(dim=0)

        # store it in the adequate folder
        info = f.split('_')
        name = f.split('.')[0]
        is_train = int(info[2].split('.')[0]) >= 5
        suffix = 'train' if is_train else 'test'
        torch.save(s, os.path.join(path_temp[suffix], name + '.th'))
    # list all the files in train and test
    files_train = sorted(
        [f for f in os.listdir(path_temp['train']) if f.endswith('.th')])
    files_test = sorted(
        [f for f in os.listdir(path_temp['test']) if f.endswith('.th')])
    return files_train, files_test


def compute_loss_and_accuracy(net, criterion, x, y):
    """
    Computes the loss and accuracy of the model net with the given
    criterion for the dataset (x, y)

    Arguments
    ---------
    net: torch Module
    criterion: torch loss function
    x: input Tensor (variable)
    y: expected output tensor (variable)

    Returns
    -------
    avg_loss: float
        average loss across the dataset
    accuracy: float
        average accuracy across the dataset, that is:
        average number of samples for which the network assigns the
        maximal probability for the correct class.
    """
    all_losses = []
    all_successes = []
    for i in range(x.shape[0]):
        resp = net.forward(x[i].unsqueeze(0))
        loss = criterion(resp, y[i:i+1])
        all_losses.append(loss.data.cpu()[0])
        # find the argmax of resp
        sub_resp = torch.squeeze(resp.data.cpu()).numpy()
        success = np.argmax(sub_resp) == y[i].data[0]
        all_successes.append(success)
    avg_loss = np.array(all_losses).mean()
    accuracy = np.array(all_successes).mean()
    return avg_loss, accuracy


if __name__ == '__main__':
    T = 2**13  # support size we use
    J = 8  # averaging scale of the scattering
    Q = 12  # quality factor for the wavelet transform
    log_transform = True  # use the log of the scattering
    average_time = True  # average scattering along time
    batch_size = 32  # batch_size
    num_epochs = 50  # number of epochs
    lr = 1e-4  # learning rate (ADAM)
    random_state = 42  # random seed for reproducibility
    use_cuda = torch.cuda.is_available()  # whether to use cuda
    verbose = True  # print intermediate steps
    cache_name = 'fsdd'  # name of the cache folder
    num_classes = 10  # number of classes for the problem
    force_compute = False  # forces to recompute scattering features
    # set the seed
    torch.manual_seed(random_state)  # set the seed
    # GETTING TRAINING FEATURES
    X_tr, y_tr = get_fsdd_scattering_coefs(
        T, J, Q, use_cuda=use_cuda, cache_name=cache_name, is_train=True,
        log_transform=log_transform, average_time=average_time,
        force_compute=force_compute)
    nsamples = X_tr.shape[0]
    nbatches = nsamples // batch_size
    # whiten the dataset
    mu_tr = X_tr.mean(dim=0)
    std_tr = X_tr.std(dim=0)
    X_tr = (X_tr - mu_tr) / std_tr

    # move to GPU if necessary
    if use_cuda:
        X_tr = X_tr.cuda()
        y_tr = y_tr.cuda()
    # embed them in variables
    X_tr = Variable(X_tr, requires_grad=False)
    y_tr = Variable(y_tr, requires_grad=False)

    # DEFINE THE MODEL
    model = Sequential(Linear(X_tr.shape[-1], num_classes), LogSoftmax())
    optimizer = Adam(model.parameters())
    criterion = NLLLoss()
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # perform the gradient descent
    for e in range(num_epochs):
        # permutation of the dataset
        perm = torch.randperm(nsamples)
        if use_cuda:
            perm = perm.cuda()
        for i in range(nbatches):
            model.zero_grad()
            resp = model.forward(
                X_tr[perm[i * batch_size: (i + 1) * batch_size]])
            loss = criterion(
                resp, y_tr[perm[i * batch_size: (i + 1) * batch_size]])
            loss.backward()
            optimizer.step()
        # compute the loss and the accuracy
        avg_loss, accu = compute_loss_and_accuracy(model, criterion,
                                                   X_tr, y_tr)
        print('Epoch {}, average loss = {:1.3f}, accuracy = {:1.3f}'.format(
            e, avg_loss, accu))

    # Load testing features
    X_te, y_te = get_fsdd_scattering_coefs(
        T, J, Q, use_cuda=use_cuda, cache_name=cache_name, is_train=False,
        log_transform=log_transform, average_time=average_time,
        force_compute=force_compute)
    X_te = (X_te - mu_tr) / std_tr
    if use_cuda:
        X_te = X_te.cuda()
        y_te = y_te.cuda()
    X_te = Variable(X_te, requires_grad=False)
    y_te = Variable(torch.squeeze(y_te), requires_grad=False)

    avg_loss, accu = compute_loss_and_accuracy(model, criterion, X_te, y_te)
    print('TEST, average loss = {:1.3f}, accuracy = {:1.3f}'.format(
          avg_loss, accu))
