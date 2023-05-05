import numpy as np
import lib


####################################################################################################
# Exercise 1: DFT
def create_einheits_matrix(n: int) -> np.ndarray:
    einheit_m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                einheit_m[i][j] = 1
    return einheit_m


def dft_matrix(n: int) -> np.ndarray:
    """
    Construct DFT matrix of size n.

    Arguments:
    n: size of DFT matrix

    Return:
    F: DFT matrix of size n

    Forbidden:
    - numpy.fft.*
    """
    # TODO: initialize matrix with proper size
    F = np.zeros((n, n), dtype='complex128')

    w_n = np.exp(-2 * np.pi * 1j / n)
    # print(w_n)
    for i in range(n):
        for j in range(n):
            F[i][j] = w_n ** (i * j) / np.sqrt(n)

    return F


def is_unitary(matrix: np.ndarray) -> bool:
    """
    Check if the passed in matrix of size (n times n) is unitary.

    Arguments:
    matrix: the matrix which is checked

    Return:
    unitary: True if the matrix is unitary
    """
    unitary = True
    (n, m) = matrix.shape
    einheit_m = create_einheits_matrix(n)

    # TODO: check that F is unitary, if not return false
    mc = np.zeros((n, n))
    # cm = np.zeros((n, n))
    conju = np.zeros((n, n))
    conju = np.transpose(matrix.conj())

    mc = np.dot(matrix, conju)

    if not np.allclose(mc, einheit_m):
        unitary = False

    return unitary


def create_harmonics(n: int = 128) -> (list, list):
    """
    Create delta impulse signals and perform the fourier transform on each signal.

    Arguments:
    n: the length of each signal

    Return:
    sigs: list of np.ndarrays that store the delta impulse signals
    fsigs: list of np.ndarrays with the fourier transforms of the signals
    """

    # list to store input signals to DFT
    sigs = []
    # Fourier-transformed signals
    fsigs = []

    # TODO: create signals and extract harmonics out of DFT matrix

    for i in range(n):
        k = np.zeros(n)
        k[i] = 1
        sigs.append(k)

    D = dft_matrix(n)

    for j in range(n):
        FT = np.dot(D, sigs.__getitem__(j))
        fsigs.append(FT)

    return sigs, fsigs


####################################################################################################
# Exercise 2: FFT

def shuffle_bit_reversed_order(data: np.ndarray) -> np.ndarray:
    """
    Shuffle elements of data using bit reversal of list index.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    data: shuffled data array
    """
    data.copy()
    n = data.size
    array = np.zeros(n, dtype='complex128')

    k = int(np.log2(n))
    a = 0
    # TODO: implement shuffling by reversing index bits
    for i in range(n):
        element = data[i]
        a = format(i, "b")
        c = a.zfill(k)
        d = c[::-1]
        index = int(d, 2)
        array[index] = element
    data = array
    # print(data)
    return data


def fft(data: np.ndarray) -> np.ndarray:
    """
    Perform real-valued discrete Fourier transform of data using fast Fourier transform.

    Arguments:
    data: data to be transformed (shape=(n,), dtype='float64')

    Return:
    fdata: Fourier transformed data

    Note:
    This is not an optimized implementation but one to demonstrate the essential ideas
    of the fast Fourier transform.

    Forbidden:
    - numpy.fft.*
    """

    fdata = np.asarray(data, dtype='complex128')
    n = fdata.size
    # check if input length is power of two
    if not n > 0 or (n & (n - 1)) != 0:
        raise ValueError

    # TODO: first step of FFT: shuffle data
    fdata = shuffle_bit_reversed_order(fdata)

    # TODO: second step, recursively merge transforms
    l = 0
    t = 1
    if n > 1:
        for i in range(int(np.log2(n))):
            for k in range(2 ** l):
                w_k = np.exp(-1 * 1j * np.pi * k / t)
                p = 2 ** (l + 1)
                b = 2 ** l
                for j in range(k, n, p):
                    o = w_k * fdata[j + 2 ** l]
                    fdata[j + 2 ** l] = fdata[j] - o
                    fdata[j] = fdata[j] + o
            l = l + 1
            t = 2 ** l
            print(fdata)
            # TODO: normalize fft signal
    fdata = fdata / np.sqrt(n)
    # print(fdata)
    return fdata


def generate_tone(f: float = 261.626, num_samples: int = 44100) -> np.ndarray:
    """
    Generate tone of length 1s with frequency f (default mid C: f = 261.626 Hz) and return the signal.

    Arguments:
    f: frequency of the tone

    Return:
    data: the generated signal
    """

    # sampling range
    x_min = 0.0
    x_max = 1.0

    data = np.zeros(num_samples)

    # TODO: Generate sine wave with proper frequency
    x_max = 2 * np.pi
    k = np.linspace(x_min, x_max, num_samples)
    data = np.sin(f * k)
    return data


def low_pass_filter(adata: np.ndarray, bandlimit: int = 1000, sampling_rate: int = 44100) -> np.ndarray:
    """
    Filter high frequencies above bandlimit.

    Arguments:
    adata: data to be filtered
    bandlimit: bandlimit in Hz above which to cut off frequencies
    sampling_rate: sampling rate in samples/second

    Return:
    adata_filtered: filtered data
    """

    # translate bandlimit from Hz to dataindex according to sampling rate and data size
    bandlimit_index = int(bandlimit * adata.size / sampling_rate)

    # TODO: compute Fourier transform of input data

    # TODO: set high frequencies above bandlimit to zero, make sure the almost symmetry of the transform is respected.

    # TODO: compute inverse transform and extract real component
    adata_filtered = np.zeros(adata.shape[0])

    return adata_filtered


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    n = 50
    # k = dft_matrix(n)
    # a, b = create_harmonics(n)
    # print(a, b)
    # print(k[3])
    b = np.array([4, 1, 6, 8, 9, 9, 6, 9])
    a = dft_matrix(4)
    c = np.array([3, 3, 1, 5])
    d = np.array([8, 1, 9, 5])
    e=np.transpose(d)*np.transpose(c)
    print(e)
    # a = np.array([0.5, 0.25, 0, 0.25, 0.5, 0.25, 0, 0.25])
    # c=shuffle_bit_reversed_order(b)
    # print(c)
    # print(b)
    # print(np.imag(k[1]))
    # print(is_unitary(k))
    # k = fft(a)
    # l = generate_tone()
    # print(l)
    # a,b=create_harmonics(n)
    # lib.plot_harmonics(a,b)
    # fdata,a=lib.read_audio_data("data/mid-c_ref.wav")
    # lib.write_audio_data("iris.npz",fdata,a)
