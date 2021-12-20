import numpy as np
import scipy as sp
import scipy.linalg




def prony(x: np.array, T: float):
    if len(x) % 2 == 1:
        x = x[:len(x)-1]

    p = len(x) // 2

    shift_x = [0] + list(x)
    a = scipy.linalg.solve([shift_x[p+i:i:-1] for i in range(p)], -x[p::])

    z = np.roots([*a[::-1], 1])

    h = scipy.linalg.solve([z**n for n in range(1, p + 1)], x[:p])

    f = 1 / (2 * np.pi * T) * np.arctan(np.imag(z) / np.real(z))
    alfa = 1 / T * np.log(np.abs(z))
    A = np.abs(h)
    fi = np.arctan(np.imag(h) / np.real(h))

    return f, alfa, A, fi    

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    N = 512
    # Time vector
    start, stop = 0, 4 * np.pi

    t = np.linspace(start, stop, N)

    # Amplitudes and freqs
    f1, f2, f3 = 2, 7, 12
    A1, A2, A3 = 5, 1, 3

    # Signal
    x = A1 * np.cos(2*np.pi*f1*t) + A2 * np.cos(2*np.pi*f2*t) + A3 * np.cos(2*np.pi*f3*t)

    #h = 0.02
    #x = np.array([sum([(k * np.exp(-h * i / k) * np.cos(2 * np.pi * k * h * i + np.pi / 4.)) for k in range(1, 4)]) for i in range(1, 201)])
    
    f, alfa, A, fi = prony(x, (stop - start) / N)

    # i = np.arange(1, 201)
    # h = 0.02

    # fun = lambda i: sum([(k * np.exp(-h * i / k) * np.cos(4 * np.pi * k * h * i + np.pi / k)) for k in range(1, 4)])

    # x = fun(i)

    # f, alfa, A, fi = prony(x, 1)

    plt.stem(2*A)
    plt.plot()
    plt.grid()
    plt.show()