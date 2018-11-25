import numpy as np
import scipy as sp
import timesynth as ts
import matplotlib as mlt
import matplotlib.pyplot as plt

#print(((samples * 10) % 11 + 1))

def generate_sequence():
    time_sampler = ts.TimeSampler(stop_time=20)
    time_sampler = ts.TimeSampler(stop_time=20)
    sinusoid = ts.signals.Sinusoidal(frequency=0.25)
    white_noise = ts.noise.GaussianNoise(std=0.3)

    irregular_time_samples = time_sampler.sample_irregular_time(num_points=2000, keep_percentage=100)
    # print(irregular_time_samples)
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    samples, signals, errors = timeseries.sample(irregular_time_samples)
    samples = (samples * 10)
    samples = samples.astype(int)
    samples = np.absolute(samples)
    samples = samples % 11
    #print(len(samples))
    return samples

def read_sequence(filename):
    samples = np.loadtxt(filename)
    return samples

def get_estimated_sequence_zeros(real_sequence):
    estimated_sequence = np.zeros(len(real_sequence), dtype = int)
    return estimated_sequence

def get_estimated_sequence_random(real_sequence):
    estimated_sequence = np.random.rand(len(real_sequence))
    return estimated_sequence

def get_estimate_performance(real_sequence, estimated_sequence):
    bin_array = np.equal(real_sequence, estimated_sequence)
    print(bin_array)
    count = np.cumsum(bin_array)
    #count = np.count_nonzero(bin_array)
    return count

def plot_performance(accuracy_count, label):
    indices = np.array([i + 1 for i in range(len(accuracy_count))])
    accuracy_count = np.divide(accuracy_count, indices) * 100
    print(accuracy_count)
    plt.xlabel('Number of Samples')
    plt.ylabel('Cumulative Accuracy (%)')
    plt.xlim([0, len(accuracy_count)])
    plt.ylim([0, 100])
    plt.plot(indices, accuracy_count, label=label)
    #plt.show()

def show_plot():
    plt.legend(loc='best')
    plt.show()

def __main__():
    seq = generate_sequence()
    est_seq = get_estimated_sequence_zeros(seq)
    rand_seq = get_estimated_sequence_random(seq)

    count_array = get_estimate_performance(seq, est_seq)
    count_random_array = get_estimate_performance(seq, rand_seq)
    plot_performance(count_array, 'All Zeros')
    plot_performance(count_random_array, 'Random')
    show_plot()

__main__()

