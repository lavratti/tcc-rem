import csv
import sys
import numpy as np
from tqdm import tqdm
import pickle


def load(s=None, offset=0):

    song_ids = []
    mean_arousals = []
    mean_valences = []
    melspectograms_list = []

    print("Loading labels")
    with open('dataset/annotations/static_annotations.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for _ in range(1 + offset):
            next(reader)
        for row in reader:
            song_ids.append(int(row[0]))
            mean_arousals.append(float(row[1]))
            mean_valences.append(float(row[3]))
    print("Done loading labels")

    if s is None:
        s = len(song_ids)

    print("Loading dataset mel-spectograms.")
    for n in tqdm(song_ids[:s], unit='files', file=sys.stdout):
        with open('dataset/melgrams/melgram_power_to_db/{}.csv'.format(n), 'rb') as f:
            a = np.loadtxt(f, delimiter=',')
            a = np.resize(a, (128, 64))
            melspectograms_list.append(a)
        if len(melspectograms_list) == s:
            break

    mean_arousals = np.array(mean_arousals[:len(melspectograms_list)])
    mean_valences = np.array(mean_valences[:len(melspectograms_list)])
    melspectograms = np.array(melspectograms_list)
    minmel = melspectograms.min(axis=(1, 2), keepdims=True)
    maxmel = melspectograms.max(axis=(1, 2), keepdims=True)
    melspectograms = (melspectograms - minmel) / (maxmel - minmel)
    print(np.shape(melspectograms))

    return mean_arousals, mean_valences, melspectograms

def quick_load():
    mean_arousals = pickle.load(open('dataset/mean_arousals.pickle', 'rb'))
    mean_valences = pickle.load(open('dataset/mean_valences.pickle', 'rb'))
    melspectograms = pickle.load(open('dataset/melspectograms.pickle', 'rb'))
    return mean_arousals, mean_valences, melspectograms

def gen_pickle():
    pickle_mean_arousals, pickle_mean_valences, pickle_melspectograms = load()
    pickle.dump(pickle_mean_arousals, open('dataset/mean_arousals.pickle', 'wb+'))
    pickle.dump(pickle_mean_valences, open('dataset/mean_valences.pickle', 'wb+'))
    pickle.dump(pickle_melspectograms, open('dataset/melspectograms.pickle', 'wb+'))
