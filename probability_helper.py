import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def plot_normalized_histogram(data):
    # Grafica y devuleve histograma
    frecuencia = Counter(data)
    espacio_muestral = frecuencia.keys()
    probabilidades_ = np.array(list(frecuencia.values()))/len(data)
    plt.bar(espacio_muestral, probabilidades_)
    plt.show()
    return list(espacio_muestral), probabilidades_

def plot_normalized_histogram_sorted(data):
    # Grafica y devuleve histograma
    plt.figure(figsize=(20,6))
    frecuencia = Counter(data)
    espacio_muestral = list(frecuencia.keys())
    probabilidades_ = np.array(list(frecuencia.values()))/len(data)
    plt.bar(espacio_muestral, probabilidades_)
    plt.show()
    indexes = np.argsort(espacio_muestral)
    return np.array(espacio_muestral)[indexes], probabilidades_[indexes]