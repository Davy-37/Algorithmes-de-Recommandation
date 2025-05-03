
import numpy as np


def simulate_results(n, p):
    """
    Simule n essais pour un traitement avec probabilité de succès p.
    :param n: Le nombre d'essais.
    :param p: La probabilité de succès pour le traitement.
    :return: Un tableau de 0 et 1 représentant les succès et les échecs.
    """
    return np.random.binomial(1, p, n)  
