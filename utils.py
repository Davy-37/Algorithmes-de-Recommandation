
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def hoeffding_margin(n_k, delta=0.05):
    """
    Calcule la marge d'erreur pour un échantillon de taille n_k en utilisant l'inégalité de Hoeffding.
       n_k: Le nombre d'essais pour un traitement.
       delta: Le niveau de confiance (par défaut 0.05 pour un intervalle de confiance de 95%).
    :return: La marge d'erreur calculée.
    """
    if n_k == 0:
        return float('inf')  
    return np.sqrt(1 / (2 * n_k) * np.log(2 / delta))

def plot_estimations(s, n, strategy_name):
    """
    Affiche un graphique montrant les estimations des probabilités d'efficacité après les essais.
       s: Le nombre de succès pour chaque traitement.
       n: Le nombre d'essais pour chaque traitement.
       strategy_name: Le nom de la stratégie utilisée pour le graphique.
    """
    plt.figure(figsize=(10,6))
    plt.bar(range(1, len(s)+1), s / n, tick_label=[f"Traitement {i+1}" for i in range(len(s))])
    plt.xlabel("Traitement")
    plt.ylabel("Probabilité estimée d'efficacité")
    plt.title(f"Estimations des probabilités d'efficacité - {strategy_name}")
    plt.savefig(f"estimations_{strategy_name}.png")
    plt.close()


def plot_gaussians(true_p, s, n, strategy_name):
    """
    Affiche la distribution normale associée à l'estimation de la probabilité d'efficacité.
       true_p: Les probabilités réelles des traitements.
       s: Le nombre de succès pour chaque traitement.
       n: Le nombre d'essais pour chaque traitement.
       strategy_name: Le nom de la stratégie utilisée pour le graphique.
    """
    x = np.linspace(0, 1, 1000) 
    plt.figure(figsize=(10,6))
    for k in range(len(s)):
        if n[k] == 0:
            continue  
        mean = true_p[k]
        variance = (mean * (1 - mean)) / n[k]
        y = norm.pdf(x, loc=mean, scale=np.sqrt(variance))
        plt.plot(x, y, label=f"Traitement {k+1} (p={mean:.2f})")
        plt.axvline(s[k] / n[k], color='r', linestyle='--')  # Estimation empirique
        
    plt.xlabel('Probabilité d\'efficacité estimée')
    plt.ylabel('Densité')
    plt.title(f"Distribution normale estimée - {strategy_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"distribution_{strategy_name}.png")
    plt.close()
