import numpy as np
import csv
from simulator import simulate_results
from strategies import strategy_uniform, strategy_mle, strategy_hoeffding, strategy_bayesienne
from utils import hoeffding_margin, plot_estimations, plot_gaussians

"""
   n_total: Le nombre total d'essais.
   n_treatments: Le nombre de traitements.
   true_p: Les vraies probabilités de succès pour chaque traitement.
   epsilon: Le paramètre d'exploration-exploitation.
   strategies: Un dictionnaire contenant les différentes stratégies d'échantillonnage.

"""


n_total = 1000  
n_treatments = 5  
true_p = np.random.rand(n_treatments)  
epsilon = 0.1 


strategies = {
    "uniforme": strategy_uniform,
    "MLE": strategy_mle,
    "hoeffding": strategy_hoeffding,
    "bayesienne": strategy_bayesienne  
}


def save_results_to_csv(s, n, strategy_name):
    """
    Sauvegarde les résultats (succès, essais, probabilité estimée, marge d'erreur) dans un fichier CSV.
       s: Le nombre de succès pour chaque traitement
       n: Le nombre d'essais pour chaque traitement
       strategy_name: Le nom de la stratégie utilisée
    """
    with open(f"results_{strategy_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Traitement', 'Nombre d\'essais', 'Nombre de succès', 'Probabilité estimée', 'Marge d\'erreur'])
        for i in range(n_treatments):
            est = s[i] / n[i] if n[i] > 0 else 0
            margin = hoeffding_margin(n[i]) if n[i] > 0 else float('inf')
            writer.writerow([i+1, int(n[i]), int(s[i]), est, margin])


for name, strategy in strategies.items():
    s = np.zeros(n_treatments) 
    n = np.zeros(n_treatments)  
    
    
    for k in range(n_treatments):
        result = simulate_results(1, true_p[k])
        s[k] += result[0]
        n[k] += 1
    

    for i in range(n_total - n_treatments):
        selected = strategy(n_treatments, s, n)
        result = simulate_results(1, true_p[selected])
        s[selected] += result[0]
        n[selected] += 1

    save_results_to_csv(s, n, name)

   
    plot_estimations(s, n, name)
    plot_gaussians(true_p, s, n, name)
    
    print(f"Stratégie {name} terminée. Fichiers enregistrés.")

    print(f"Nombre d'essais : {n}")
    print(f"Nombre de succès : {s}")
    print(f"Vrais probabilités : {true_p}")
    print(f"Estimations : {[s[k] / n[k] if n[k] > 0 else 0 for k in range(n_treatments)]}") 