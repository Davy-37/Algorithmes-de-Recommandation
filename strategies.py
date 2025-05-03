import numpy as np

def strategy_uniform(n_treatments, s, n):
    """
    Stratégie : choix uniforme du traitement.
    
    À chaque instant, un traitement est sélectionné aléatoirement parmi les traitements disponibles.
    
    :param n_treatments: Le nombre total de traitements
    :param s: Le nombre de succès pour chaque traitement (non utilisé ici)
    :param n: Le nombre d'essais pour chaque traitement (non utilisé ici)
    :return: L'indice du traitement sélectionné
    """
    return np.random.randint(0, n_treatments)

def strategy_mle(n_treatments, s, n):
    """
    Stratégie : choix du traitement ayant la meilleure probabilité d'efficacité observée (MLE).
    
    À chaque instant, on choisit le traitement ayant la meilleure fréquence de succès observée.
    
    :param n_treatments: Le nombre total de traitements
    :param s: Le nombre de succès pour chaque traitement
    :param n: Le nombre d'essais pour chaque traitement
    :return: L'indice du traitement sélectionné
    """
    estimates = np.array([
        s[k] / n[k] if n[k] > 0 else 0
        for k in range(n_treatments)
    ])
    return np.argmax(estimates)

def hoeffding_margin(n, delta=0.05):
    """
    Calcule la marge d'erreur selon l'inégalité de Hoeffding.
    
    Permet d'ajuster l'estimation de la probabilité en fonction du nombre d'observations.
    
    :param n: Le nombre d'essais du traitement
    :param delta: Le paramètre de confiance (par défaut 0.05 pour un niveau de 95%)
    :return: La marge d'erreur calculée
    """
    return np.sqrt(1 / (2 * n) * np.log(2 / delta)) if n > 0 else np.inf

def strategy_hoeffding(n_treatments, s, n, delta=0.05):
    """
    Stratégie : choix du traitement ayant la meilleure probabilité ajustée par une marge d'erreur (Hoeffding).
    
    À chaque instant, on ajuste l'estimation de la probabilité d'efficacité avec une marge de sécurité.
    Cela encourage l'exploration des traitements peu testés.
    
    :param n_treatments: Le nombre total de traitements
    :param s: Le nombre de succès pour chaque traitement
    :param n: Le nombre d'essais pour chaque traitement
    :param delta: Le paramètre de confiance pour le calcul de la marge d'erreur
    :return: L'indice du traitement sélectionné
    """
    scores = np.array([
        (s[k] / n[k] if n[k] > 0 else 0) + hoeffding_margin(n[k], delta)
        for k in range(n_treatments)
    ])
    return np.argmax(scores)

def strategy_bayesienne(n_treatments, s, n, alpha0=1, beta0=1):
    """
    Stratégie avec inférence bayésienne utilisant une loi Beta a priori pour chaque traitement.
    
    À chaque étape n, les probabilités a posteriori sont calculées et un échantillon est tiré
    pour chaque traitement. Le traitement ayant la plus grande valeur tirée est sélectionné (Thompson Sampling).
    
    :param n_treatments: Le nombre total de traitements
    :param s: Le nombre de succès pour chaque traitement
    :param n: Le nombre d'essais pour chaque traitement
    :param alpha0: Le paramètre alpha a priori de la loi Beta
    :param beta0: Le paramètre beta a priori de la loi Beta
    :return: L'indice du traitement sélectionné
    """
    samples = []
    for k in range(n_treatments):
        alpha_k = alpha0 + s[k]
        beta_k = beta0 + (n[k] - s[k])
        sample = np.random.beta(alpha_k, beta_k)
        samples.append(sample)
    return np.argmax(samples)
