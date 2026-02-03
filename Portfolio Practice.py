"""
Optimisation de Portfolio - Méthode de Markowitz
Création étape par étape
"""

# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Config de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports réussis")
print("Bibliothèques disponibles:")
print("  - NumPy:", np.__version__)
print("  - Pandas:", pd.__version__)
print("  - Matplotlib")
print("  - SciPy (optimize)")

# ÉTAPE 1: IMPORTATION DES DONNÉES DEPUIS YAHOO FINANCE


def import_data(tickers, date_debut, date_fin):
     
    print(f"\n{'='*60}")
    print(f"Téléchargement des données pour: {', '.join(tickers)}")
    print(f"Période: {date_debut} à {date_fin}")
    print(f"{'='*60}")

# Téléchargement des données
    data = yf.download(tickers, start=date_debut, end=date_fin, progress=False)

# Récupération des prix de clôture ajustés
    if len(tickers) == 1:
        prix = data['Adj Close'].to_frame()
        prix.columns = tickers
    else:
        prix = data['Adj Close']
    
    # Affichage des informations
    print(f"\n✓ Données téléchargées avec succès!")
    print(f"  - Nombre de jours: {len(prix)}")
    print(f"  - Période réelle: {prix.index[0].date()} à {prix.index[-1].date()}")
    print(f"\nAperçu des 5 premiers jours:")
    print(prix.head())
    print(f"\nAperçu des 5 derniers jours:")
    print(prix.tail())
    
    return prix


# ÉTAPE 2: CALCUL DES RENDEMENTS


def calc_return(prix):
    """
    Calcule les rendements logarithmiques quotidiens
    
    Formule: r(t) = ln(P(t) / P(t-1))
    """
    rendements = np.log(prix / prix.shift(1))
    rendements = rendements.dropna()  # Supprimer la première ligne (NaN)
    
    print(f"\n{'='*60}")
    print("RENDEMENTS CALCULÉS")
    print(f"{'='*60}")
    print(f"Nombre d'observations: {len(rendements)}")
    print(f"\nAperçu des rendements:")
    print(rendements.head())
    
    return rendements


# ÉTAPE 3: CALCUL DES STATISTIQUES

def calc_stats(rendements, jours_annee=252):
   
    stats = {}
    
    # 1. Rendements moyens annualisés
    stats['rendements_moyens'] = rendements.mean() * jours_annee
    
    # 2. Variance annualisée
    stats['variance'] = rendements.var() * jours_annee
    
    # 3. Écart-type (volatilité) annualisé
    stats['volatilite'] = rendements.std() * np.sqrt(jours_annee)
    
    # 4. Matrice de covariance annualisée
    stats['matrice_covariance'] = rendements.cov() * jours_annee
    
    # 5. Matrice de corrélation
    stats['matrice_correlation'] = rendements.corr()
    
    return stats


def aff_stats(stats):
  
    print(f"\n{'='*70}")
    print("📈 STATISTIQUES DES ACTIFS")
    print(f"{'='*70}")
    
    # Tableau récapitulatif
    resume = pd.DataFrame({
        'Rendement Annuel (%)': stats['rendements_moyens'] * 100,
        'Volatilité (%)': stats['volatilite'] * 100,
        'Variance': stats['variance']
    })
    print("\n", resume.round(4))
    
    # Matrice de corrélation
    print(f"\n{'-'*70}")
    print("🔗 MATRICE DE CORRÉLATION")
    print(f"{'-'*70}")
    print()
    print(stats['matrice_correlation'].round(4))
    
    # Matrice de covariance
    print(f"\n{'-'*70}")
    print("📊 MATRICE DE COVARIANCE")
    print(f"{'-'*70}")
    print()
    print(stats['matrice_covariance'].round(6))

# ÉTAPE 4: OPTIMISATION DE MARKOWITZ

def portfolio_performance(poids, rendements_moyens, matrice_cov):
    """
    Calcule la performance d'un portefeuille
    
    Paramètres:
    -----------
    poids : array
        Poids de chaque actif (doivent sommer à 1)
    rendements_moyens : Series
        Rendements moyens annualisés
    matrice_cov : DataFrame
        Matrice de covariance
    
    Retour:
    -------
    tuple : (rendement du portefeuille, volatilité du portefeuille)
    """
    # Rendement du portefeuille = somme pondérée des rendements
    rendement_portfolio = np.sum(poids * rendements_moyens)
    
    # Volatilité du portefeuille = sqrt(poids^T * Covariance * poids)
    volatilite_portfolio = np.sqrt(np.dot(poids.T, np.dot(matrice_cov, poids)))
    
    return rendement_portfolio, volatilite_portfolio

def ratio_sharpe_negatif(poids, rendements_moyens, matrice_cov, taux_sans_risque=0.02):
    """
    Calcule le ratio de Sharpe NÉGATIF (pour minimisation)
    
    Le ratio de Sharpe mesure le rendement excédentaire par unité de risque
    Sharpe = (Rendement - Taux sans risque) / Volatilité
    
    On retourne la version négative car scipy.optimize MINIMISE
    et on veut MAXIMISER le Sharpe
    
    Paramètres:
    -----------
    poids : array
        Poids des actifs
    rendements_moyens : Series
        Rendements moyens annualisés
    matrice_cov : DataFrame
        Matrice de covariance
    taux_sans_risque : float
        Taux sans risque annuel (2% par défaut)
    
    Retour:
    -------
    float : -Sharpe ratio (négatif pour minimisation)
    """
    rdt, vol = performance_portefeuille(poids, rendements_moyens, matrice_cov)
    return -(rdt - taux_sans_risque) / vol

def optimiser_portefeuille(rendements_moyens, matrice_cov):
    """
    Trouve les portefeuilles optimaux selon deux critères:
    1. Maximiser le ratio de Sharpe
    2. Minimiser la volatilité
    
    Paramètres:
    -----------
    rendements_moyens : Series
        Rendements moyens annualisés
    matrice_cov : DataFrame
        Matrice de covariance
    
    Retour:
    -------
    dict : Résultats d'optimisation pour les deux stratégies
    """
    n_actifs = len(rendements_moyens)
    
    # Contrainte : la somme des poids doit être égale à 1 (100%)
    contraintes = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bornes : pas de vente à découvert (poids entre 0 et 1)
    bornes = tuple((0, 1) for _ in range(n_actifs))
    
    # Point de départ : équipondération (tous les actifs ont le même poids)
    poids_initial = np.array([1/n_actifs] * n_actifs)
    
    print(f"\n{'='*70}")
    print("⚙️  OPTIMISATION EN COURS...")
    print(f"{'='*70}")
    
    # OPTIMISATION 1: Maximiser le ratio de Sharpe
    print("\n→ Recherche du portefeuille à Sharpe maximum...")
    resultat_sharpe = minimize(
        ratio_sharpe_negatif,           # Fonction à minimiser
        poids_initial,                   # Point de départ
        args=(rendements_moyens, matrice_cov),  # Arguments supplémentaires
        method='SLSQP',                  # Méthode d'optimisation
        bounds=bornes,                   # Contraintes sur les poids
        constraints=contraintes          # Somme = 1
    )
    
    # OPTIMISATION 2: Minimiser la volatilité
    print("→ Recherche du portefeuille à volatilité minimum...")
    resultat_min_vol = minimize(
        lambda poids: performance_portefeuille(poids, rendements_moyens, matrice_cov)[1],
        poids_initial,
        method='SLSQP',
        bounds=bornes,
        constraints=contraintes
    )
    
    print("✓ Optimisation terminée!")
    
    return {
        'max_sharpe': resultat_sharpe,
        'min_volatilite': resultat_min_vol
    }


def afficher_portefeuilles_optimaux(resultats_optim, stats, tickers):
    """
    Affiche les résultats des portefeuilles optimaux
    
    Paramètres:
    -----------
    resultats_optim : dict
        Résultats de l'optimisation
    stats : dict
        Statistiques des actifs
    tickers : list
        Liste des symboles boursiers
    """
    print(f"\n{'='*70}")
    print("🎯 PORTEFEUILLES OPTIMAUX")
    print(f"{'='*70}")
    
    # PORTEFEUILLE 1: Max Sharpe
    max_sharpe = resultats_optim['max_sharpe']
    rdt_sharpe, vol_sharpe = performance_portefeuille(
        max_sharpe.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    sharpe_ratio = (rdt_sharpe - 0.02) / vol_sharpe
    
    print("\n🏆 PORTEFEUILLE À SHARPE MAXIMUM")
    print(f"{'-'*70}")
    print(f"  Rendement annuel espéré : {rdt_sharpe*100:>6.2f}%")
    print(f"  Volatilité (risque)      : {vol_sharpe*100:>6.2f}%")
    print(f"  Ratio de Sharpe          : {sharpe_ratio:>6.4f}")
    print(f"\n  💡 Ce portefeuille offre le meilleur compromis rendement/risque")
    print(f"\n  Allocation des actifs:")
    for i, ticker in enumerate(tickers):
        poids = max_sharpe.x[i] * 100
        if poids > 0.5:  # Afficher seulement si > 0.5%
            print(f"    {ticker:>6} : {poids:>6.2f}%")
    
    # PORTEFEUILLE 2: Min Volatilité
    min_vol = resultats_optim['min_volatilite']
    rdt_min, vol_min = performance_portefeuille(
        min_vol.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    sharpe_min = (rdt_min - 0.02) / vol_min
    
    print(f"\n🛡️  PORTEFEUILLE À VOLATILITÉ MINIMUM")
    print(f"{'-'*70}")
    print(f"  Rendement annuel espéré : {rdt_min*100:>6.2f}%")
    print(f"  Volatilité (risque)      : {vol_min*100:>6.2f}%")
    print(f"  Ratio de Sharpe          : {sharpe_min:>6.4f}")
    print(f"\n  💡 Ce portefeuille minimise le risque (idéal pour profil conservateur)")
    print(f"\n  Allocation des actifs:")
    for i, ticker in enumerate(tickers):
        poids = min_vol.x[i] * 100
        if poids > 0.5:  # Afficher seulement si > 0.5%
            print(f"    {ticker:>6} : {poids:>6.2f}%")

def afficher_portefeuilles_optimaux(resultats_optim, stats, tickers):
    """
    Affiche les résultats des portefeuilles optimaux
    
    Paramètres:
    -----------
    resultats_optim : dict
        Résultats de l'optimisation
    stats : dict
        Statistiques des actifs
    tickers : list
        Liste des symboles boursiers
    """
    print(f"\n{'='*70}")
    print("🎯 PORTEFEUILLES OPTIMAUX")
    print(f"{'='*70}")
    
    # PORTEFEUILLE 1: Max Sharpe
    max_sharpe = resultats_optim['max_sharpe']
    rdt_sharpe, vol_sharpe = performance_portefeuille(
        max_sharpe.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    sharpe_ratio = (rdt_sharpe - 0.02) / vol_sharpe
    
    print("\n🏆 PORTEFEUILLE À SHARPE MAXIMUM")
    print(f"{'-'*70}")
    print(f"  Rendement annuel espéré : {rdt_sharpe*100:>6.2f}%")
    print(f"  Volatilité (risque)      : {vol_sharpe*100:>6.2f}%")
    print(f"  Ratio de Sharpe          : {sharpe_ratio:>6.4f}")
    print(f"\n  💡 Ce portefeuille offre le meilleur compromis rendement/risque")
    print(f"\n  Allocation des actifs:")
    for i, ticker in enumerate(tickers):
        poids = max_sharpe.x[i] * 100
        if poids > 0.5:  # Afficher seulement si > 0.5%
            print(f"    {ticker:>6} : {poids:>6.2f}%")
    
    # PORTEFEUILLE 2: Min Volatilité
    min_vol = resultats_optim['min_volatilite']
    rdt_min, vol_min = performance_portefeuille(
        min_vol.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    sharpe_min = (rdt_min - 0.02) / vol_min
    
    print(f"\n🛡️  PORTEFEUILLE À VOLATILITÉ MINIMUM")
    print(f"{'-'*70}")
    print(f"  Rendement annuel espéré : {rdt_min*100:>6.2f}%")
    print(f"  Volatilité (risque)      : {vol_min*100:>6.2f}%")
    print(f"  Ratio de Sharpe          : {sharpe_min:>6.4f}")
    print(f"\n  💡 Ce portefeuille minimise le risque (idéal pour profil conservateur)")
    print(f"\n  Allocation des actifs:")
    for i, ticker in enumerate(tickers):
        poids = min_vol.x[i] * 100
        if poids > 0.5:  # Afficher seulement si > 0.5%
            print(f"    {ticker:>6} : {poids:>6.2f}%")

     
