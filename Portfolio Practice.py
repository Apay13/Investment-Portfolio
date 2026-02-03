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

