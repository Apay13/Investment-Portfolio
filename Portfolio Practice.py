"""
Optimisation de Portfolio - Méthode de Markowitz
Création étape par étape
"""

# Imports nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

# Configuration de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports réussis")
print("Bibliothèques disponibles:")
print("  - NumPy:", np.__version__)
print("  - Pandas:", pd.__version__)
print("  - Matplotlib")
print("  - SciPy (optimize)")
print("  - yfinance")

# ============================================================
# ÉTAPE 1: IMPORTATION DES DONNÉES DEPUIS YAHOO FINANCE
# ============================================================

def importer_donnees(tickers, date_debut, date_fin):
    """
    Importe les prix de clôture ajustés depuis Yahoo Finance
    
    Paramètres:
    -----------
    tickers : list
        Liste des symboles boursiers (ex: ['AAPL', 'MSFT', 'GOOGL'])
    date_debut : str
        Date de début au format 'YYYY-MM-DD'
    date_fin : str
        Date de fin au format 'YYYY-MM-DD'
    
    Retour:
    -------
    DataFrame : Prix de clôture ajustés pour chaque ticker
    """
    print(f"\n{'='*60}")
    print(f"Téléchargement des données pour: {', '.join(tickers)}")
    print(f"Période: {date_debut} à {date_fin}")
    print(f"{'='*60}")
    
    # Téléchargement des données
    data = yf.download(tickers, start=date_debut, end=date_fin, progress=False)
    
    # ===== VÉRIFICATION : Données téléchargées =====
    if data.empty:
        raise ValueError(f"\n❌ ERREUR : Aucune donnée téléchargée.\n"
                        f"   Vérifiez votre connexion internet et les symboles boursiers.")
    
    # ===== DEBUG : Afficher la structure =====
    print(f"\n[DEBUG] Type de colonnes: {type(data.columns)}")
    print(f"[DEBUG] Colonnes: {data.columns.tolist() if hasattr(data.columns, 'tolist') else data.columns}")
    
    # ===== EXTRACTION DES PRIX =====
    prix = None
    
    # Méthode 1 : Colonnes multi-index (cas normal avec plusieurs tickers)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prix = data['Adj Close'].copy()
            print("[DEBUG] Méthode 1 : Multi-index avec 'Adj Close'")
        elif 'Close' in data.columns.get_level_values(0):
            prix = data['Close'].copy()
            print("[DEBUG] Méthode 1 : Multi-index avec 'Close'")
    
    # Méthode 2 : Colonnes simples (un seul ticker ou format différent)
    else:
        if 'Adj Close' in data.columns:
            prix = data[['Adj Close']].copy()
            prix.columns = tickers
            print("[DEBUG] Méthode 2 : Colonnes simples avec 'Adj Close'")
        elif 'Close' in data.columns:
            prix = data[['Close']].copy()
            prix.columns = tickers
            print("[DEBUG] Méthode 2 : Colonnes simples avec 'Close'")
    
    # Méthode 3 : Télécharger ticker par ticker en cas d'échec
    if prix is None:
        print("\n[DEBUG] Méthode 3 : Téléchargement ticker par ticker...")
        prix = pd.DataFrame()
        for ticker in tickers:
            try:
                temp = yf.download(ticker, start=date_debut, end=date_fin, progress=False)
                if not temp.empty:
                    if 'Adj Close' in temp.columns:
                        prix[ticker] = temp['Adj Close']
                    elif 'Close' in temp.columns:
                        prix[ticker] = temp['Close']
                    print(f"  ✓ {ticker} téléchargé")
                else:
                    print(f"  ✗ {ticker} échec")
            except:
                print(f"  ✗ {ticker} erreur")
    
    # Vérification finale
    if prix is None or prix.empty:
        raise ValueError(f"\n❌ ERREUR : Impossible d'extraire les prix.\n"
                        f"   Format de données: {type(data.columns)}\n"
                        f"   Colonnes disponibles: {data.columns}\n"
                        f"   Essayez de relancer le script ou changez de tickers.")
    
    # ===== VÉRIFICATION 1 : Données manquantes =====
    colonnes_invalides = prix.columns[prix.isna().all()].tolist()
    if colonnes_invalides:
        print(f"\n⚠️  ATTENTION : Échec du téléchargement pour : {', '.join(colonnes_invalides)}")
        print(f"   Ces tickers seront supprimés de l'analyse.")
        prix = prix.dropna(axis=1, how='all')
    
    # ===== VÉRIFICATION 2 : Au moins 2 actifs nécessaires =====
    if len(prix.columns) < 2:
        raise ValueError(f"\n❌ ERREUR : Il faut au moins 2 actifs valides pour Markowitz.\n"
                        f"   Actifs valides trouvés : {len(prix.columns)}\n"
                        f"   Tickers en échec : {colonnes_invalides}\n"
                        f"   Vérifiez les symboles boursiers et réessayez.")
    
    # ===== VÉRIFICATION 3 : Données suffisantes =====
    prix = prix.dropna()  # Supprimer lignes avec NaN
    
    if len(prix) < 50:
        raise ValueError(f"\n❌ ERREUR : Pas assez de données ({len(prix)} jours).\n"
                        f"   Minimum requis : 50 jours de cotation.\n"
                        f"   Essayez une période plus longue.")
    
    # Affichage des informations
    print(f"\n✓ Données téléchargées avec succès!")
    print(f"  - Actifs valides : {', '.join(prix.columns.tolist())}")
    print(f"  - Nombre de jours: {len(prix)}")
    print(f"  - Période réelle: {prix.index[0].date()} à {prix.index[-1].date()}")
    print(f"\nAperçu des 5 premiers jours:")
    print(prix.head())
    print(f"\nAperçu des 5 derniers jours:")
    print(prix.tail())
    
    return prix


# ============================================================
# ÉTAPE 2: CALCUL DES RENDEMENTS
# ============================================================

def calculer_rendements(prix):
    """
    Calcule les rendements logarithmiques quotidiens
    
    Formule: r(t) = ln(P(t) / P(t-1))
    
    Paramètres:
    -----------
    prix : DataFrame
        Prix de clôture ajustés
    
    Retour:
    -------
    DataFrame : Rendements quotidiens
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


# ============================================================
# ÉTAPE 3: CALCUL DES STATISTIQUES
# ============================================================

def calculer_statistiques(rendements, jours_annee=252):
    """
    Calcule toutes les statistiques nécessaires pour Markowitz
    
    Paramètres:
    -----------
    rendements : DataFrame
        Rendements quotidiens
    jours_annee : int
        Nombre de jours de trading par an (252 par défaut)
    
    Retour:
    -------
    dict : Dictionnaire contenant toutes les statistiques
    """
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


def afficher_statistiques(stats):
    """
    Affiche les statistiques de manière formatée
    
    Paramètres:
    -----------
    stats : dict
        Dictionnaire des statistiques
    """
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
    print("\nLa corrélation mesure la relation entre les actifs (entre -1 et 1)")
    print("  • 1 = parfaitement corrélés (bougent ensemble)")
    print("  • 0 = pas de corrélation")
    print("  • -1 = corrélés négativement (bougent en sens inverse)")
    print()
    print(stats['matrice_correlation'].round(4))
    
    # Matrice de covariance
    print(f"\n{'-'*70}")
    print("📊 MATRICE DE COVARIANCE")
    print(f"{'-'*70}")
    print("\nLa covariance mesure comment les rendements varient ensemble")
    print()
    print(stats['matrice_covariance'].round(6))


# ============================================================
# ÉTAPE 4: OPTIMISATION DE MARKOWITZ
# ============================================================

def performance_portefeuille(poids, rendements_moyens, matrice_cov):
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


# ============================================================
# ÉTAPE 5: FRONTIÈRE EFFICIENTE
# ============================================================

def calculer_frontiere_efficiente(rendements_moyens, matrice_cov, n_portefeuilles=100):
    """
    Calcule la frontière efficiente
    
    La frontière efficiente est l'ensemble de tous les portefeuilles optimaux
    qui offrent le rendement maximum pour un niveau de risque donné.
    
    Paramètres:
    -----------
    rendements_moyens : Series
        Rendements moyens annualisés
    matrice_cov : DataFrame
        Matrice de covariance
    n_portefeuilles : int
        Nombre de points sur la frontière (100 par défaut)
    
    Retour:
    -------
    DataFrame : Points de la frontière efficiente
    """
    print(f"\n{'='*70}")
    print("📊 CALCUL DE LA FRONTIÈRE EFFICIENTE")
    print(f"{'='*70}")
    
    n_actifs = len(rendements_moyens)
    
    # Trouver le portefeuille à volatilité minimum (point de départ de la frontière)
    contraintes = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bornes = tuple((0, 1) for _ in range(n_actifs))
    poids_initial = np.array([1/n_actifs] * n_actifs)
    
    resultat_min_vol = minimize(
        lambda poids: performance_portefeuille(poids, rendements_moyens, matrice_cov)[1],
        poids_initial,
        method='SLSQP',
        bounds=bornes,
        constraints=contraintes
    )
    
    # Rendement minimum et maximum pour la frontière
    rendement_min, _ = performance_portefeuille(resultat_min_vol.x, rendements_moyens, matrice_cov)
    rendement_max = np.max(rendements_moyens)  # Le meilleur actif individuel
    
    # Générer des rendements cibles entre min et max
    rendements_cibles = np.linspace(rendement_min, rendement_max * 0.95, n_portefeuilles)
    
    portefeuilles_efficaces = []
    
    print(f"\nCalcul de {n_portefeuilles} portefeuilles optimaux...")
    
    for i, rendement_cible in enumerate(rendements_cibles):
        # Contraintes : somme = 1 ET rendement = rendement cible
        contraintes_avec_rendement = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: performance_portefeuille(x, rendements_moyens, matrice_cov)[0] - rendement_cible}
        ]
        
        # Minimiser la volatilité pour ce rendement cible
        resultat = minimize(
            lambda poids: performance_portefeuille(poids, rendements_moyens, matrice_cov)[1],
            poids_initial,
            method='SLSQP',
            bounds=bornes,
            constraints=contraintes_avec_rendement
        )
        
        if resultat.success:
            rdt, vol = performance_portefeuille(resultat.x, rendements_moyens, matrice_cov)
            sharpe = (rdt - 0.02) / vol
            
            portefeuilles_efficaces.append({
                'rendement': rdt,
                'volatilite': vol,
                'sharpe': sharpe,
                'poids': resultat.x
            })
        
        # Barre de progression
        if (i + 1) % 20 == 0:
            print(f"  → {i + 1}/{n_portefeuilles} portefeuilles calculés")
    
    print(f"✓ Frontière efficiente calculée : {len(portefeuilles_efficaces)} portefeuilles")
    
    return pd.DataFrame(portefeuilles_efficaces)


def generer_portefeuilles_aleatoires(rendements_moyens, matrice_cov, n_portefeuilles=5000):
    """
    Génère des portefeuilles aléatoires pour comparaison visuelle
    
    Ces portefeuilles servent de "fond" pour montrer que la frontière
    efficiente domine tous les autres portefeuilles possibles.
    
    Paramètres:
    -----------
    rendements_moyens : Series
        Rendements moyens annualisés
    matrice_cov : DataFrame
        Matrice de covariance
    n_portefeuilles : int
        Nombre de portefeuilles aléatoires (5000 par défaut)
    
    Retour:
    -------
    DataFrame : Portefeuilles aléatoires
    """
    print(f"\nGénération de {n_portefeuilles} portefeuilles aléatoires...")
    
    n_actifs = len(rendements_moyens)
    resultats = []
    
    for _ in range(n_portefeuilles):
        # Générer des poids aléatoires qui somment à 1
        poids = np.random.random(n_actifs)
        poids /= np.sum(poids)  # Normaliser pour que la somme = 1
        
        rdt, vol = performance_portefeuille(poids, rendements_moyens, matrice_cov)
        sharpe = (rdt - 0.02) / vol
        
        resultats.append({
            'rendement': rdt,
            'volatilite': vol,
            'sharpe': sharpe
        })
    
    print(f"✓ {n_portefeuilles} portefeuilles aléatoires générés")
    
    return pd.DataFrame(resultats)


def tracer_frontiere_efficiente(frontiere, aleatoires, stats, resultats_optim, tickers):
    """
    Trace la frontière efficiente avec tous les éléments visuels
    
    Paramètres:
    -----------
    frontiere : DataFrame
        Points de la frontière efficiente
    aleatoires : DataFrame
        Portefeuilles aléatoires
    stats : dict
        Statistiques des actifs
    resultats_optim : dict
        Résultats de l'optimisation
    tickers : list
        Liste des symboles boursiers
    """
    print(f"\n{'='*70}")
    print("📈 TRACÉ DE LA FRONTIÈRE EFFICIENTE")
    print(f"{'='*70}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ========== GRAPHIQUE 1 : FRONTIÈRE EFFICIENTE ==========
    
    # 1. Portefeuilles aléatoires (fond gris)
    scatter = ax1.scatter(
        aleatoires['volatilite'] * 100,
        aleatoires['rendement'] * 100,
        c=aleatoires['sharpe'],
        cmap='viridis',
        alpha=0.3,
        s=10,
        label='Portefeuilles aléatoires'
    )
    
    # 2. Frontière efficiente (ligne rouge)
    ax1.plot(
        frontiere['volatilite'] * 100,
        frontiere['rendement'] * 100,
        'r-',
        linewidth=3,
        label='Frontière Efficiente',
        zorder=5
    )
    
    # 3. Portefeuille Max Sharpe (étoile dorée)
    max_sharpe = resultats_optim['max_sharpe']
    rdt_sharpe, vol_sharpe = performance_portefeuille(
        max_sharpe.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    ax1.scatter(
        vol_sharpe * 100, rdt_sharpe * 100,
        marker='*', color='gold', s=800,
        label='Max Sharpe Ratio',
        edgecolors='black', linewidth=2, zorder=10
    )
    
    # 4. Portefeuille Min Volatilité (étoile rouge)
    min_vol = resultats_optim['min_volatilite']
    rdt_min, vol_min = performance_portefeuille(
        min_vol.x, stats['rendements_moyens'], stats['matrice_covariance']
    )
    ax1.scatter(
        vol_min * 100, rdt_min * 100,
        marker='*', color='red', s=800,
        label='Min Volatilité',
        edgecolors='black', linewidth=2, zorder=10
    )
    
    # 5. Actifs individuels (losanges bleus)
    ax1.scatter(
        stats['volatilite'] * 100,
        stats['rendements_moyens'] * 100,
        marker='D', s=250, alpha=0.9, c='blue',
        label='Actifs individuels',
        edgecolors='black', linewidth=1.5, zorder=8
    )
    
    # Annotations des actifs
    for i, ticker in enumerate(tickers):
        ax1.annotate(
            ticker,
            (stats['volatilite'].iloc[i] * 100, stats['rendements_moyens'].iloc[i] * 100),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=11,
            fontweight='bold'
        )
    
    # Labels et titre
    ax1.set_xlabel('Volatilité / Risque (% annuel)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Rendement Espéré (% annuel)', fontsize=13, fontweight='bold')
    ax1.set_title('Frontière Efficiente de Markowitz', fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Colorbar pour le ratio de Sharpe
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Ratio de Sharpe', fontsize=11)
    
    # ========== GRAPHIQUE 2 : ALLOCATION DES POIDS ==========
    
    poids_sharpe = max_sharpe.x * 100
    poids_min_vol = min_vol.x * 100
    
    x = np.arange(len(tickers))
    largeur = 0.35
    
    # Barres pour Max Sharpe
    barres1 = ax2.bar(
        x - largeur/2, poids_sharpe, largeur,
        label='Max Sharpe', alpha=0.8, color='gold', edgecolor='black'
    )
    
    # Barres pour Min Volatilité
    barres2 = ax2.bar(
        x + largeur/2, poids_min_vol, largeur,
        label='Min Volatilité', alpha=0.8, color='red', edgecolor='black'
    )
    
    # Ajouter les valeurs sur les barres
    for barres in [barres1, barres2]:
        for barre in barres:
            hauteur = barre.get_height()
            if hauteur > 2:  # Afficher seulement si > 2%
                ax2.text(
                    barre.get_x() + barre.get_width()/2., hauteur,
                    f'{hauteur:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold'
                )
    
    # Labels et titre
    ax2.set_xlabel('Actifs', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Allocation (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Allocation Optimale des Actifs', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, fontsize=11)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(max(poids_sharpe), max(poids_min_vol)) * 1.15)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig('/mnt/user-data/outputs/frontiere_efficiente.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphique sauvegardé : frontiere_efficiente.png")
    
    plt.show()


# ============================================================
# FONCTION PRINCIPALE - EXÉCUTION COMPLÈTE
# ============================================================

def main():
    """
    Fonction principale qui exécute toutes les étapes de l'analyse Markowitz
    """
    print("\n" + "="*70)
    print("  🎓 OPTIMISATION DE PORTFOLIO - MÉTHODE DE MARKOWITZ")
    print("="*70)
    print("\n  La théorie moderne du portefeuille développée par Harry Markowitz")
    print("  permet d'optimiser l'allocation d'actifs pour maximiser le rendement")
    print("  pour un niveau de risque donné.\n")
    
    # ===== CONFIGURATION =====
    # Modifiez ces paramètres selon vos besoins
    tickers_demandes = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    date_fin = datetime.now().strftime('%Y-%m-%d')
    date_debut = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    # ===== ÉTAPE 1 : IMPORTATION DES DONNÉES =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 1 : Importation des données depuis Yahoo Finance")
    print(f"{'─'*70}")
    
    try:
        prix = importer_donnees(tickers_demandes, date_debut, date_fin)
        # Récupérer les tickers RÉELLEMENT téléchargés (peut être différent si échec)
        tickers = prix.columns.tolist()
    except ValueError as e:
        print(e)
        print("\n💡 CONSEIL : Essayez avec d'autres symboles boursiers ou une période différente.")
        return
    
    # ===== ÉTAPE 2 : CALCUL DES RENDEMENTS =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 2 : Calcul des rendements")
    print(f"{'─'*70}")
    rendements = calculer_rendements(prix)
    
    # Vérification : rendements valides
    if len(rendements) == 0:
        print("\n❌ ERREUR : Impossible de calculer les rendements (données insuffisantes).")
        return
    
    # ===== ÉTAPE 3 : CALCUL DES STATISTIQUES =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 3 : Calcul des statistiques (variance, covariance, corrélation)")
    print(f"{'─'*70}")
    stats = calculer_statistiques(rendements)
    afficher_statistiques(stats)
    
    # ===== ÉTAPE 4 : OPTIMISATION =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 4 : Optimisation de Markowitz")
    print(f"{'─'*70}")
    resultats_optim = optimiser_portefeuille(stats['rendements_moyens'], stats['matrice_covariance'])
    afficher_portefeuilles_optimaux(resultats_optim, stats, tickers)  # Utiliser tickers réels
    
    # ===== ÉTAPE 5 : FRONTIÈRE EFFICIENTE =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 5 : Calcul et tracé de la frontière efficiente")
    print(f"{'─'*70}")
    
    frontiere = calculer_frontiere_efficiente(
        stats['rendements_moyens'], 
        stats['matrice_covariance'], 
        n_portefeuilles=100
    )
    
    aleatoires = generer_portefeuilles_aleatoires(
        stats['rendements_moyens'], 
        stats['matrice_covariance'], 
        n_portefeuilles=5000
    )
    
    tracer_frontiere_efficiente(frontiere, aleatoires, stats, resultats_optim, tickers)
    
    # ===== RÉSUMÉ FINAL =====
    print(f"\n{'='*70}")
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
    print(f"{'='*70}")
    print("\n📊 Points clés de la théorie de Markowitz :")
    print("  • La frontière efficiente montre tous les portefeuilles optimaux")
    print("  • La diversification réduit le risque grâce aux corrélations")
    print("  • Le portefeuille Max Sharpe offre le meilleur ratio rendement/risque")
    print("  • Le portefeuille Min Volatilité est idéal pour un profil conservateur")
    print("\n💾 Fichier sauvegardé : frontiere_efficiente.png")
    print()


if __name__ == "__main__":
    main()
