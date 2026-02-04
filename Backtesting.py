"""
BACKTEST RÉALISTE DE MARKOWITZ
=================================

Ce script teste la stratégie de Markowitz en conditions réelles avec :
- Rolling covariance (fenêtre glissante)
- Rebalancement mensuel
- Coûts de transaction
- Métriques de performance complètes

Objectif : Comprendre pourquoi Markowitz "casse" en pratique
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION DU BACKTEST
# ============================================================

CONFIG = {
    # Paramètres du portfolio
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'],
    
    # Période de backtest
    'date_debut': '2020-01-01',
    'date_fin': '2024-12-31',
    
    # Paramètres de la fenêtre roulante
    'fenetre_estimation': 252,  # 1 an de données pour estimer cov
    'frequence_rebalancement': 21,  # Rebalancer tous les 21 jours (≈1 mois)
    
    # Coûts de transaction
    'cout_transaction': 0.001,  # 0.1% par trade (buy + sell)
    
    # Paramètres financiers
    'taux_sans_risque': 0.02,
    'capital_initial': 100000,  # 100k $
    
    # Contraintes sur les poids (éviter positions extrêmes)
    'poids_min': 0.00,  # Minimum par actif
    'poids_max': 0.40,  # Maximum 40% par actif
    
    # Graphiques
    'style_graphique': 'seaborn-v0_8-darkgrid',
    'taille_figure': (16, 10),
}

plt.style.use(CONFIG['style_graphique'])

print("="*70)
print("🔬 BACKTEST RÉALISTE DE MARKOWITZ")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Actifs : {', '.join(CONFIG['tickers'])}")
print(f"  • Période : {CONFIG['date_debut']} à {CONFIG['date_fin']}")
print(f"  • Fenêtre d'estimation : {CONFIG['fenetre_estimation']} jours")
print(f"  • Rebalancement tous les : {CONFIG['frequence_rebalancement']} jours")
print(f"  • Coûts de transaction : {CONFIG['cout_transaction']*100}%")
print(f"  • Capital initial : ${CONFIG['capital_initial']:,.0f}")
print()


# ============================================================
# FONCTIONS D'IMPORT ET PRÉPARATION DES DONNÉES
# ============================================================

def telecharger_donnees(tickers, date_debut, date_fin):
    """
    Télécharge les données historiques pour le backtest
    """
    print(f"📊 Téléchargement des données...")
    
    data = yf.download(tickers, start=date_debut, end=date_fin, progress=False)
    
    # Extraction des prix
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            prix = data['Adj Close'].copy()
        else:
            prix = data['Close'].copy()
    else:
        if 'Adj Close' in data.columns:
            prix = data[['Adj Close']].copy()
            prix.columns = tickers
        else:
            prix = data[['Close']].copy()
            prix.columns = tickers
    
    # Nettoyage
    prix = prix.dropna()
    
    print(f"✓ {len(prix)} jours de données téléchargés")
    print(f"  Période : {prix.index[0].date()} à {prix.index[-1].date()}")
    
    return prix


def calculer_rendements(prix):
    """Calcule les rendements logarithmiques"""
    return np.log(prix / prix.shift(1)).dropna()


# ============================================================
# FONCTIONS D'OPTIMISATION MARKOWITZ
# ============================================================

def calculer_performance_portfolio(poids, rendements_moyens, matrice_cov):
    """Calcule rendement et volatilité d'un portfolio"""
    rendement = np.sum(poids * rendements_moyens)
    volatilite = np.sqrt(np.dot(poids.T, np.dot(matrice_cov, poids)))
    return rendement, volatilite


def sharpe_negatif(poids, rendements_moyens, matrice_cov, rf=0.02):
    """Ratio de Sharpe négatif pour minimisation"""
    r, v = calculer_performance_portfolio(poids, rendements_moyens, matrice_cov)
    return -(r - rf) / v


def optimiser_markowitz(rendements_moyens, matrice_cov, poids_min=0.0, poids_max=1.0):
    """
    Optimise le portfolio selon Markowitz (Max Sharpe)
    
    Avec contraintes sur les poids pour éviter les positions extrêmes
    """
    n_actifs = len(rendements_moyens)
    
    # Contraintes
    contraintes = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Somme = 1
    ]
    
    # Bornes sur les poids
    bornes = tuple((poids_min, poids_max) for _ in range(n_actifs))
    
    # Point de départ
    poids_initial = np.array([1/n_actifs] * n_actifs)
    
    # Optimisation
    resultat = minimize(
        sharpe_negatif,
        poids_initial,
        args=(rendements_moyens, matrice_cov),
        method='SLSQP',
        bounds=bornes,
        constraints=contraintes
    )
    
    if resultat.success:
        return resultat.x
    else:
        # Si échec, retourner équipondération
        return poids_initial


# ============================================================
# STRATÉGIES DE BENCHMARK
# ============================================================

def strategie_buy_and_hold(prix):
    """
    Buy and Hold équipondéré
    Achète au début et ne touche plus
    """
    n_actifs = len(prix.columns)
    poids = np.array([1/n_actifs] * n_actifs)
    
    # Valeur du portfolio chaque jour
    rendements = calculer_rendements(prix)
    rendements_portfolio = (rendements * poids).sum(axis=1)
    valeur_portfolio = CONFIG['capital_initial'] * (1 + rendements_portfolio).cumprod()
    
    return valeur_portfolio, poids


def strategie_equiponderee_rebalancee(prix, frequence_rebal):
    """
    Portfolio équipondéré avec rebalancement périodique
    Simple mais efficace comme benchmark
    """
    n_actifs = len(prix.columns)
    poids_cible = np.array([1/n_actifs] * n_actifs)
    
    rendements = calculer_rendements(prix)
    valeur_portfolio = pd.Series(index=rendements.index, dtype=float)
    valeur_portfolio.iloc[0] = CONFIG['capital_initial']
    
    poids_actuels = poids_cible.copy()
    jours_depuis_rebal = 0
    
    for i in range(1, len(rendements)):
        # Rendement du jour
        rdt_jour = rendements.iloc[i].values
        
        # Mise à jour de la valeur
        valeur_portfolio.iloc[i] = valeur_portfolio.iloc[i-1] * (1 + np.sum(poids_actuels * rdt_jour))
        
        # Mise à jour des poids (drift naturel)
        poids_actuels = poids_actuels * (1 + rdt_jour)
        poids_actuels = poids_actuels / np.sum(poids_actuels)
        
        jours_depuis_rebal += 1
        
        # Rebalancement périodique
        if jours_depuis_rebal >= frequence_rebal:
            # Coûts de transaction
            turnover = np.sum(np.abs(poids_actuels - poids_cible))
            cout = turnover * CONFIG['cout_transaction'] * valeur_portfolio.iloc[i]
            valeur_portfolio.iloc[i] -= cout
            
            # Reset des poids
            poids_actuels = poids_cible.copy()
            jours_depuis_rebal = 0
    
    return valeur_portfolio, poids_cible


print("✓ Fonctions de base chargées")
print()


# ============================================================
# BACKTEST MARKOWITZ AVEC ROLLING WINDOW
# ============================================================

def backtest_markowitz_rolling(prix, fenetre, frequence_rebal, poids_min, poids_max):
    """
    Backtest de Markowitz avec fenêtre glissante (rolling window)
    
    Processus :
    1. À chaque date de rebalancement :
       - Utiliser les N derniers jours pour estimer rendements et covariance
       - Optimiser les poids selon Markowitz
       - Calculer les coûts de transaction
    2. Entre les rebalancements :
       - Laisser les poids dériver naturellement
    
    Paramètres :
    -----------
    prix : DataFrame
        Prix historiques
    fenetre : int
        Taille de la fenêtre d'estimation (ex: 252 jours)
    frequence_rebal : int
        Fréquence de rebalancement en jours
    poids_min, poids_max : float
        Contraintes sur les poids
    
    Retours :
    --------
    valeur_portfolio : Series
        Valeur du portfolio dans le temps
    historique_poids : DataFrame
        Évolution des poids
    metriques : dict
        Statistiques de performance
    """
    print(f"🔄 Backtest Markowitz avec rolling window...")
    print(f"   Fenêtre d'estimation : {fenetre} jours")
    print(f"   Rebalancement tous les : {frequence_rebal} jours")
    
    n_actifs = len(prix.columns)
    rendements = calculer_rendements(prix)
    
    # Initialisation
    valeur_portfolio = pd.Series(index=rendements.index, dtype=float)
    valeur_portfolio.iloc[0] = CONFIG['capital_initial']
    
    # Historique des poids et métriques
    historique_poids = []
    historique_turnover = []
    historique_couts = []
    
    # Poids initiaux (équipondérés)
    poids_actuels = np.array([1/n_actifs] * n_actifs)
    
    jours_depuis_rebal = 0
    n_rebalancements = 0
    
    # Parcourir chaque jour
    for i in range(1, len(rendements)):
        
        # Rendement du jour
        rdt_jour = rendements.iloc[i].values
        
        # Mise à jour de la valeur du portfolio
        rendement_portfolio = np.sum(poids_actuels * rdt_jour)
        valeur_portfolio.iloc[i] = valeur_portfolio.iloc[i-1] * (1 + rendement_portfolio)
        
        # Mise à jour des poids (drift naturel suite aux rendements)
        poids_actuels = poids_actuels * (1 + rdt_jour)
        poids_actuels = poids_actuels / np.sum(poids_actuels)  # Renormaliser
        
        jours_depuis_rebal += 1
        
        # REBALANCEMENT si nécessaire
        if jours_depuis_rebal >= frequence_rebal and i >= fenetre:
            
            # Extraire la fenêtre de données pour estimation
            rendements_fenetre = rendements.iloc[i-fenetre:i]
            
            # Calculer statistiques sur cette fenêtre
            rendements_moyens = rendements_fenetre.mean() * 252  # Annualisé
            matrice_cov = rendements_fenetre.cov() * 252  # Annualisée
            
            # Optimisation de Markowitz
            poids_optimaux = optimiser_markowitz(
                rendements_moyens, 
                matrice_cov, 
                poids_min, 
                poids_max
            )
            
            # Calculer le turnover (combien on trade)
            turnover = np.sum(np.abs(poids_optimaux - poids_actuels))
            
            # Coûts de transaction
            cout = turnover * CONFIG['cout_transaction'] * valeur_portfolio.iloc[i]
            valeur_portfolio.iloc[i] -= cout
            
            # Enregistrer les métriques
            historique_poids.append({
                'date': rendements.index[i],
                **{f'poids_{ticker}': poids_optimaux[j] for j, ticker in enumerate(prix.columns)}
            })
            historique_turnover.append(turnover)
            historique_couts.append(cout)
            
            # Appliquer les nouveaux poids
            poids_actuels = poids_optimaux.copy()
            jours_depuis_rebal = 0
            n_rebalancements += 1
            
            if n_rebalancements % 10 == 0:
                print(f"   → {n_rebalancements} rebalancements effectués...")
    
    print(f"✓ Backtest terminé : {n_rebalancements} rebalancements")
    
    # Convertir l'historique en DataFrame
    df_poids = pd.DataFrame(historique_poids)
    if len(df_poids) > 0:
        df_poids.set_index('date', inplace=True)
    
    # Calculer les métriques
    metriques = calculer_metriques_performance(valeur_portfolio, historique_turnover, historique_couts)
    
    return valeur_portfolio, df_poids, metriques


# ============================================================
# CALCUL DES MÉTRIQUES DE PERFORMANCE
# ============================================================

def calculer_metriques_performance(valeur_portfolio, historique_turnover, historique_couts):
    """
    Calcule toutes les métriques de performance
    
    Métriques calculées :
    - Rendement total
    - Rendement annualisé
    - Volatilité annualisée
    - Sharpe ratio
    - Maximum Drawdown
    - Turnover moyen
    - Coûts totaux
    """
    # Rendements
    rendements_portfolio = valeur_portfolio.pct_change().dropna()
    
    # Rendement total
    rendement_total = (valeur_portfolio.iloc[-1] / valeur_portfolio.iloc[0]) - 1
    
    # Nombre d'années
    n_jours = len(valeur_portfolio)
    n_annees = n_jours / 252
    
    # Rendement annualisé
    rendement_annualise = (1 + rendement_total) ** (1 / n_annees) - 1
    
    # Volatilité annualisée
    volatilite = rendements_portfolio.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = (rendement_annualise - CONFIG['taux_sans_risque']) / volatilite
    
    # Maximum Drawdown
    cummax = valeur_portfolio.cummax()
    drawdown = (valeur_portfolio - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Turnover moyen
    turnover_moyen = np.mean(historique_turnover) if historique_turnover else 0
    
    # Coûts totaux
    couts_totaux = np.sum(historique_couts) if historique_couts else 0
    pct_couts = couts_totaux / CONFIG['capital_initial']
    
    return {
        'rendement_total': rendement_total,
        'rendement_annualise': rendement_annualise,
        'volatilite': volatilite,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'turnover_moyen': turnover_moyen,
        'couts_totaux': couts_totaux,
        'pct_couts': pct_couts,
    }


def afficher_metriques(nom_strategie, metriques):
    """Affiche les métriques de manière formatée"""
    print(f"\n{'='*60}")
    print(f"📊 {nom_strategie}")
    print(f"{'='*60}")
    print(f"  Rendement total       : {metriques['rendement_total']*100:>8.2f}%")
    print(f"  Rendement annualisé   : {metriques['rendement_annualise']*100:>8.2f}%")
    print(f"  Volatilité annuelle   : {metriques['volatilite']*100:>8.2f}%")
    print(f"  Sharpe Ratio          : {metriques['sharpe']:>8.2f}")
    print(f"  Maximum Drawdown      : {metriques['max_drawdown']*100:>8.2f}%")
    
    if 'turnover_moyen' in metriques:
        print(f"  Turnover moyen        : {metriques['turnover_moyen']*100:>8.2f}%")
    if 'couts_totaux' in metriques:
        print(f"  Coûts totaux          : ${metriques['couts_totaux']:>8,.0f}")
        print(f"  Coûts (% capital)     : {metriques['pct_couts']*100:>8.2f}%")


# ============================================================
# VISUALISATIONS
# ============================================================

def visualiser_resultats(resultats_strategies, prix):
    """
    Crée des visualisations complètes pour comparer les stratégies
    
    Graphiques créés :
    1. Courbes de performance (valeur du portfolio)
    2. Drawdown dans le temps
    3. Tableau comparatif des métriques
    4. Évolution des poids (Markowitz)
    """
    fig = plt.figure(figsize=CONFIG['taille_figure'])
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========== GRAPHIQUE 1 : PERFORMANCE ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    for nom, data in resultats_strategies.items():
        valeur = data['valeur']
        ax1.plot(valeur.index, valeur.values, label=nom, linewidth=2)
    
    ax1.set_title('Performance des Stratégies (Valeur du Portfolio)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Valeur ($)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ========== GRAPHIQUE 2 : DRAWDOWN ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    for nom, data in resultats_strategies.items():
        valeur = data['valeur']
        cummax = valeur.cummax()
        drawdown = (valeur - cummax) / cummax * 100
        ax2.plot(drawdown.index, drawdown.values, label=nom, linewidth=1.5)
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.2)
    
    ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ========== GRAPHIQUE 3 : RENDEMENTS CUMULÉS (%) ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    for nom, data in resultats_strategies.items():
        valeur = data['valeur']
        rendement_cumule = (valeur / valeur.iloc[0] - 1) * 100
        ax3.plot(rendement_cumule.index, rendement_cumule.values, label=nom, linewidth=2)
    
    ax3.set_title('Rendements Cumulés (%)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Rendement (%)', fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # ========== GRAPHIQUE 4 : TABLEAU COMPARATIF ==========
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Préparer les données du tableau
    metriques_noms = [
        'Rendement Total',
        'Rendement Annualisé',
        'Volatilité',
        'Sharpe Ratio',
        'Max Drawdown',
        'Turnover Moyen',
        'Coûts Totaux'
    ]
    
    donnees_tableau = []
    for nom in resultats_strategies.keys():
        m = resultats_strategies[nom]['metriques']
        donnees_tableau.append([
            f"{m['rendement_total']*100:.2f}%",
            f"{m['rendement_annualise']*100:.2f}%",
            f"{m['volatilite']*100:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']*100:.2f}%",
            f"{m.get('turnover_moyen', 0)*100:.2f}%",
            f"${m.get('couts_totaux', 0):,.0f}",
        ])
    
    # Transposer pour avoir stratégies en colonnes
    donnees_tableau_t = list(map(list, zip(*donnees_tableau)))
    
    table = ax4.table(
        cellText=donnees_tableau_t,
        rowLabels=metriques_noms,
        colLabels=list(resultats_strategies.keys()),
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Colorer les headers
    for i in range(len(resultats_strategies)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(metriques_noms)):
        table[(i+1, -1)].set_facecolor('#d9d9d9')
        table[(i+1, -1)].set_text_props(weight='bold')
    
    ax4.set_title('Comparaison des Métriques', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('BACKTEST COMPARATIF - MARKOWITZ vs BENCHMARKS', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Sauvegarder
    import os
    if os.path.exists('/mnt/user-data/outputs/'):
        chemin = '/mnt/user-data/outputs/backtest_resultats.png'
    else:
        chemin = 'backtest_resultats.png'
    
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé : {chemin}")
    
    plt.show()


def visualiser_evolution_poids(historique_poids, tickers):
    """
    Visualise l'évolution des poids du portfolio Markowitz dans le temps
    """
    if historique_poids is None or len(historique_poids) == 0:
        print("⚠️  Pas d'historique de poids à afficher")
        return
    
    print(f"\n📊 Visualisation de l'évolution des poids...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Extraire les colonnes de poids
    cols_poids = [col for col in historique_poids.columns if col.startswith('poids_')]
    
    # ========== GRAPHIQUE 1 : Évolution des poids (stacked area) ==========
    poids_data = historique_poids[cols_poids]
    poids_data.columns = [col.replace('poids_', '') for col in cols_poids]
    
    ax1.stackplot(poids_data.index, *[poids_data[col].values for col in poids_data.columns],
                  labels=poids_data.columns, alpha=0.7)
    
    ax1.set_title('Évolution des Poids du Portfolio Markowitz', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Poids (%)', fontsize=11)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    # ========== GRAPHIQUE 2 : Poids individuels (lignes) ==========
    for col in poids_data.columns:
        ax2.plot(poids_data.index, poids_data[col].values * 100, label=col, linewidth=2, marker='o', markersize=3)
    
    ax2.set_title('Poids Individuels dans le Temps', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Poids (%)', fontsize=11)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=CONFIG['poids_max']*100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max')
    ax2.axhline(y=CONFIG['poids_min']*100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Min')
    
    plt.tight_layout()
    
    # Sauvegarder
    import os
    if os.path.exists('/mnt/user-data/outputs/'):
        chemin = '/mnt/user-data/outputs/evolution_poids.png'
    else:
        chemin = 'evolution_poids.png'
    
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique des poids sauvegardé : {chemin}")
    
    plt.show()



# ============================================================
# FONCTION PRINCIPALE

def main():
    """
    Exécute le backtest complet et compare les stratégies
    """
    print("\n" + "="*70)
    print("🚀 LANCEMENT DU BACKTEST")
    print("="*70)
    
    # ===== ÉTAPE 1 : TÉLÉCHARGEMENT DES DONNÉES =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 1 : Téléchargement des données")
    print(f"{'─'*70}")
    
    prix = telecharger_donnees(
        CONFIG['tickers'],
        CONFIG['date_debut'],
        CONFIG['date_fin']
    )
    
    # ===== ÉTAPE 2 : BACKTEST MARKOWITZ =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 2 : Backtest Markowitz (Rolling Window)")
    print(f"{'─'*70}")
    
    valeur_markowitz, poids_markowitz, metriques_markowitz = backtest_markowitz_rolling(
        prix,
        fenetre=CONFIG['fenetre_estimation'],
        frequence_rebal=CONFIG['frequence_rebalancement'],
        poids_min=CONFIG['poids_min'],
        poids_max=CONFIG['poids_max']
    )
    
    afficher_metriques("MARKOWITZ (Rolling)", metriques_markowitz)
    
    # ===== ÉTAPE 3 : BENCHMARK BUY & HOLD =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 3 : Benchmark - Buy & Hold Équipondéré")
    print(f"{'─'*70}")
    
    valeur_buyhold, poids_buyhold = strategie_buy_and_hold(prix)
    metriques_buyhold = calculer_metriques_performance(valeur_buyhold, [], [])
    
    afficher_metriques("BUY & HOLD", metriques_buyhold)
    
    # ===== ÉTAPE 4 : BENCHMARK ÉQUIPONDÉRÉ REBALANCÉ =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 4 : Benchmark - Équipondéré Rebalancé")
    print(f"{'─'*70}")
    
    valeur_equi, poids_equi = strategie_equiponderee_rebalancee(
        prix, 
        CONFIG['frequence_rebalancement']
    )
    metriques_equi = calculer_metriques_performance(valeur_equi, [0.2]*10, [1000]*10)
    
    afficher_metriques("ÉQUIPONDÉRÉ REBALANCÉ", metriques_equi)
    
    # ===== ÉTAPE 5 : VISUALISATIONS =====
    print(f"\n{'─'*70}")
    print("ÉTAPE 5 : Visualisations comparatives")
    print(f"{'─'*70}")
    
    resultats_strategies = {
        'Markowitz (Rolling)': {
            'valeur': valeur_markowitz,
            'metriques': metriques_markowitz
        },
        'Buy & Hold': {
            'valeur': valeur_buyhold,
            'metriques': metriques_buyhold
        },
        'Équipondéré Rebalancé': {
            'valeur': valeur_equi,
            'metriques': metriques_equi
        }
    }
    
    visualiser_resultats(resultats_strategies, prix)
    visualiser_evolution_poids(poids_markowitz, CONFIG['tickers'])
    
    print(f"\n{'='*70}")
    print("✅ BACKTEST TERMINÉ !")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
