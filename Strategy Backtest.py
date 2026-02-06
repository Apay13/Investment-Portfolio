"""
RISK PARITY AVANCÉ - Shrinkage + HRP
=====================================

Nouvelles stratégies :
1. Markowitz avec Shrinkage (Ledoit-Wolf)
2. Risk Parity Simple
3. Risk Parity avec Shrinkage
4. Hierarchical Risk Parity (HRP)

Objectif : Comparer la robustesse et stabilité
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Paramètres du portfolio
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'GS', 'JNJ', 'PFE'],
    
    # Période de backtest
    'date_debut': '2019-01-01',
    'date_fin': '2025-12-31',
    
    # Paramètres de la fenêtre roulante
    'fenetre_estimation': 252,
    'frequence_rebalancement': 21,
    
    # Coûts de transaction
    'cout_transaction': 0.001,
    
    # Paramètres financiers
    'taux_sans_risque': 0.02,
    'capital_initial': 100000,
    
    # Graphiques
    'style_graphique': 'seaborn-v0_8-darkgrid',
    'taille_figure': (18, 12),
}

plt.style.use(CONFIG['style_graphique'])

print("="*70)
print("🔬 RISK PARITY AVANCÉ - Shrinkage + HRP")
print("="*70)
print(f"\nNouveautés:")
print(f"  • Shrinkage Ledoit-Wolf (réduction bruit)")
print(f"  • Hierarchical Risk Parity (clustering)")
print()


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def telecharger_donnees(tickers, date_debut, date_fin):
    """Télécharge les données historiques"""
    print(f"📊 Téléchargement des données...")
    
    data = yf.download(tickers, start=date_debut, end=date_fin, progress=False)
    
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
    
    prix = prix.dropna()
    
    print(f"✓ {len(prix)} jours de données téléchargés")
    print(f"  Période : {prix.index[0].date()} à {prix.index[-1].date()}")
    
    return prix


def calculer_rendements(prix):
    """Calcule les rendements logarithmiques"""
    return np.log(prix / prix.shift(1)).dropna()


def calculer_performance_portfolio(poids, rendements_moyens, matrice_cov):
    """Calcule rendement et volatilité d'un portfolio"""
    rendement = np.sum(poids * rendements_moyens)
    volatilite = np.sqrt(np.dot(poids.T, np.dot(matrice_cov, poids)))
    return rendement, volatilite


# ============================================================
# SHRINKAGE LEDOIT-WOLF
# ============================================================

def ledoit_wolf_shrinkage(rendements):
    """
    Implémente le shrinkage de Ledoit-Wolf
    
    Formule : Σ_shrunk = δ×F + (1-δ)×S
    
    Où :
    - S = matrice de covariance empirique
    - F = matrice cible (constant correlation)
    - δ = intensité de shrinkage optimale
    
    Retour :
    --------
    tuple : (matrice_cov_shrunk, delta)
    """
    X = rendements.values
    n, p = X.shape
    
    # Centrer les données
    X = X - X.mean(axis=0)
    
    # Matrice de covariance empirique
    S = np.cov(X.T, bias=True)
    
    # Variance moyenne
    var_mean = np.trace(S) / p
    
    # Matrice cible F (constant correlation)
    corr_mean = (np.sum(S) - np.trace(S)) / (p * (p - 1))
    F = corr_mean * np.ones((p, p))
    np.fill_diagonal(F, var_mean)
    
    # Calcul de delta optimal (formule de Ledoit-Wolf)
    # Simplification : utiliser une formule approchée
    diff = S - F
    delta = min(1, max(0, np.sum(diff ** 2) / (n * np.sum(S ** 2))))
    
    # Matrice shrunk
    S_shrunk = delta * F + (1 - delta) * S
    
    return S_shrunk, delta


# ============================================================
# HIERARCHICAL RISK PARITY (HRP)
# ============================================================

def get_quasi_diag(link):
    """
    Récupère l'ordre quasi-diagonal à partir du clustering
    (algorithme de Marcos López de Prado)
    
    Retourne les INDICES numériques
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    # Retourner liste d'entiers
    return [int(x) for x in sort_ix.tolist()]


def get_cluster_var(cov, c_items):
    """
    Calcule la variance d'un cluster
    
    c_items peut être des labels (strings) ou des indices (int)
    """
    # Utiliser .loc au lieu de .iloc pour gérer les labels
    cov_slice = cov.loc[c_items, c_items]
    w = inverse_variance_weights(np.diag(cov_slice))
    c_var = np.dot(np.dot(w, cov_slice), w)
    return c_var


def inverse_variance_weights(variances):
    """Poids inverse variance"""
    inv_var = 1.0 / variances
    return inv_var / inv_var.sum()


def hrp_allocation(rendements, matrice_cov):
    """
    Hierarchical Risk Parity
    
    Algorithme :
    1. Clustering hiérarchique des actifs (similitude = corrélation)
    2. Réorganisation quasi-diagonale
    3. Allocation récursive top-down
    
    Retour :
    --------
    array : Poids optimaux
    """
    # 1. Matrice de corrélation
    corr = rendements.corr()
    
    # 2. Distance = sqrt(0.5 × (1 - corrélation))
    dist = np.sqrt(0.5 * (1 - corr))
    
    # 3. Clustering hiérarchique (single linkage)
    dist_condensed = squareform(dist.values, checks=False)
    link = linkage(dist_condensed, method='single')
    
    # 4. Ordre quasi-diagonal (indices numériques)
    sort_ix_numeric = get_quasi_diag(link)
    
    # 5. Convertir en labels
    sort_ix_labels = [corr.index[i] for i in sort_ix_numeric]
    
    # 6. Réorganiser la matrice de covariance
    cov_sorted = matrice_cov.loc[sort_ix_labels, sort_ix_labels]
    
    # 7. Allocation récursive
    weights = pd.Series(1.0, index=sort_ix_labels)
    c_items = [sort_ix_labels]
    
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items 
                   for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                   if len(i) > 1]
        
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            
            # Variance de chaque cluster
            c_var0 = get_cluster_var(cov_sorted, c_items0)
            c_var1 = get_cluster_var(cov_sorted, c_items1)
            
            # Allocation inverse variance entre clusters
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            
            weights[c_items0] *= alpha
            weights[c_items1] *= (1 - alpha)
    
    # Réorganiser dans l'ordre original
    weights = weights.reindex(matrice_cov.index)
    
    return weights.values


# ============================================================
# STRATÉGIES
# ============================================================

def risk_parity_simple(volatilites):
    """Risk Parity Simple : Inverse Volatility"""
    inverse_vol = 1 / volatilites
    return inverse_vol / np.sum(inverse_vol)


def sharpe_negatif(poids, rendements_moyens, matrice_cov, rf=0.02):
    """Ratio de Sharpe négatif pour minimisation"""
    r, v = calculer_performance_portfolio(poids, rendements_moyens, matrice_cov)
    return -(r - rf) / v


# ============================================================
# BACKTESTS
# ============================================================

def backtest_strategie(prix, fenetre, frequence_rebal, strategie='rp_simple', use_shrinkage=False):
    """
    Backtest générique
    
    Stratégies disponibles :
    - 'rp_simple' : Risk Parity Simple
    - 'hrp' : Hierarchical Risk Parity
    - 'markowitz' : Markowitz Max Sharpe
    """
    nom_strat = {
        'rp_simple': 'Risk Parity Simple',
        'hrp': 'HRP',
        'markowitz': 'Markowitz'
    }[strategie]
    
    if use_shrinkage:
        nom_strat += ' + Shrinkage'
    
    print(f"\n🎯 Backtest {nom_strat}...")
    
    n_actifs = len(prix.columns)
    rendements = calculer_rendements(prix)
    
    valeur_portfolio = pd.Series(index=rendements.index, dtype=float)
    valeur_portfolio.iloc[0] = CONFIG['capital_initial']
    
    historique_poids = []
    historique_turnover = []
    historique_couts = []
    historique_shrinkage = []
    
    poids_actuels = np.array([1/n_actifs] * n_actifs)
    jours_depuis_rebal = 0
    n_rebalancements = 0
    
    for i in range(1, len(rendements)):
        
        rdt_jour = rendements.iloc[i].values
        rendement_portfolio = np.sum(poids_actuels * rdt_jour)
        valeur_portfolio.iloc[i] = valeur_portfolio.iloc[i-1] * (1 + rendement_portfolio)
        
        poids_actuels = poids_actuels * (1 + rdt_jour)
        poids_actuels = poids_actuels / np.sum(poids_actuels)
        
        jours_depuis_rebal += 1
        
        # REBALANCEMENT
        if jours_depuis_rebal >= frequence_rebal and i >= fenetre:
            
            rendements_fenetre = rendements.iloc[i-fenetre:i]
            
            # Calcul de la matrice de covariance
            if use_shrinkage:
                matrice_cov_np, delta = ledoit_wolf_shrinkage(rendements_fenetre)
                matrice_cov = pd.DataFrame(
                    matrice_cov_np * 252, 
                    index=prix.columns, 
                    columns=prix.columns
                )
                historique_shrinkage.append(delta)
            else:
                matrice_cov = rendements_fenetre.cov() * 252
            
            # Calcul des poids selon la stratégie
            if strategie == 'rp_simple':
                volatilites = rendements_fenetre.std() * np.sqrt(252)
                poids_optimaux = risk_parity_simple(volatilites.values)
            
            elif strategie == 'hrp':
                poids_optimaux = hrp_allocation(rendements_fenetre, matrice_cov)
            
            elif strategie == 'markowitz':
                rendements_moyens = rendements_fenetre.mean() * 252
                
                contraintes = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                bornes = tuple((0, 0.40) for _ in range(n_actifs))
                poids_init = np.array([1/n_actifs] * n_actifs)
                
                resultat = minimize(
                    sharpe_negatif,
                    poids_init,
                    args=(rendements_moyens, matrice_cov),
                    method='SLSQP',
                    bounds=bornes,
                    constraints=contraintes
                )
                
                poids_optimaux = resultat.x if resultat.success else poids_init
            
            # Turnover et coûts
            turnover = np.sum(np.abs(poids_optimaux - poids_actuels))
            cout = turnover * CONFIG['cout_transaction'] * valeur_portfolio.iloc[i]
            valeur_portfolio.iloc[i] -= cout
            
            historique_poids.append({
                'date': rendements.index[i],
                **{f'poids_{ticker}': poids_optimaux[j] for j, ticker in enumerate(prix.columns)}
            })
            historique_turnover.append(turnover)
            historique_couts.append(cout)
            
            poids_actuels = poids_optimaux.copy()
            jours_depuis_rebal = 0
            n_rebalancements += 1
            
            if n_rebalancements % 10 == 0:
                print(f"   → {n_rebalancements} rebalancements...")
    
    print(f"✓ Backtest terminé : {n_rebalancements} rebalancements")
    
    if use_shrinkage and len(historique_shrinkage) > 0:
        print(f"   Shrinkage moyen : {np.mean(historique_shrinkage):.3f}")
    
    df_poids = pd.DataFrame(historique_poids)
    if len(df_poids) > 0:
        df_poids.set_index('date', inplace=True)
    
    metriques = calculer_metriques_performance(
        valeur_portfolio, historique_turnover, historique_couts
    )
    
    return valeur_portfolio, df_poids, metriques


# ============================================================
# MÉTRIQUES
# ============================================================

def calculer_metriques_performance(valeur_portfolio, historique_turnover, historique_couts):
    """Calcule toutes les métriques de performance"""
    rendements_portfolio = valeur_portfolio.pct_change().dropna()
    
    rendement_total = (valeur_portfolio.iloc[-1] / valeur_portfolio.iloc[0]) - 1
    n_jours = len(valeur_portfolio)
    n_annees = n_jours / 252
    rendement_annualise = (1 + rendement_total) ** (1 / n_annees) - 1
    
    volatilite = rendements_portfolio.std() * np.sqrt(252)
    sharpe = (rendement_annualise - CONFIG['taux_sans_risque']) / volatilite
    
    cummax = valeur_portfolio.cummax()
    drawdown = (valeur_portfolio - cummax) / cummax
    max_drawdown = drawdown.min()
    
    turnover_moyen = np.mean(historique_turnover) if historique_turnover else 0
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
    """Affiche les métriques"""
    print(f"\n{'='*60}")
    print(f"📊 {nom_strategie}")
    print(f"{'='*60}")
    print(f"  Rendement total       : {metriques['rendement_total']*100:>8.2f}%")
    print(f"  Rendement annualisé   : {metriques['rendement_annualise']*100:>8.2f}%")
    print(f"  Volatilité annuelle   : {metriques['volatilite']*100:>8.2f}%")
    print(f"  Sharpe Ratio          : {metriques['sharpe']:>8.2f}")
    print(f"  Maximum Drawdown      : {metriques['max_drawdown']*100:>8.2f}%")
    print(f"  Turnover moyen        : {metriques['turnover_moyen']*100:>8.2f}%")
    print(f"  Coûts totaux          : ${metriques['couts_totaux']:>8,.0f}")
    print(f"  Coûts (% capital)     : {metriques['pct_couts']*100:>8.2f}%")


# ============================================================
# VISUALISATIONS
# ============================================================

def visualiser_comparaison_complete(resultats, prix):
    """Visualisation comparative complète - VERSION OPTIMISÉE"""
    
    # Palette de couleurs distinctes et professionnelles
    couleurs = {
        'Risk Parity Simple': '#2E86AB',          # Bleu
        'Risk Parity + Shrinkage': '#A23B72',     # Violet
        'HRP': '#F18F01',                          # Orange
        'HRP + Shrinkage': '#C73E1D',             # Rouge
        'Markowitz': '#6A994E',                    # Vert
        'Markowitz + Shrinkage': '#BC4B51'        # Bordeaux
    }
    
    n_strategies = len(resultats)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.25)
    
    # ===== GRAPHIQUE 1 : PERFORMANCE =====
    ax1 = fig.add_subplot(gs[0, :])
    
    for nom, data in resultats.items():
        ax1.plot(data['valeur'].index, data['valeur'].values, 
                label=nom, linewidth=2.2, alpha=0.85, color=couleurs.get(nom, None))
    
    ax1.set_title('Performance Comparée (Valeur du Portfolio)', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Valeur ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=8.5, ncol=3, framealpha=0.95)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ===== GRAPHIQUE 2 : DRAWDOWN =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    for nom, data in resultats.items():
        valeur = data['valeur']
        cummax = valeur.cummax()
        drawdown = (valeur - cummax) / cummax * 100
        ax2.plot(drawdown.index, drawdown.values, 
                label=nom, linewidth=1.8, alpha=0.85, color=couleurs.get(nom, None))
    
    ax2.set_title('Drawdown (%)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.legend(loc='lower left', fontsize=7.5, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ===== GRAPHIQUE 3 : VOLATILITÉ ROULANTE =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    for nom, data in resultats.items():
        rendements = data['valeur'].pct_change()
        vol_roulante = rendements.rolling(63).std() * np.sqrt(252) * 100
        ax3.plot(vol_roulante.index, vol_roulante.values, 
                label=nom, linewidth=1.8, alpha=0.85, color=couleurs.get(nom, None))
    
    ax3.set_title('Volatilité Roulante (63j)', fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Volatilité (%)', fontsize=10)
    ax3.legend(loc='upper left', fontsize=7.5, framealpha=0.95)
    ax3.grid(True, alpha=0.3)
    
    # ===== GRAPHIQUE 4 : SHARPE ROULANT =====
    ax4 = fig.add_subplot(gs[2, 0])
    
    for nom, data in resultats.items():
        rendements = data['valeur'].pct_change()
        sharpe_roll = (rendements.rolling(252).mean() * 252 - 0.02) / (rendements.rolling(252).std() * np.sqrt(252))
        ax4.plot(sharpe_roll.index, sharpe_roll.values, 
                label=nom, linewidth=1.8, alpha=0.85, color=couleurs.get(nom, None))
    
    ax4.set_title('Sharpe Ratio Roulant (252j)', fontsize=13, fontweight='bold', pad=12)
    ax4.set_xlabel('Date', fontsize=10)
    ax4.set_ylabel('Sharpe Ratio', fontsize=10)
    ax4.legend(loc='upper left', fontsize=7.5, framealpha=0.95)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # ===== GRAPHIQUE 5 : TURNOVER CUMULÉ =====
    ax5 = fig.add_subplot(gs[2, 1])
    
    for nom, data in resultats.items():
        m = data['metriques']
        n_rebal = (len(data['valeur']) - 252) // 21
        turnover_cumule = np.linspace(0, m['turnover_moyen'] * n_rebal * 100, len(data['valeur']))
        ax5.plot(data['valeur'].index, turnover_cumule, 
                label=nom, linewidth=1.8, alpha=0.85, color=couleurs.get(nom, None))
    
    ax5.set_title('Turnover Cumulé Estimé (%)', fontsize=13, fontweight='bold', pad=12)
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Turnover Cumulé (%)', fontsize=10)
    ax5.legend(loc='upper left', fontsize=7.5, framealpha=0.95)
    ax5.grid(True, alpha=0.3)
    
    # ===== GRAPHIQUE 6 : TABLEAU COMPARATIF =====
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('tight')
    ax6.axis('off')
    
    # Noms de métriques plus courts
    metriques_noms = [
        'Rdt Total',
        'Rdt Ann.',
        'Vol.',
        'Sharpe',
        'Max DD',
        'Turnover',
        'Coûts'
    ]
    
    # Noms de stratégies abrégés
    noms_courts = {
        'Risk Parity Simple': 'RP Simple',
        'Risk Parity + Shrinkage': 'RP + Shr',
        'HRP': 'HRP',
        'HRP + Shrinkage': 'HRP + Shr',
        'Markowitz': 'Markow.',
        'Markowitz + Shrinkage': 'Mark. + Shr'
    }
    
    donnees_tableau = []
    for nom in resultats.keys():
        m = resultats[nom]['metriques']
        donnees_tableau.append([
            f"{m['rendement_total']*100:.1f}%",
            f"{m['rendement_annualise']*100:.1f}%",
            f"{m['volatilite']*100:.1f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']*100:.1f}%",
            f"{m['turnover_moyen']*100:.1f}%",
            f"${m['couts_totaux']:,.0f}",
        ])
    
    donnees_tableau_t = list(map(list, zip(*donnees_tableau)))
    
    table = ax6.table(
        cellText=donnees_tableau_t,
        rowLabels=metriques_noms,
        colLabels=[noms_courts.get(nom, nom[:12]) for nom in resultats.keys()],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)
    
    # Style du tableau
    for i in range(len(resultats)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=8.5)
    
    for i in range(len(metriques_noms)):
        table[(i+1, -1)].set_facecolor('#ecf0f1')
        table[(i+1, -1)].set_text_props(weight='bold', fontsize=8.5)
    
    # Colorer les meilleures valeurs
    # Sharpe (ligne 4)
    sharpes = [resultats[nom]['metriques']['sharpe'] for nom in resultats.keys()]
    best_sharpe_idx = sharpes.index(max(sharpes))
    table[(4, best_sharpe_idx)].set_facecolor('#d5f4e6')
    table[(4, best_sharpe_idx)].set_text_props(weight='bold')
    
    # Coûts (ligne 7) - minimum est meilleur
    couts = [resultats[nom]['metriques']['couts_totaux'] for nom in resultats.keys()]
    best_cout_idx = couts.index(min(couts))
    table[(7, best_cout_idx)].set_facecolor('#d5f4e6')
    table[(7, best_cout_idx)].set_text_props(weight='bold')
    
    ax6.set_title('Tableau Comparatif des Métriques', fontsize=13, fontweight='bold', pad=15)
    
    # ===== GRAPHIQUE 7 : RENDEMENTS ANNUELS =====
    ax7 = fig.add_subplot(gs[4, :])
    
    # Calculer les années présentes dans les données
    premiere_date = min(data['valeur'].index[0] for data in resultats.values())
    derniere_date = max(data['valeur'].index[-1] for data in resultats.values())
    annees = range(premiere_date.year, derniere_date.year + 1)
    
    width = 0.13  # Largeur des barres
    x_base = np.arange(len(annees))
    
    for i, (nom, data) in enumerate(resultats.items()):
        rendements = data['valeur'].pct_change()
        rendements_annuels = rendements.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        
        values = []
        for year in annees:
            if year in rendements_annuels.index.year:
                idx = list(rendements_annuels.index.year).index(year)
                values.append(rendements_annuels.iloc[idx])
            else:
                values.append(0)
        
        x_pos = x_base + (i - len(resultats)/2 + 0.5) * width
        ax7.bar(x_pos, values, width=width, 
               label=noms_courts.get(nom, nom[:12]), 
               alpha=0.85, color=couleurs.get(nom, None))
    
    ax7.set_title('Rendements Annuels par Année', fontsize=13, fontweight='bold', pad=12)
    ax7.set_xlabel('Année', fontsize=11)
    ax7.set_ylabel('Rendement (%)', fontsize=11)
    ax7.set_xticks(x_base)
    ax7.set_xticklabels(annees, fontsize=10)
    ax7.legend(loc='upper left', fontsize=8, ncol=3, framealpha=0.95)
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Titre principal
    plt.suptitle('ANALYSE COMPARATIVE - Shrinkage + HRP', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    # Sauvegarder
    import os
    if os.path.exists('/mnt/user-data/outputs/'):
        chemin = '/mnt/user-data/outputs/shrinkage_hrp_comparison.png'
    else:
        chemin = 'shrinkage_hrp_comparison.png'
    
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé : {chemin}")
    
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🚀 LANCEMENT DU BACKTEST COMPARATIF AVANCÉ")
    print("="*70)
    
    # Téléchargement
    prix = telecharger_donnees(
        CONFIG['tickers'],
        CONFIG['date_debut'],
        CONFIG['date_fin']
    )
    
    # Liste des stratégies à tester
    strategies_a_tester = [
        ('rp_simple', False, 'Risk Parity Simple'),
        ('rp_simple', True, 'Risk Parity + Shrinkage'),
        ('hrp', False, 'HRP'),
        ('hrp', True, 'HRP + Shrinkage'),
        ('markowitz', False, 'Markowitz'),
        ('markowitz', True, 'Markowitz + Shrinkage'),
    ]
    
    resultats = {}
    
    for strategie, use_shrink, nom in strategies_a_tester:
        print(f"\n{'─'*70}")
        print(f"STRATÉGIE : {nom}")
        print(f"{'─'*70}")
        
        val, poids, met = backtest_strategie(
            prix,
            CONFIG['fenetre_estimation'],
            CONFIG['frequence_rebalancement'],
            strategie=strategie,
            use_shrinkage=use_shrink
        )
        
        afficher_metriques(nom, met)
        
        resultats[nom] = {
            'valeur': val,
            'poids': poids,
            'metriques': met
        }
    
    # Visualisations
    print(f"\n{'─'*70}")
    print("VISUALISATIONS COMPARATIVES")
    print(f"{'─'*70}")
    
    visualiser_comparaison_complete(resultats, prix)
    
    # Résumé final
    print(f"\n{'='*70}")
    print("✅ ANALYSE TERMINÉE")
    print(f"{'='*70}")
    
    print("\n🏆 CLASSEMENT PAR SHARPE RATIO:")
    sharpes = {nom: data['metriques']['sharpe'] for nom, data in resultats.items()}
    for i, (nom, sharpe) in enumerate(sorted(sharpes.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {nom:<30} : {sharpe:.4f}")
    
    print("\n💰 CLASSEMENT PAR COÛTS (meilleur = moins cher):")
    couts = {nom: data['metriques']['couts_totaux'] for nom, data in resultats.items()}
    for i, (nom, cout) in enumerate(sorted(couts.items(), key=lambda x: x[1]), 1):
        print(f"  {i}. {nom:<30} : ${cout:,.0f}")
    
    print("\n🛡️  CLASSEMENT PAR DRAWDOWN (meilleur = moins négatif):")
    drawdowns = {nom: data['metriques']['max_drawdown'] for nom, data in resultats.items()}
    for i, (nom, dd) in enumerate(sorted(drawdowns.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {nom:<30} : {dd*100:.2f}%")
    
    print("\n🔄 CLASSEMENT PAR TURNOVER (meilleur = plus faible):")
    turnovers = {nom: data['metriques']['turnover_moyen'] for nom, data in resultats.items()}
    for i, (nom, turn) in enumerate(sorted(turnovers.items(), key=lambda x: x[1]), 1):
        print(f"  {i}. {nom:<30} : {turn*100:.2f}%")
    
    print("\n" + "="*70)
    print()


if __name__ == "__main__":
    main()
