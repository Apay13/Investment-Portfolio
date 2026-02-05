"""
RISK PARITY - ALLOCATION PAR CONTRIBUTION AU RISQUE
====================================================

Risk Parity alloue les actifs de manière à ce que chaque actif 
contribue ÉGALEMENT au risque total du portfolio.

Principe : Les actifs moins volatils reçoivent plus de poids
          Les actifs plus volatils reçoivent moins de poids

Avantages vs Markowitz :
- Plus stable (poids changent moins)
- Moins de turnover
- Plus robuste aux erreurs d'estimation
- Meilleure diversification du risque
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
# CONFIGURATION
# ============================================================

CONFIG = {
    # Paramètres du portfolio
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'GS', 'JNJ', 'PFE'],
    
    # Période de backtest
    'date_debut': '2018-01-01',
    'date_fin': '2025-12-31',
    
    # Paramètres de la fenêtre roulante
    'fenetre_estimation': 252,
    'frequence_rebalancement': 21,
    
    # Coûts de transaction
    'cout_transaction': 0.001,
    
    # Paramètres financiers
    'taux_sans_risque': 0.02,
    'capital_initial': 100000,
    
    # Vol targeting (optionnel)
    'vol_target': 0.20,  # Cible de volatilité à 20%
    'use_vol_targeting': True,
    
    # Graphiques
    'style_graphique': 'seaborn-v0_8-darkgrid',
    'taille_figure': (16, 10),
}

plt.style.use(CONFIG['style_graphique'])

print("="*70)
print("🎯 RISK PARITY - ALLOCATION PAR CONTRIBUTION AU RISQUE")
print("="*70)
print(f"\nConfiguration:")
print(f"  • Actifs : {', '.join(CONFIG['tickers'])}")
print(f"  • Période : {CONFIG['date_debut']} à {CONFIG['date_fin']}")
print(f"  • Vol Targeting : {'Activé' if CONFIG['use_vol_targeting'] else 'Désactivé'}")
if CONFIG['use_vol_targeting']:
    print(f"  • Volatilité cible : {CONFIG['vol_target']*100}%")
print()


# ============================================================
# FONCTIONS UTILITAIRES (réutilisées)
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
# RISK PARITY - ALGORITHMES
# ============================================================

def risk_parity_simple(volatilites):
    """
    Risk Parity Simple : Inverse Volatility Weighting
    
    Formule : w(i) = (1/σ(i)) / Σ(1/σ(j))
    
    Principe : Plus un actif est volatil, moins on en prend
    
    Paramètres :
    -----------
    volatilites : array
        Volatilités annualisées de chaque actif
    
    Retour :
    --------
    array : Poids optimaux
    """
    # Inverse des volatilités
    inverse_vol = 1 / volatilites
    
    # Normaliser pour que la somme = 1
    poids = inverse_vol / np.sum(inverse_vol)
    
    return poids


def risk_contribution(poids, matrice_cov):
    """
    Calcule la contribution de chaque actif au risque total
    
    RC(i) = w(i) × (Σ × w)(i) / σ(portfolio)
    
    où :
    - w = vecteur des poids
    - Σ = matrice de covariance
    - σ(portfolio) = volatilité totale du portfolio
    """
    # Volatilité du portfolio
    portfolio_vol = np.sqrt(np.dot(poids.T, np.dot(matrice_cov, poids)))
    
    # Contribution marginale au risque
    marginal_contrib = np.dot(matrice_cov, poids)
    
    # Contribution au risque
    risk_contrib = poids * marginal_contrib / portfolio_vol
    
    return risk_contrib


def objectif_risk_parity(poids, matrice_cov):
    """
    Fonction objectif pour Risk Parity
    
    Minimise la différence entre les contributions au risque
    On veut que tous les actifs contribuent également
    """
    # Calculer les contributions au risque
    rc = risk_contribution(poids, matrice_cov)
    
    # Contribution cible (égale pour tous)
    rc_cible = np.ones(len(poids)) / len(poids)
    
    # Minimiser la somme des carrés des différences
    return np.sum((rc - rc_cible) ** 2)


def risk_parity_optimise(matrice_cov):
    """
    Risk Parity Optimisé
    
    Trouve les poids qui égalisent les contributions au risque
    via optimisation numérique
    
    Paramètres :
    -----------
    matrice_cov : DataFrame
        Matrice de covariance
    
    Retour :
    --------
    array : Poids optimaux
    """
    n_actifs = len(matrice_cov)
    
    # Contrainte : somme = 1
    contraintes = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Bornes : poids positifs uniquement
    bornes = tuple((0.01, 1) for _ in range(n_actifs))
    
    # Point de départ : équipondération
    poids_init = np.array([1/n_actifs] * n_actifs)
    
    # Optimisation
    resultat = minimize(
        objectif_risk_parity,
        poids_init,
        args=(matrice_cov,),
        method='SLSQP',
        bounds=bornes,
        constraints=contraintes
    )
    
    if resultat.success:
        return resultat.x
    else:
        # Fallback sur inverse volatility
        volatilites = np.sqrt(np.diag(matrice_cov))
        return risk_parity_simple(volatilites)


# ============================================================
# BACKTEST RISK PARITY
# ============================================================

def backtest_risk_parity(prix, fenetre, frequence_rebal, method='optimise'):
    """
    Backtest de Risk Parity avec rolling window
    
    Paramètres :
    -----------
    method : str
        'simple' = Inverse Volatility
        'optimise' = Equal Risk Contribution
    """
    print(f"\n🎯 Backtest Risk Parity ({method})...")
    print(f"   Fenêtre d'estimation : {fenetre} jours")
    print(f"   Rebalancement tous les : {frequence_rebal} jours")
    
    n_actifs = len(prix.columns)
    rendements = calculer_rendements(prix)
    
    # Initialisation
    valeur_portfolio = pd.Series(index=rendements.index, dtype=float)
    valeur_portfolio.iloc[0] = CONFIG['capital_initial']
    
    historique_poids = []
    historique_turnover = []
    historique_couts = []
    historique_leverage = []
    
    # Poids initiaux
    poids_actuels = np.array([1/n_actifs] * n_actifs)
    
    jours_depuis_rebal = 0
    n_rebalancements = 0
    
    for i in range(1, len(rendements)):
        
        rdt_jour = rendements.iloc[i].values
        
        # Mise à jour de la valeur
        rendement_portfolio = np.sum(poids_actuels * rdt_jour)
        valeur_portfolio.iloc[i] = valeur_portfolio.iloc[i-1] * (1 + rendement_portfolio)
        
        # Drift des poids
        poids_actuels = poids_actuels * (1 + rdt_jour)
        poids_actuels = poids_actuels / np.sum(poids_actuels)
        
        jours_depuis_rebal += 1
        
        # REBALANCEMENT
        if jours_depuis_rebal >= frequence_rebal and i >= fenetre:
            
            rendements_fenetre = rendements.iloc[i-fenetre:i]
            matrice_cov = rendements_fenetre.cov() * 252
            
            # Calculer les poids Risk Parity
            if method == 'simple':
                volatilites = rendements_fenetre.std() * np.sqrt(252)
                poids_optimaux = risk_parity_simple(volatilites.values)
            else:
                poids_optimaux = risk_parity_optimise(matrice_cov)
            
            # Vol Targeting (optionnel)
            leverage = 1.0
            if CONFIG['use_vol_targeting']:
                rendements_moyens = rendements_fenetre.mean() * 252
                _, portfolio_vol = calculer_performance_portfolio(
                    poids_optimaux, rendements_moyens, matrice_cov
                )
                leverage = CONFIG['vol_target'] / portfolio_vol
                leverage = np.clip(leverage, 0.5, 2.0)  # Limiter le leverage
            
            historique_leverage.append(leverage)
            
            # Turnover et coûts
            turnover = np.sum(np.abs(poids_optimaux - poids_actuels))
            cout = turnover * CONFIG['cout_transaction'] * valeur_portfolio.iloc[i]
            valeur_portfolio.iloc[i] -= cout
            
            historique_poids.append({
                'date': rendements.index[i],
                **{f'poids_{ticker}': poids_optimaux[j] for j, ticker in enumerate(prix.columns)},
                'leverage': leverage
            })
            historique_turnover.append(turnover)
            historique_couts.append(cout)
            
            poids_actuels = poids_optimaux.copy()
            jours_depuis_rebal = 0
            n_rebalancements += 1
            
            if n_rebalancements % 10 == 0:
                print(f"   → {n_rebalancements} rebalancements effectués...")
    
    print(f"✓ Backtest terminé : {n_rebalancements} rebalancements")
    
    df_poids = pd.DataFrame(historique_poids)
    if len(df_poids) > 0:
        df_poids.set_index('date', inplace=True)
    
    metriques = calculer_metriques_performance(
        valeur_portfolio, historique_turnover, historique_couts
    )
    metriques['leverage_moyen'] = np.mean(historique_leverage) if historique_leverage else 1.0
    
    return valeur_portfolio, df_poids, metriques


# ============================================================
# MARKOWITZ (pour comparaison)
# ============================================================

def sharpe_negatif(poids, rendements_moyens, matrice_cov, rf=0.02):
    """Ratio de Sharpe négatif pour minimisation"""
    r, v = calculer_performance_portfolio(poids, rendements_moyens, matrice_cov)
    return -(r - rf) / v


def backtest_markowitz_rolling(prix, fenetre, frequence_rebal):
    """Backtest Markowitz (copié du script précédent)"""
    print(f"\n📊 Backtest Markowitz...")
    
    n_actifs = len(prix.columns)
    rendements = calculer_rendements(prix)
    
    valeur_portfolio = pd.Series(index=rendements.index, dtype=float)
    valeur_portfolio.iloc[0] = CONFIG['capital_initial']
    
    historique_turnover = []
    historique_couts = []
    
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
        
        if jours_depuis_rebal >= frequence_rebal and i >= fenetre:
            
            rendements_fenetre = rendements.iloc[i-fenetre:i]
            rendements_moyens = rendements_fenetre.mean() * 252
            matrice_cov = rendements_fenetre.cov() * 252
            
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
            
            turnover = np.sum(np.abs(poids_optimaux - poids_actuels))
            cout = turnover * CONFIG['cout_transaction'] * valeur_portfolio.iloc[i]
            valeur_portfolio.iloc[i] -= cout
            
            historique_turnover.append(turnover)
            historique_couts.append(cout)
            
            poids_actuels = poids_optimaux.copy()
            jours_depuis_rebal = 0
            n_rebalancements += 1
            
            if n_rebalancements % 10 == 0:
                print(f"   → {n_rebalancements} rebalancements...")
    
    print(f"✓ Backtest terminé : {n_rebalancements} rebalancements")
    
    metriques = calculer_metriques_performance(
        valeur_portfolio, historique_turnover, historique_couts
    )
    
    return valeur_portfolio, metriques


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
    if 'leverage_moyen' in metriques:
        print(f"  Leverage moyen        : {metriques['leverage_moyen']:>8.2f}x")


# ============================================================
# VISUALISATIONS
# ============================================================

def visualiser_comparaison(resultats, prix):
    """Visualisation comparative Risk Parity vs Markowitz"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # ===== GRAPHIQUE 1 : PERFORMANCE =====
    ax1 = fig.add_subplot(gs[0, :])
    
    for nom, data in resultats.items():
        ax1.plot(data['valeur'].index, data['valeur'].values, label=nom, linewidth=2.5)
    
    ax1.set_title('Performance Comparée (Valeur du Portfolio)', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Valeur ($)', fontsize=12)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # ===== GRAPHIQUE 2 : DRAWDOWN =====
    ax2 = fig.add_subplot(gs[1, 0])
    
    for nom, data in resultats.items():
        valeur = data['valeur']
        cummax = valeur.cummax()
        drawdown = (valeur - cummax) / cummax * 100
        ax2.plot(drawdown.index, drawdown.values, label=nom, linewidth=2)
    
    ax2.set_title('Drawdown (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # ===== GRAPHIQUE 3 : VOLATILITÉ ROULANTE =====
    ax3 = fig.add_subplot(gs[1, 1])
    
    for nom, data in resultats.items():
        rendements = data['valeur'].pct_change()
        vol_roulante = rendements.rolling(63).std() * np.sqrt(252) * 100
        ax3.plot(vol_roulante.index, vol_roulante.values, label=nom, linewidth=2)
    
    ax3.set_title('Volatilité Roulante (63j)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volatilité (%)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # ===== GRAPHIQUE 4 : TABLEAU COMPARATIF =====
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
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
    for nom in resultats.keys():
        m = resultats[nom]['metriques']
        donnees_tableau.append([
            f"{m['rendement_total']*100:.2f}%",
            f"{m['rendement_annualise']*100:.2f}%",
            f"{m['volatilite']*100:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']*100:.2f}%",
            f"{m['turnover_moyen']*100:.2f}%",
            f"${m['couts_totaux']:,.0f}",
        ])
    
    donnees_tableau_t = list(map(list, zip(*donnees_tableau)))
    
    table = ax4.table(
        cellText=donnees_tableau_t,
        rowLabels=metriques_noms,
        colLabels=list(resultats.keys()),
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(resultats)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(metriques_noms)):
        table[(i+1, -1)].set_facecolor('#d9d9d9')
        table[(i+1, -1)].set_text_props(weight='bold')
    
    ax4.set_title('Comparaison des Métriques', fontsize=13, fontweight='bold', pad=20)
    
    # ===== GRAPHIQUE 5 : RENDEMENTS ANNUELS =====
    ax5 = fig.add_subplot(gs[3, :])
    
    for nom, data in resultats.items():
        rendements = data['valeur'].pct_change()
        rendements_annuels = rendements.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
        ax5.bar(rendements_annuels.index.year + list(resultats.keys()).index(nom)*0.2 - 0.2,
                rendements_annuels.values, width=0.2, label=nom, alpha=0.8)
    
    ax5.set_title('Rendements Annuels (%)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Année')
    ax5.set_ylabel('Rendement (%)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('RISK PARITY vs MARKOWITZ - Analyse Comparative', 
                 fontsize=17, fontweight='bold', y=0.995)
    
    import os
    if os.path.exists('/mnt/user-data/outputs/'):
        chemin = '/mnt/user-data/outputs/risk_parity_vs_markowitz.png'
    else:
        chemin = 'risk_parity_vs_markowitz.png'
    
    plt.savefig(chemin, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphique sauvegardé : {chemin}")
    
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    """Fonction principale"""
    
    print("\n" + "="*70)
    print("🚀 LANCEMENT DU BACKTEST COMPARATIF")
    print("="*70)
    
    # Téléchargement
    prix = telecharger_donnees(
        CONFIG['tickers'],
        CONFIG['date_debut'],
        CONFIG['date_fin']
    )
    
    # Backtest Risk Parity Simple
    print(f"\n{'─'*70}")
    print("STRATÉGIE 1 : Risk Parity (Inverse Volatility)")
    print(f"{'─'*70}")
    
    val_rp_simple, poids_rp_simple, met_rp_simple = backtest_risk_parity(
        prix,
        CONFIG['fenetre_estimation'],
        CONFIG['frequence_rebalancement'],
        method='simple'
    )
    afficher_metriques("RISK PARITY (Simple)", met_rp_simple)
    
    # Backtest Risk Parity Optimisé
    print(f"\n{'─'*70}")
    print("STRATÉGIE 2 : Risk Parity (Equal Risk Contribution)")
    print(f"{'─'*70}")
    
    val_rp_opt, poids_rp_opt, met_rp_opt = backtest_risk_parity(
        prix,
        CONFIG['fenetre_estimation'],
        CONFIG['frequence_rebalancement'],
        method='optimise'
    )
    afficher_metriques("RISK PARITY (Optimisé)", met_rp_opt)
    
    # Backtest Markowitz
    print(f"\n{'─'*70}")
    print("STRATÉGIE 3 : Markowitz (Max Sharpe)")
    print(f"{'─'*70}")
    
    val_markowitz, met_markowitz = backtest_markowitz_rolling(
        prix,
        CONFIG['fenetre_estimation'],
        CONFIG['frequence_rebalancement']
    )
    afficher_metriques("MARKOWITZ", met_markowitz)
    
    # Visualisations
    print(f"\n{'─'*70}")
    print("VISUALISATIONS COMPARATIVES")
    print(f"{'─'*70}")
    
    resultats = {
        'Risk Parity (Simple)': {
            'valeur': val_rp_simple,
            'metriques': met_rp_simple
        },
        'Risk Parity (Optimisé)': {
            'valeur': val_rp_opt,
            'metriques': met_rp_opt
        },
        'Markowitz': {
            'valeur': val_markowitz,
            'metriques': met_markowitz
        }
    }
    
    visualiser_comparaison(resultats, prix)
    
    # Résumé final
    print(f"\n{'='*70}")
    print("✅ ANALYSE TERMINÉE")
    print(f"{'='*70}")
    
    print("\n🏆 CLASSEMENT PAR SHARPE RATIO:")
    sharpes = {nom: data['metriques']['sharpe'] for nom, data in resultats.items()}
    for i, (nom, sharpe) in enumerate(sorted(sharpes.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {nom:<30} : {sharpe:.4f}")
    
    print("\n🛡️  CLASSEMENT PAR DRAWDOWN (meilleur = moins négatif):")
    drawdowns = {nom: data['metriques']['max_drawdown'] for nom, data in resultats.items()}
    for i, (nom, dd) in enumerate(sorted(drawdowns.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  {i}. {nom:<30} : {dd*100:.2f}%")
    
    print("\n💰 CLASSEMENT PAR COÛTS (meilleur = moins cher):")
    couts = {nom: data['metriques']['couts_totaux'] for nom, data in resultats.items()}
    for i, (nom, cout) in enumerate(sorted(couts.items(), key=lambda x: x[1]), 1):
        print(f"  {i}. {nom:<30} : ${cout:,.0f}")
    
    print("\n" + "="*70)
    print()


if __name__ == "__main__":
    main()
