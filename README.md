# 🔬 DOCUMENTATION - Strategy Backtest

## 🎯 Vue d'ensemble

Ce script implémente des stratégies d'allocation **avancées** qui améliorent Markowitz en réduisant l'instabilité et les coûts de transaction.

**Fichier :** `shrinkage_hrp.py`

**Stratégies testées :**
1. Risk Parity Simple (Inverse Volatility)
2. Risk Parity + Shrinkage Ledoit-Wolf
3. Hierarchical Risk Parity (HRP)
4. HRP + Shrinkage
5. Markowitz Standard
6. Markowitz + Shrinkage

---

## 📋 Table des matières

1. [Architecture](#architecture)
2. [Configuration](#configuration)
3. [Théorie des stratégies](#theorie)
4. [Blocs de code détaillés](#blocs)
5. [Mathématiques](#maths)
6. [Interprétation résultats](#interpretation)

---

## 🏗️ Architecture {#architecture}

```
shrinkage_hrp.py
├── IMPORTS & CONFIG (lignes 1-60)
├── UTILITIES (lignes 61-150)
├── SHRINKAGE LEDOIT-WOLF (lignes 151-200)
├── HRP - HIERARCHICAL RISK PARITY (lignes 201-350)
├── STRATÉGIES CLASSIQUES (lignes 351-450)
├── BACKTEST GÉNÉRIQUE (lignes 451-600)
├── MÉTRIQUES (lignes 601-700)
├── VISUALISATIONS (lignes 701-950)
└── MAIN (lignes 951-1100)
```

---

## ⚙️ Configuration {#configuration}

```python
CONFIG = {
    # Portfolio
    'tickers': ['AAPL', 'MSFT', 'GOOGL', ...],
    'date_debut': '2018-01-01',
    'date_fin': '2025-12-31',
    
    # Backtest
    'fenetre_estimation': 252,      # Rolling window
    'frequence_rebalancement': 21,  # Mensuel
    'cout_transaction': 0.001,      # 0.1%
    
    # Finance
    'taux_sans_risque': 0.02,
    'capital_initial': 100000,
}
```

---

## 📚 Théorie des stratégies {#theorie}

### **1. Risk Parity Simple (Inverse Volatility)**

**Principe :**
```
Plus un actif est volatile, moins on en prend
```

**Formule :**
```python
w(i) = (1 / σ(i)) / Σ(1 / σ(j))
```

**Avantages :**
- ✅ Ultra simple
- ✅ Très stable (turnover ~5%)
- ✅ Coûts faibles
- ✅ Robuste

**Inconvénients :**
- ⚠️ Ignore les rendements
- ⚠️ Ignore les corrélations

---

### **2. Shrinkage Ledoit-Wolf**

**Problème résolu :**
```
Matrice de covariance empirique = BRUITÉE

Exemple :
Cov(AAPL, MSFT) estimée = 0.045
Vraie covariance          = 0.040
Erreur                    = +12%
```

**Solution :**
```
Σ_shrunk = δ × F + (1-δ) × S

où :
S = matrice empirique (bruitée)
F = matrice cible (structurée)
δ = intensité de shrinkage (auto-optimisée)
```

**Quand utile ?**
- Beaucoup d'actifs (50+)
- Peu de données (<100 jours)
- Matrice très bruitée

**Constat de nos tests :**
```
δ moyen = 0.001 → QUASI NUL !
Raison : 10 actifs + 252 jours = données suffisantes
```

---

### **3. Hierarchical Risk Parity (HRP)**

**Principe :**
```
Allouer en 2 temps :
1. Entre CLUSTERS (Tech, Finance, Santé)
2. Dans CHAQUE cluster
```

**Algorithme (Marcos López de Prado, 2016) :**

**ÉTAPE 1 : Clustering**
```python
Distance(i,j) = √(0.5 × (1 - Corr(i,j)))
Clustering hiérarchique → Dendrogram
```

**ÉTAPE 2 : Quasi-diagonalisation**
```
Réorganiser la matrice de covariance pour regrouper
les actifs similaires ensemble
```

**ÉTAPE 3 : Allocation récursive**
```
Fonction récursive :
1. Diviser cluster en 2 sous-clusters
2. Allouer entre les 2 (inverse variance)
3. Répéter sur chaque sous-cluster
```

**Exemple concret :**
```
Portfolio [AAPL, MSFT, GOOGL, JPM, BAC, JNJ]

Clustering :
├─ Tech [AAPL, MSFT, GOOGL]
├─ Finance [JPM, BAC]
└─ Santé [JNJ]

Allocation niveau 1 :
Tech    : 40%
Finance : 35%
Santé   : 25%

Allocation niveau 2 :
AAPL  : 40% × 0.33 = 13%
MSFT  : 40% × 0.33 = 13%
GOOGL : 40% × 0.34 = 14%
JPM   : 35% × 0.55 = 19%
BAC   : 35% × 0.45 = 16%
JNJ   : 25% × 1.00 = 25%
```

**Avantages :**
- ✅ Diversification VRAIE (sectorielle)
- ✅ Stable (turnover ~15%)
- ✅ Meilleur drawdown en crise
- ✅ Pas de matrice à inverser

**Inconvénients :**
- ⚠️ Complexité algorithmique
- ⚠️ Turnover > Risk Parity Simple
- ⚠️ Coûts 3× plus élevés

---

## 🔧 Blocs de code détaillés {#blocs}

### **BLOC 1 : Shrinkage Ledoit-Wolf**

```python
def ledoit_wolf_shrinkage(rendements):
```

**Implémentation :**

**1. Matrice empirique**
```python
S = np.cov(rendements.T, bias=True)
```

**2. Matrice cible (constant correlation)**
```python
var_mean = np.trace(S) / n_actifs
corr_mean = (np.sum(S) - np.trace(S)) / (n × (n-1))
F = corr_mean × ones((n,n))
F[diagonal] = var_mean
```

**3. Intensité de shrinkage**
```python
diff = S - F
delta = min(1, max(0, np.sum(diff²) / (n_obs × np.sum(S²))))
```

**4. Matrice shrunk**
```python
S_shrunk = delta × F + (1 - delta) × S
```

**Retour :**
```python
return S_shrunk, delta
```

---

### **BLOC 2 : HRP - Partie A (Clustering)**

```python
def get_quasi_diag(link):
```

**Ce qu'il fait :**
1. Prend la sortie de `scipy.cluster.hierarchy.linkage`
2. Extrait l'ordre optimal des actifs
3. Retourne indices pour réorganiser la matrice

**Algorithme récursif :**
```python
Tant que (il reste des clusters fusionnés):
    Remplacer cluster_id par ses 2 enfants
    Trier par index
```

---

### **BLOC 3 : HRP - Partie B (Allocation)**

```python
def hrp_allocation(rendements, matrice_cov):
```

**Étapes :**

**1. Clustering**
```python
corr = rendements.corr()
dist = np.sqrt(0.5 × (1 - corr))
link = linkage(squareform(dist), method='single')
```

**2. Réorganisation**
```python
sort_ix = get_quasi_diag(link)
cov_sorted = matrice_cov.loc[sort_ix, sort_ix]
```

**3. Allocation récursive**
```python
weights = Series(1.0)  # Tous à 100% au départ
clusters = [all_assets]

while len(clusters) > 0:
    for cluster in clusters:
        # Split en 2
        left = cluster[:len//2]
        right = cluster[len//2:]
        
        # Variance de chaque moitié
        var_left = cluster_variance(left)
        var_right = cluster_variance(right)
        
        # Allouer inverse variance
        alpha = 1 - var_left / (var_left + var_right)
        
        weights[left] *= alpha
        weights[right] *= (1 - alpha)
```

---

### **BLOC 4 : Backtest générique**

```python
def backtest_strategie(prix, fenetre, frequence_rebal, 
                       strategie='rp_simple', use_shrinkage=False):
```

**Architecture :**

**1. Initialisation**
```python
valeur_portfolio = Series(index=rendements.index)
valeur_portfolio[0] = CAPITAL_INITIAL
poids_actuels = array([1/n] * n)
```

**2. Boucle quotidienne**
```python
for jour in range(1, len(rendements)):
    # Rendement quotidien
    rdt_jour = rendements.iloc[jour]
    valeur_portfolio[jour] = valeur_portfolio[jour-1] × (1 + rdt_pf)
    
    # Drift naturel des poids
    poids_actuels = poids_actuels × (1 + rdt_jour)
    poids_actuels /= sum(poids_actuels)  # Re-normaliser
    
    jours_depuis_rebal += 1
```

**3. Rebalancement (tous les 21 jours)**
```python
if jours_depuis_rebal >= 21 and jour >= 252:
    # Fenêtre roulante
    rendements_fenetre = rendements[jour-252:jour]
    
    # Covariance (avec ou sans shrinkage)
    if use_shrinkage:
        matrice_cov, delta = ledoit_wolf_shrinkage(...)
    else:
        matrice_cov = rendements_fenetre.cov() × 252
    
    # Calculer nouveaux poids
    if strategie == 'rp_simple':
        poids = inverse_volatility(...)
    elif strategie == 'hrp':
        poids = hrp_allocation(...)
    elif strategie == 'markowitz':
        poids = optimize_sharpe(...)
    
    # Turnover et coûts
    turnover = sum(|poids_nouveaux - poids_actuels|)
    cout = turnover × 0.001 × valeur_portfolio[jour]
    valeur_portfolio[jour] -= cout
    
    # Appliquer
    poids_actuels = poids_nouveaux
    jours_depuis_rebal = 0
```

**Particularité importante :**
```python
# PAS de look-ahead bias !
# On utilise UNIQUEMENT les données jusqu'à jour-1
rendements_fenetre = rendements[jour-252:jour]
```

---

### **BLOC 5 : Métriques de performance**

```python
def calculer_metriques_performance(...):
```

**Métriques calculées :**

**1. Rendement total et annualisé**
```python
rendement_total = (valeur_finale / valeur_initiale) - 1
n_annees = n_jours / 252
rendement_annualise = (1 + rendement_total) ** (1/n_annees) - 1
```

**2. Volatilité**
```python
rendements_quotidiens = valeur_portfolio.pct_change()
volatilite = rendements_quotidiens.std() × √252
```

**3. Sharpe Ratio**
```python
sharpe = (rendement_annualise - 0.02) / volatilite
```

**4. Maximum Drawdown**
```python
cummax = valeur_portfolio.cummax()
drawdown = (valeur_portfolio - cummax) / cummax
max_drawdown = drawdown.min()
```

**5. Turnover moyen**
```python
turnover_moyen = mean(historique_turnover)
```

**6. Coûts totaux**
```python
couts_totaux = sum(historique_couts)
pct_couts = couts_totaux / capital_initial
```

---

### **BLOC 6 : Visualisations**

```python
def visualiser_comparaison_complete(resultats, prix):
```

**7 graphiques générés :**

**1. Performance comparée**
- Courbe de la valeur du portfolio
- Une couleur distincte par stratégie
- Légende en 3 colonnes

**2. Drawdown**
- % de perte depuis le pic
- Montre la souffrance en crise

**3. Volatilité roulante (63 jours)**
- Montre si le risque est stable
- Détecte les périodes volatiles

**4. Sharpe roulant (252 jours)**
- Performance ajustée au risque dans le temps
- Montre quelle stratégie est robuste

**5. Turnover cumulé**
- Volume de trading total
- Visualise le coût du rebalancement

**6. Tableau comparatif**
- Toutes les métriques
- Noms abrégés pour lisibilité
- Highlight des meilleurs (vert)

**7. Rendements annuels**
- Barres par année
- Montre quelle stratégie domine quand

---

## 📐 Mathématiques {#maths}

### **1. Distance pour clustering**

```
D(i,j) = √(0.5 × (1 - ρ(i,j)))

où ρ = corrélation
```

**Propriétés :**
- ρ = 1 → D = 0 (très similaires)
- ρ = 0 → D = 0.707
- ρ = -1 → D = 1 (opposés)

---

### **2. Variance d'un cluster**

```
Var(cluster) = w_cluster^T × Σ_cluster × w_cluster

où w_cluster = poids inverse variance dans le cluster
```

---

### **3. Allocation entre 2 clusters**

```
α = 1 - Var(left) / (Var(left) + Var(right))

w(left) = α
w(right) = 1 - α
```

**Interprétation :**
- Si Var(left) < Var(right) → α > 0.5 → plus de poids à gauche

---

### **4. Inverse Volatility Weighting**

```
w(i) = (1/σ(i)) / Σ_j(1/σ(j))

Normalisation :
Σ_i w(i) = 1
```

---

## 📊 Interprétation des résultats {#interpretation}

### **Classement Sharpe Ratio**

```
> 1.0  : Excellent (2012-2015 bull market)
0.7-1.0: Bon      (2016-2020 avec COVID)
0.5-0.7: Moyen    (Période volatile)
< 0.5  : Médiocre
```

---

### **Classement Drawdown**

```
< -20% : Faible   (Très bon)
-20 à -30% : Modéré  (Acceptable)
-30 à -40% : Élevé   (Douloureux)
> -40% : Sévère  (Catastrophique)
```

---

### **Classement Turnover**

```
< 10%  : Ultra faible (RP Simple)
10-20% : Faible       (HRP)
20-40% : Modéré       (RP Optimisé)
> 40%  : Élevé        (Markowitz)
```

---

### **Classement Coûts (sur 7 ans, $100k)**

```
< $1,000   : Excellent (RP Simple)
$1-3k      : Bon       (HRP)
$3-5k      : Moyen
> $5k      : Élevé     (Markowitz)
```

---

## 🎯 Guide de décision

### **Choisir Risk Parity Simple si :**
- ✅ Tu veux la simplicité
- ✅ Tu veux minimiser les coûts
- ✅ Tu acceptes drawdown moyen
- ✅ Tu veux robustesse

### **Choisir HRP si :**
- ✅ Tu veux meilleur drawdown
- ✅ Tu comprends le clustering
- ✅ Tu acceptes coûts 3× plus élevés
- ✅ Tu trades sur crises

### **Choisir Markowitz si :**
- ✅ Tu veux rendement maximum
- ✅ Tu acceptes turnover fou
- ✅ Tu acceptes coûts élevés
- ✅ Tu es sophistiqué

---

## ⚠️ Pièges à éviter

### **1. Shrinkage inutile dans nos tests**
```
δ = 0.001 → Aucun effet
Raison : 10 actifs + 252 jours = suffisant

Si tu veux voir shrinkage fonctionner :
- Utilise 50+ actifs
- OU réduis fenêtre à 100 jours
```

### **2. HRP sensible au nombre d'actifs**
```
Minimum : 6 actifs
Optimal : 10-20 actifs
Maximum : ~50 actifs

Trop peu → Pas de clusters intéressants
Trop → Clustering confus
```

### **3. Fenêtre d'estimation**
```
Trop courte (<100j) → Bruit
Trop longue (>500j) → Pas adaptatif

Sweet spot : 252 jours (1 an)
```

---

## 🔬 Résultats empiriques (nos tests)

### **Synthèse 4 périodes testées :**

| Stratégie | Sharpe moyen | Drawdown moyen | Coûts moyens | Turnover |
|-----------|--------------|----------------|--------------|----------|
| **RP Simple** | **0.84** 🥇 | -27% | **$640** 🥇 | **5%** 🥇 |
| **HRP** | 0.81 | **-25%** 🥇 | $1,900 | 15% |
| **Markowitz** | 0.73 | -31% | $5,800 | 43% |

**Verdict :**
```
Risk Parity Simple = GAGNANT GÉNÉRAL
- Meilleur Sharpe
- Coûts ridicules
- Ultra stable
```

---

## 🚀 Utilisation

### **Test standard**

```bash
python shrinkage_hrp.py
```

### **Changer période**

```python
CONFIG = {
    'date_debut': '2020-01-01',
    'date_fin': '2025-12-31',
}
```

### **Changer portfolio**

```python
CONFIG = {
    'tickers': ['SPY', 'TLT', 'GLD', 'VNQ', 'IEF'],
}
```

---

## 📚 Références académiques

**Risk Parity :**
- Qian, E. (2005). "Risk Parity Portfolios"
- Asness et al. (2012). "Leverage Aversion and Risk Parity"

**HRP :**
- López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample"

**Shrinkage :**
- Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"

---

## 🐛 Troubleshooting

**Problème : HRP IndexError**
```
Solution : Vérifier que get_quasi_diag() retourne des entiers
```

**Problème : Shrinkage δ = 0**
```
Normal si peu d'actifs + beaucoup de données
```

**Problème : Turnover trop élevé**
```
Augmenter frequence_rebalancement à 63 (trimestriel)
```

---

**Fin de la documentation - Version 1.0**
