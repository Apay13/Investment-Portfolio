# 📊 DOCUMENTATION - Script Markowitz

## 🎯 Vue d'ensemble

Ce script implémente la **théorie moderne du portefeuille de Markowitz (1952)** avec calcul de la frontière efficiente et optimisation du ratio de Sharpe.

**Fichier :** `portfolio_markowitz_v2.py`

---

## 📋 Table des matières

1. [Architecture du code](#architecture)
2. [Configuration](#configuration)
3. [Blocs fonctionnels détaillés](#blocs)
4. [Mathématiques utilisées](#maths)
5. [Outputs générés](#outputs)
6. [Utilisation](#utilisation)

---

## 🏗️ Architecture du code {#architecture}

```
portfolio_markowitz_v2.py
├── IMPORTS & CONFIGURATION (lignes 1-50)
├── ÉTAPE 1 : Import données (lignes 51-150)
├── ÉTAPE 2 : Calcul rendements (lignes 151-200)
├── ÉTAPE 3 : Statistiques (lignes 201-350)
├── ÉTAPE 4 : Optimisation Markowitz (lignes 351-500)
├── ÉTAPE 5 : Frontière efficiente (lignes 501-700)
├── ÉTAPE 6 : Exports (lignes 701-850)
└── MAIN (lignes 851-950)
```

---

## ⚙️ Configuration {#configuration}

### **Dictionnaire CONFIG**

```python
CONFIG = {
    # Portfolio
    'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM'],
    'periode_annees': 3,
    
    # Finance
    'taux_sans_risque': 0.02,  # 2%
    'jours_trading_annee': 252,
    
    # Optimisation
    'n_portefeuilles_frontiere': 100,
    'n_portefeuilles_aleatoires': 5000,
    
    # Affichage
    'afficher_debug': False,
    'seuil_affichage_poids': 0.5,  # % minimum
}
```

**À modifier :**
- `tickers` : Votre liste d'actifs
- `periode_annees` : Historique à utiliser
- `n_portefeuilles_frontiere` : Précision de la courbe

---

## 🔧 Blocs fonctionnels détaillés {#blocs}

### **BLOC 1 : Import des données**

```python
def importer_donnees(tickers, date_debut, date_fin):
```

**Ce qu'il fait :**
1. Télécharge prix via yfinance
2. Gère 3 formats de sortie différents (robustesse)
3. Nettoie les données (supprime NaN)
4. Valide (minimum 2 actifs, 50 jours)

**Sorties :**
- DataFrame avec prix de clôture ajustés
- Index = dates
- Colonnes = tickers

---

### **BLOC 2 : Calcul des rendements**

```python
def calculer_rendements(prix):
```

**Formule utilisée :**
```
r(t) = ln(P(t) / P(t-1))
```

**Pourquoi logarithmiques ?**
- Additivité : r_total = r1 + r2 + r3
- Symétrie : -10% puis +10% ≠ retour au point de départ
- Normalité : Distribution plus proche de Gaussienne

**Sortie :**
- DataFrame de rendements quotidiens
- 1ère ligne supprimée (NaN)

---

### **BLOC 3 : Calcul des statistiques**

```python
def calculer_statistiques(rendements):
```

**Calculs effectués :**

**A) Rendements moyens annualisés**
```python
rendements_moyens = rendements.mean() * 252
```

**B) Volatilité annualisée**
```python
volatilite = rendements.std() * sqrt(252)
```

**C) Matrice de covariance annualisée**
```python
matrice_cov = rendements.cov() * 252
```

**D) Matrice de corrélation**
```python
matrice_corr = rendements.corr()
```

**Sortie : Dictionnaire**
```python
{
    'rendements_moyens': Series,
    'volatilite': Series,
    'variance': Series,
    'matrice_covariance': DataFrame,
    'matrice_correlation': DataFrame
}
```

---

### **BLOC 4 : Optimisation Markowitz**

#### **4.1 Performance d'un portfolio**

```python
def performance_portefeuille(poids, rendements_moyens, matrice_cov):
```

**Formules :**
```
Rendement portfolio = Σ(w(i) × r(i))
Volatilité portfolio = √(w^T × Σ × w)
```

**Pourquoi cette formule de volatilité ?**
```
Cas simple : w^T × Σ × w capture la covariance
Si corrélation = +1 : σ_p = Σ(w(i) × σ(i))
Si corrélation = -1 : σ_p < Σ(w(i) × σ(i))  ← diversification !
```

---

#### **4.2 Ratio de Sharpe**

```python
def ratio_sharpe_negatif(poids, rendements_moyens, matrice_cov, rf=0.02):
```

**Formule :**
```
Sharpe = (Rendement - Taux_sans_risque) / Volatilité
```

**Interprétation :**
- Sharpe > 1 : Bon
- Sharpe > 2 : Très bon
- Sharpe > 3 : Excellent

**Pourquoi négatif ?**
```python
return -(r - rf) / v  # On veut MINIMISER le négatif = MAXIMISER le positif
```

---

#### **4.3 Optimisation**

```python
def optimiser_portefeuille(rendements_moyens, matrice_covariance):
```

**Deux optimisations :**

**A) Portfolio Max Sharpe**
```python
minimize(ratio_sharpe_negatif, ...)
```
- Objectif : Meilleur rendement ajusté au risque
- Contrainte : Σ(w) = 1
- Bornes : 0 ≤ w(i) ≤ 1

**B) Portfolio Min Volatilité**
```python
minimize(lambda w: performance_portefeuille(w)[1], ...)
```
- Objectif : Risque minimal
- Contrainte : Σ(w) = 1
- Bornes : 0 ≤ w(i) ≤ 1

**Méthode d'optimisation :** SLSQP (Sequential Least SQuares Programming)

---

### **BLOC 5 : Frontière efficiente**

#### **5.1 Calcul de la frontière**

```python
def calculer_frontiere_efficiente(..., n_portefeuilles=100):
```

**Algorithme :**
1. Trouver portfolio min volatilité (point de départ)
2. Générer 100 rendements cibles entre min et max
3. Pour chaque cible :
   ```python
   minimize(volatilité)
   constraint: rendement = cible
   ```
4. Stocker résultats

**Sortie : DataFrame**
```python
columns: ['rendement', 'volatilite', 'sharpe', 'poids_AAPL', ...]
```

---

#### **5.2 Portfolios aléatoires**

```python
def generer_portefeuilles_aleatoires(..., n=5000):
```

**Pourquoi ?**
- Visualiser que la frontière domine TOUS les autres portfolios
- Montrer l'amélioration vs allocation aléatoire

**Méthode :**
```python
poids_random = np.random.random(n_actifs)
poids_normalized = poids / poids.sum()
```

---

### **BLOC 6 : Visualisations**

#### **6.1 Frontière efficiente**

```python
def tracer_frontiere_efficiente(...):
```

**Graphique 1 : Courbe frontière + scatter**
- Axe X : Volatilité (risque)
- Axe Y : Rendement
- Gris : 5000 portfolios aléatoires (colorés par Sharpe)
- Rouge : Frontière efficiente
- Étoile dorée : Max Sharpe
- Étoile rouge : Min Volatilité
- Diamants bleus : Actifs individuels

**Graphique 2 : Allocation des actifs**
- Barre dorée : Poids Max Sharpe
- Barre rouge : Poids Min Volatilité
- Valeurs affichées si > 2%

---

### **BLOC 7 : Exports**

#### **7.1 Export Excel**

```python
def exporter_resultats_excel(...):
```

**5 onglets créés :**
1. **Statistiques** : Rendement, volatilité, variance par actif
2. **Corrélation** : Matrice de corrélation complète
3. **Covariance** : Matrice de covariance complète
4. **Portfolios Optimaux** : Poids + métriques des 2 portfolios
5. **Frontière Efficiente** : 100 points de la frontière

---

#### **7.2 Export CSV**

```python
def exporter_resultats_csv(...):
```

Fichier simple avec :
- Ticker
- Poids_Max_Sharpe
- Poids_Min_Volatilite

---

#### **7.3 Heatmap corrélation**

```python
def tracer_heatmap_correlation(...):
```

**Visualisation :**
- Colormap vert-rouge (-1 à +1)
- Valeurs affichées dans chaque cellule
- Identifie rapidement les actifs corrélés

---

## 📐 Mathématiques utilisées {#maths}

### **1. Rendement d'un portfolio**

```
R_p = Σ(w_i × R_i)

où :
w_i = poids de l'actif i
R_i = rendement de l'actif i
```

### **2. Volatilité d'un portfolio (CLÉ !)**

```
σ_p = √(w^T × Σ × w)

Développé :
σ_p = √(Σ Σ w_i × w_j × Cov(i,j))
     i  j

où :
Σ = matrice de covariance
```

**Exemple 2 actifs :**
```
σ_p² = w1² σ1² + w2² σ2² + 2×w1×w2×Cov(1,2)
                                    └─ diversification !
```

### **3. Ratio de Sharpe**

```
Sharpe = (R_p - R_f) / σ_p

où :
R_f = taux sans risque (ex: 2%)
```

### **4. Optimisation sous contrainte**

```
Maximize : Sharpe(w)
Subject to : Σ w_i = 1
             0 ≤ w_i ≤ 1
```

Résolu par programmation quadratique (SLSQP).

---

## 📤 Outputs générés {#outputs}

### **Fichiers créés :**

1. **frontiere_efficiente.png**
   - Graphique 2 panels
   - Résolution : 300 DPI
   - Taille : ~2 MB

2. **heatmap_correlation.png**
   - Matrice colorée
   - Résolution : 300 DPI

3. **resultats_markowitz.xlsx**
   - 5 onglets
   - Toutes les métriques
   - Prêt pour analyse

4. **portfolios_optimaux.csv**
   - Format simple
   - Importable partout

---

## 🚀 Utilisation {#utilisation}

### **Basique**

```bash
python portfolio_markowitz_v2.py
```

### **Personnaliser**

```python
# Dans le script, modifier CONFIG

CONFIG = {
    'tickers': ['SPY', 'TLT', 'GLD'],  # Vos actifs
    'periode_annees': 5,                # 5 ans
}
```

### **Désactiver debug**

```python
CONFIG = {
    'afficher_debug': False,  # Pas de messages [DEBUG]
}
```

---

## 📊 Interprétation des résultats

### **Sharpe Ratio**
- < 0.5 : Médiocre
- 0.5 - 1.0 : Acceptable
- 1.0 - 2.0 : Bon
- \> 2.0 : Excellent

### **Corrélation**
- 0.0 - 0.3 : Faible (bonne diversification)
- 0.3 - 0.7 : Modérée
- 0.7 - 1.0 : Forte (mauvaise diversification)

### **Poids du portfolio**
- Si 1 actif > 40% : Sur-concentration
- Si 1 actif < 5% : Négligeable
- Idéal : Distribution équilibrée

---

## ⚠️ Limitations

1. **Sensibilité aux données passées**
   - Optimise sur historique ≠ futur
   - "Overfitting" possible

2. **Hypothèses de Markowitz**
   - Rendements = distribution normale
   - Corrélations = stables
   - Pas de coûts de transaction

3. **Instabilité des poids**
   - Petite variation de données → gros changement de poids
   - Voir Risk Parity / HRP pour alternative

---

## 🔗 Ressources

**Papers fondateurs :**
- Markowitz, H. (1952). "Portfolio Selection"
- Sharpe, W. (1964). "Capital Asset Pricing Model"

**Extensions possibles :**
- Black-Litterman (intégrer vues)
- Risk Parity (allocation par risque)
- Hierarchical Risk Parity (clustering)

---

## 📝 Notes techniques

### **Pourquoi 252 jours ?**
```
Jours de trading par an :
- US : 252 jours
- Europe : ~250 jours
- Asie : ~240 jours
```

### **Annualisation**
```python
# Volatilité
vol_annuelle = vol_quotidienne * √252

# Rendement
rdt_annuel = rdt_quotidien × 252
```

### **Format des dates**
```python
date_debut = '2023-01-01'  # Format ISO 8601
```

---

## 🐛 Troubleshooting

**Problème : "No data found"**
```
Solution : Vérifier symboles boursiers (AAPL pas APPLE)
```

**Problème : "Optimization failed"**
```
Solution : Données insuffisantes ou actifs trop corrélés
```

**Problème : Excel error**
```bash
pip install openpyxl
```

---

**Fin de la documentation - Version 2.0**
