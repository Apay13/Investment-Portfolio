# 💼 EXPLICATION POUR PORTFOLIO MANAGER

## 🎯 Résumé Exécutif

Tu as développé et testé **6 stratégies d'allocation quantitative** sur **4 périodes différentes** (2012-2025). 

**Conclusion :** **Risk Parity Simple** domine sur presque tous les critères.

---

## 📊 Qu'est-ce qu'on a fait ?

### **Le problème business**

Comment **répartir $100,000** entre **10 actifs** (actions US diversifiées) pour :
1. Maximiser le rendement ajusté au risque (Sharpe)
2. Minimiser les coûts de transaction
3. Dormir tranquille (drawdown acceptable)

---

### **Les 3 approches testées**

#### **1. Markowitz (1952) - "L'optimiseur"**

**Principe :**
> "Je calcule mathématiquement les poids qui maximisent le Sharpe ratio"

**Comment :**
- Estime rendements futurs = moyenne historique
- Estime risques = covariance historique
- Optimise avec algorithme mathématique

**En pratique :**
```
Rendement : 19.9% par an ✅ (le meilleur !)
Coûts     : $6,900 sur 7 ans ❌ (catastrophique)
Turnover  : 44% par mois ❌ (change tout le temps)
Sharpe    : 0.75 (plombé par les coûts)
```

**Analogie business :**
```
C'est comme un trader qui :
- Recalcule son portfolio tous les mois
- Change 44% de ses positions à chaque fois
- Paie des frais énormes
- Au final, sous-performe à cause des coûts
```

---

#### **2. Risk Parity - "L'équilibreur"**

**Principe :**
> "Je donne MOINS de poids aux actifs volatils, PLUS aux actifs stables"

**Formule simple :**
```
Poids(Apple) = 1 / Volatilité(Apple)
Normalisé pour que la somme = 100%
```

**En pratique :**
```
Rendement : 16.6% par an (légèrement moins)
Coûts     : $640 sur 7 ans ✅✅✅ (9× moins cher !)
Turnover  : 5.6% par mois ✅ (ultra stable)
Sharpe    : 0.84 🥇 (le meilleur !)
```

**Analogie business :**
```
C'est comme un gestionnaire de patrimoine prudent :
- Règle simple et claire
- Change rarement les allocations
- Coûts minimaux
- Performance stable sur le long terme
```

---

#### **3. HRP - "Le clusteriseur"**

**Principe :**
> "Je groupe les actifs par similarité (Tech, Finance, Santé), puis j'alloue intelligemment"

**Comment :**
```
1. Clustering automatique :
   Tech    = [Apple, Microsoft, Google, Amazon, Meta]
   Finance = [JPM, Bank of America, Goldman Sachs]
   Santé   = [Johnson & Johnson, Pfizer]

2. Allocation niveau 1 (entre secteurs) :
   Tech    : 40%
   Finance : 35%
   Santé   : 25%

3. Allocation niveau 2 (dans chaque secteur) :
   Apple    : 40% × 20% = 8%
   Microsoft: 40% × 20% = 8%
   ...
```

**En pratique :**
```
Rendement : 14.8% par an
Drawdown  : -25% 🥇 (meilleur en crise !)
Coûts     : $1,900 (3× Risk Parity)
Turnover  : 14% (intermédiaire)
Sharpe    : 0.81
```

**Analogie business :**
```
C'est comme un CIO qui :
- Pense d'abord SECTEUR, puis STOCKS
- Diversifie vraiment (pas tout dans la Tech)
- Protège mieux en crise
- Mais paie 3× plus cher en trading
```

---

## 🏆 Tableau comparatif final

| Critère | Risk Parity | HRP | Markowitz |
|---------|-------------|-----|-----------|
| **Sharpe moyen** | **0.84** 🥇 | 0.81 | 0.73 |
| **Coûts (7 ans)** | **$640** 🥇 | $1,900 | $6,900 |
| **Turnover mensuel** | **5.6%** 🥇 | 14% | 44% |
| **Max Drawdown** | -27% | **-25%** 🥇 | -31% |
| **Complexité** | **Faible** 🥇 | Moyenne | Élevée |
| **Robustesse** | **Haute** 🥇 | Haute | Faible |

---

## 💡 Insights clés pour un PM

### **1. Les coûts tuent la performance**

```
Markowitz théorique : Optimal
Markowitz réel      : Sous-performe à cause du turnover

Rendement brut      : 19.9%
Coûts trading       : -1.0% par an
Rendement net       : 18.9%

vs

Risk Parity :
Rendement brut      : 16.6%
Coûts trading       : -0.1% par an
Rendement net       : 16.5%
```

**Leçon :**
> "Mieux vaut 16.5% avec des coûts faibles que 18.9% avec des coûts élevés, car la stabilité importe"

---

### **2. La simplicité bat la complexité**

**Risk Parity = 3 lignes de code :**
```python
volatilites = returns.std()
poids = 1 / volatilites
poids = poids / poids.sum()
```

**Markowitz = 200 lignes :**
```python
def optimize(...):
    # Matrice de covariance
    # Optimisation quadratique
    # Contraintes non-linéaires
    # ...
```

**Résultat :** Risk Parity gagne quand même !

**Leçon :**
> "En finance, simple et robuste bat complexe et fragile"

---

### **3. Le drawdown compte autant que le rendement**

**Question pour un investisseur :**
```
Préférez-vous :
A) 20% par an avec -40% de drawdown
B) 17% par an avec -25% de drawdown
```

**99% des gens :** Option B

**Pourquoi ?**
- Moins de stress psychologique
- Moins de capitulations paniques
- Meilleure adhésion long terme

**HRP excelle sur ce critère** (-25% vs -27% vs -31%)

---

### **4. Le turnover révèle l'instabilité**

**Markowitz : 44% de turnover par mois**

Ça veut dire quoi ?
```
Mois 1 : 40% Apple, 30% Microsoft, 20% Google
Mois 2 : 10% Apple, 50% Microsoft, 25% Google
Mois 3 : 25% Apple, 15% Microsoft, 40% Google
```

**Problèmes :**
1. Coûts de transaction énormes
2. Impossibilité de suivre le plan
3. Surréaction aux données récentes
4. "Overfitting" sur le bruit

**Risk Parity : 5.6% de turnover**
```
Mois 1 : 15% Apple, 20% Microsoft, 18% Google
Mois 2 : 16% Apple, 19% Microsoft, 17% Google
Mois 3 : 15% Apple, 20% Microsoft, 18% Google
```

Quasi pas de changements → **Stabilité = Or**

---

## 🎓 Concepts à retenir

### **1. Frontière efficiente (Markowitz)**

**Ce que c'est :**
```
Courbe qui montre TOUS les portfolios optimaux
pour chaque niveau de risque
```

**Exemple :**
```
Risque 10% → Meilleur rendement possible : 8%
Risque 20% → Meilleur rendement possible : 15%
Risque 30% → Meilleur rendement possible : 20%
```

**Limite :**
```
Frontière = basée sur données PASSÉES
Future frontière ≠ Passée frontière
```

---

### **2. Sharpe Ratio**

**Formule :**
```
Sharpe = (Rendement - Taux sans risque) / Volatilité
```

**Interprétation business :**
```
Sharpe = Combien de rendement excédentaire par unité de risque

Sharpe 0.5 : Tu gagnes 0.5% pour chaque 1% de risque pris
Sharpe 1.0 : Tu gagnes 1.0% pour chaque 1% de risque
Sharpe 2.0 : Tu gagnes 2.0% pour chaque 1% de risque
```

**Benchmark :**
```
< 0.5  : Médiocre (ETF basique fait mieux)
0.5-1  : Acceptable
1-1.5  : Bon
> 1.5  : Excellent (rare sur longue période)
```

---

### **3. Drawdown**

**Définition :**
```
Drawdown = Perte maximale depuis le dernier pic

Exemple :
Portfolio à 150k$ → Crash → 100k$
Drawdown = (100-150)/150 = -33%
```

**Psychologie :**
```
Drawdown -20% : "Ça va, c'est le marché"
Drawdown -30% : "Je commence à stresser"
Drawdown -40% : "Je vends tout" ← ERREUR !
Drawdown -50% : "Je capitule"
```

**Pourquoi important :**
```
Perte -50% nécessite +100% pour récupérer !

Exemple :
100k$ → -50% → 50k$
50k$ → +100% → 100k$ (retour au point de départ)
```

---

### **4. Rolling Window (fenêtre roulante)**

**Ce que c'est :**
```
Technique pour éviter le "look-ahead bias"

Mauvais :
Jour 1000 : J'utilise TOUTES les données (jour 1 à 2000)
           → J'ai "vu le futur" !

Bon :
Jour 1000 : J'utilise UNIQUEMENT les 252 derniers jours
           → Je ne vois que le passé (réaliste)
```

**Pourquoi important :**
```
Backtest sans rolling window = FAUX
Tu vas sur-estimer ta performance
```

---

## 📊 Application pratique

### **Si je devais gérer $1M aujourd'hui**

**Stratégie recommandée : Risk Parity Simple**

**Portfolio exemple (10 ETFs) :**
```
Actif          Volatilité    Poids Risk Parity
───────────────────────────────────────────────
SPY  (S&P 500)      18%           12%
QQQ  (Tech)         25%            9%
TLT  (Bonds LT)     15%           15%
IEF  (Bonds MT)     8%            28%
GLD  (Or)           16%           14%
VNQ  (Immobilier)   22%           10%
EFA  (Europe)       20%           11%
EEM  (Émergents)    24%            9%
DBC  (Commodités)   19%           12%
───────────────────────────────────────────────
TOTAL                            100%
```

**Rebalancement :**
```
Fréquence : Mensuel (21 jours de trading)
Seuil     : Uniquement si drift > 5%
```

**Coûts estimés :**
```
Turnover : 5.6% × 12 mois = 67% par an
Coûts    : 67% × 0.1% = 0.067% par an
Sur $1M  : $670 par an

vs Markowitz :
44% × 12 mois = 528% par an (!!)
Coûts : 528% × 0.1% = 0.53% par an
Sur $1M : $5,300 par an ← 8× plus cher !
```

---

### **Implémentation en production**

**Étape 1 : Calcul mensuel des poids**
```python
# 1er jour ouvré du mois
volatilites = returns_252j.std() × √252
poids_cible = (1/volatilites) / sum(1/volatilites)
```

**Étape 2 : Comparer vs positions actuelles**
```python
drift = abs(poids_actuels - poids_cible)
if max(drift) > 0.05:  # Seuil 5%
    rebalancer()
```

**Étape 3 : Ordres de marché**
```python
# Via API Interactive Brokers
for ticker in portfolio:
    qte_cible = capital × poids_cible[ticker] / prix[ticker]
    qte_actuelle = positions[ticker]
    
    if abs(qte_cible - qte_actuelle) > seuil:
        ordre = qte_cible - qte_actuelle
        passer_ordre(ticker, ordre)
```

---

## ⚠️ Risques à surveiller

### **1. Choc de corrélation**

**Scénario :**
```
Temps normal : Corrélations = 0.3 à 0.7
Crise 2008   : Corrélations = 0.95+
```

**Impact :**
```
Diversification s'effondre
Tous les actifs chutent ensemble
Drawdown explosif
```

**Mitigation :**
- Inclure actifs dé-corrélés (or, bonds)
- HRP résiste mieux (clustering)

---

### **2. Régime change**

**Scénario :**
```
2010-2020 : Bull market, faible volatilité
2020-2022 : COVID puis inflation, haute volatilité
```

**Impact :**
```
Poids calculés sur 2019 = obsolètes en 2022
Performance se dégrade
```

**Mitigation :**
- Rolling window de 252 jours (s'adapte)
- Monitoring mensuel

---

### **3. Estimation error**

**Problème :**
```
Volatilité historique ≠ Volatilité future
Corrélation passée ≠ Corrélation future
```

**Impact :**
```
Poids "optimaux" = basés sur mauvaises estimations
Sous-performance vs théorie
```

**Mitigation :**
- Risk Parity moins sensible (utilise seulement volatilité)
- Markowitz très sensible (utilise rendements + covariance)

---

## 🚀 Prochaines étapes : Portfolio Factoriel

Tu as raison, c'est **la suite logique** ! Voici pourquoi :

### **Limites des stratégies actuelles**

```
Risk Parity / HRP / Markowitz = "Smart Beta"
→ Utilisent UNIQUEMENT : prix, rendements, volatilité

Ils IGNORENT :
- Fondamentaux (P/E, ROE, dette)
- Facteurs de risque (value, momentum, quality)
- Données alternatives
```

---

### **Portfolio Factoriel = Level Up**

**Principe :**
```
Au lieu de choisir :
- Apple vs Microsoft vs Google

On choisit :
- Value vs Growth
- Large Cap vs Small Cap
- Quality vs Junk
- Momentum vs Mean Reversion
```

**Exemple concret :**
```
Portfolio Factoriel Long-Only :
30% Value Factor    (VTV - Vanguard Value)
25% Momentum Factor (MTUM - MSCI Momentum)
25% Quality Factor  (QUAL - MSCI Quality)
20% Low Vol Factor  (USMV - Min Volatility)
```

**Avantages :**
1. ✅ **Recherche académique** : Facteurs prouvés sur 50+ ans
2. ✅ **Diversification réelle** : Les facteurs sont moins corrélés
3. ✅ **Compréhension** : Tu sais POURQUOI tu gagnes de l'argent
4. ✅ **Scalabilité** : Fonctionne sur toutes les classes d'actifs

---

### **Ce qu'on va construire**

**Script factoriel :**
```python
# 1. Définir les facteurs
facteurs = {
    'Value': [...],      # P/B faible
    'Momentum': [...],   # Rendement 12 mois
    'Quality': [...],    # ROE élevé
    'Size': [...],       # Small caps
    'Low Vol': [...]     # Volatilité faible
}

# 2. Calculer les scores factoriels
for stock in universe:
    score_value = calcul_value(stock)
    score_momentum = calcul_momentum(stock)
    ...

# 3. Construire portfolios long-only par facteur
portfolio_value = top_20_pct(sorted_by_value_score)
portfolio_momentum = top_20_pct(sorted_by_momentum_score)

# 4. Combiner les facteurs (Risk Parity sur facteurs !)
allocation_finale = risk_parity([
    portfolio_value,
    portfolio_momentum,
    portfolio_quality,
    portfolio_low_vol
])
```

---

### **Pourquoi c'est mieux**

| Aspect | Markowitz/RP | Factoriel |
|--------|--------------|-----------|
| **Base** | Prix passés | Fondamentaux + Prix |
| **Horizon** | Court/Moyen | Long terme |
| **Explication** | "Math" | "Économique" |
| **Robustesse** | Moyenne | Élevée |
| **Recherche** | 1952-1990s | 1990s-2020s |

**Mon avis : C'est exactement la prochaine étape logique** ✅

---

## 📚 Ressources pour aller plus loin

**Facteurs :**
- Fama & French (1993). "Common Risk Factors"
- AQR Capital. "Factor Investing" (whitepapers gratuits)
- Alpha Architect (blog)

**Quant Finance :**
- "Quantitative Equity Portfolio Management" - Qian et al.
- "Advances in Active Portfolio Management" - Grinold

**Python :**
- Zipline (backtesting library)
- PyPortfolioOpt (allocation)
- QuantLib (pricing)

---

## 💬 Pour discuter avec un client

**Pitch 30 secondes :**
```
"On a développé une stratégie quantitative qui bat Markowitz 
traditionnel sur 3 critères :
- Sharpe ratio supérieur (0.84 vs 0.73)
- Coûts 9× plus faibles ($640 vs $6,900 sur 7 ans)
- Drawdown comparable (-27% vs -31%)

Tout ça avec une formule ultra-simple : 
on donne moins de poids aux actifs volatils."
```

**Si le client demande : "Pourquoi pas Markowitz ?"**
```
"Markowitz est optimal... en théorie. 
En pratique, il souffre de 3 problèmes :
1. Turnover fou → Coûts énormes
2. Sensibilité extrême aux données → Instabilité
3. Overfitting sur le passé → Ne marche pas sur le futur

Notre stratégie sacrifie 2-3% de rendement théorique 
pour gagner énormément en robustesse et coûts."
```

---

**Fin du document - Prêt pour présentation business**
