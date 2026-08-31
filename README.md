# Lab : Algorithmes d'Ensemble

Module : Machine Learning Avancé — Cohorte IA

## Objectif

Explorer les algorithmes d'ensemble (Bagging, Boosting, Voting, Stacking) pour lutter contre le surapprentissage et améliorer les performances d'un modèle de classification binaire, en comprenant la mécanique interne de la réduction de variance (Bagging) et de la réduction de biais (Boosting).

## Données

Jeu de données synthétique généré via `sklearn.datasets.make_classification` :

- 2500 échantillons
- 20 features, dont 12 informatives
- Classification binaire
- Split train/test : 80% / 20% (`random_state=42`)

## Tableau récapitulatif des performances

| Modèle                               | Accuracy (Test)                             | Temps d'entraînement (s) | Observations / Surapprentissage                                                                                                                                                                                                                                                |
| ------------------------------------ | ------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Arbre Seul                           | 0.840                                       | 0.0736                   | Fort surapprentissage : accuracy train = 1.00 vs test = 0.84 (écart de 0.16). L'arbre sans limite de profondeur mémorise les données d'entraînement au lieu de généraliser.                                                                                                    |
| Bagging manuel (B=50)                | 0.922                                       | 2.4860                   | Gain net vs arbre seul grâce au vote majoritaire sur 50 arbres entraînés sur des échantillons bootstrap différents. Réduit la variance en moyennant des erreurs individuelles peu corrélées.                                                                                   |
| Random Forest                        | 0.930 (par défaut) / 0.946 (max_features=5) | 0.6119                   | Courbe en cloche selon`max_features` : trop bas (1) → arbres individuellement trop faibles ; trop haut (20) → arbres trop corrélés (proche du Bagging classique). Optimal proche de √20 ≈ 5, valeur par défaut de scikit-learn.                                                |
| AdaBoost (stumps, learning_rate=1.0) | 0.812                                       | 0.5692                   | Meilleur score parmi les learning_rate testés (0.01 → 0.682, 1.0 → 0.812, 1.5 → 0.790). Reste en retrait des méthodes de Bagging : les stumps (profondeur 1) peinent sur un problème où l'information est répartie sur plusieurs features en interaction.                      |
| XGBoost (n_estimators=200)           | 0.956                                       | 0.3563                   | Meilleur modèle individuel. Courbe d'apprentissage (log-loss) : le train continue de descendre vers 0 jusqu'à la fin, tandis que le test atteint son minimum vers l'itération 40-50 puis stagne/remonte légèrement — signe d'un léger surapprentissage à partir de cette zone. |
| Voting (Hard)                        | 0.952                                       | 1.7885                   | Vote majoritaire sur les classes prédites par LogReg, SVM, RF, XGBoost.                                                                                                                                                                                                        |
| Voting (Soft)                        | 0.966                                       | 1.6734                   | **Meilleur score global.** Moyenne des probabilités plutôt qu'un vote binaire : exploite le niveau de confiance de chaque modèle et dépasse même le meilleur modèle individuel (XGBoost, 0.956).                                                                               |
| Stacking                             | 0.962                                       | 9.2508                   | Très proche du Soft Voting sans le dépasser. Le méta-modèle (Régression Logistique, cv=5) n'a pas trouvé de combinaison non-linéaire suffisamment robuste pour battre la simple moyenne des probabilités sur ce jeu de données.                                                |

## Réponses aux questions de réflexion

### Pourquoi la limitation des variables à chaque nœud (max_features) améliore-t-elle la diversité des arbres par rapport à un Bagging classique ?

Dans le Bagging classique, toutes les features restent disponibles à chaque nœud. Si une feature est très informative, elle est choisie comme meilleure coupure à la racine de la quasi-totalité des arbres, quel que soit l'échantillon bootstrap utilisé — les arbres se ressemblent donc fortement et sont corrélés entre eux, ce qui limite le bénéfice du vote majoritaire.

Random Forest force, à chaque nœud, le choix de la meilleure coupure parmi un sous-ensemble aléatoire de features (`max_features`). Une feature dominante n'est donc pas systématiquement disponible, ce qui oblige les arbres à explorer des structures différentes et à s'appuyer sur des features secondaires. Les arbres deviennent moins corrélés entre eux, leurs erreurs sont plus indépendantes, et le vote majoritaire réduit davantage la variance globale.

### Point de surapprentissage sur XGBoost (courbe Log-Loss)

La log-loss sur le train diminue continuellement jusqu'à quasiment 0 sur les 200 itérations. La log-loss sur le test diminue rapidement au début, atteint un minimum autour de l'itération 40-50 (≈ 0.14), puis se stabilise et remonte très légèrement (0.135-0.145) jusqu'à la fin. Ce point de divergence entre les deux courbes marque le début du surapprentissage : au-delà, les arbres supplémentaires captent du bruit spécifique au train plutôt que du signal généralisable. Le surapprentissage reste toutefois léger sur ce jeu de données.

## Conclusion générale

- Les méthodes d'ensemble améliorent systématiquement les performances par rapport à un arbre seul, en s'attaquant soit à la variance (Bagging, Random Forest), soit au biais (Boosting).
- Le Boosting par repondération (AdaBoost) est moins performant ici que le Gradient Boosting (XGBoost), qui corrige directement les résidus via la descente de gradient.
- Combiner des modèles hétérogènes (Voting, Stacking) apporte le meilleur résultat global, le Soft Voting (0.966) surpassant tous les modèles individuels ainsi que le Stacking.
- La complexité supplémentaire du Stacking n'est pas toujours synonyme de meilleure performance : sur ce dataset, la simple moyenne des probabilités (Soft Voting) captait déjà l'essentiel de la complémentarité entre modèles.
