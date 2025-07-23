# laboratoire-biomidical-project
-----

# 🩺 Système intelligent de prédiction du risque de diabète

-----

## Description du projet

Ce projet vise à développer un système intelligent capable de **prédire le risque de développer le diabète** chez un patient. Basé sur des critères cliniques clés tels que la glycémie (Glucose), la pression artérielle (Blood Pressure), l'épaisseur du pli cutané (Skin Thickness), l'insuline, l'Indice de Masse Corporelle (BMI), la fonction de prédisposition génétique au diabète (Diabetes Pedigree Function) et l'âge, ce système offre une double fonctionnalité :

1.  **Classification des patients** : Catégoriser un patient comme présentant un risque élevé ou faible de diabète.
2.  **Clustering des données** : Regrouper les patients ayant des profils cliniques similaires afin d'identifier des comportements ou des caractéristiques communes au sein de la population étudiée.

L'objectif est de fournir un outil d'aide à la décision pour les professionnels de la santé, permettant une intervention précoce et une gestion proactive du risque de diabète.

-----

## 📊 Méthodologie

Le développement de ce système intelligent a suivi une approche structurée, englobant l'analyse exploratoire, le prétraitement, le clustering et la classification :

### 1\. Chargement et Analyse Exploratoire des Données (EDA)

  * **Importation** des données cliniques historiques (format CSV) à l'aide de Pandas.
  * **Compréhension de la structure** du jeu de données (types de colonnes, dimensions, aperçu des premières lignes).
  * **Identification des valeurs manquantes et des doublons**.
  * **Analyse de la distribution** des variables numériques via des histogrammes et des boîtes à moustaches.
  * **Étude des corrélations** entre les variables à l'aide d'une matrice de corrélation et de `pairplots`.

### 2\. Prétraitement des Données

  * **Gestion des valeurs manquantes** : Remplacement des valeurs 0 (représentant des NaN) dans des colonnes comme 'Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness' par la **médiane** de leur colonne respective.
  * **Détection et suppression des valeurs aberrantes (outliers)** : Utilisation de la méthode de l'**IQR (InterQuartile Range)** pour identifier et supprimer les lignes contenant des valeurs extrêmes dans les colonnes pertinentes, garantissant la robustesse du modèle.
  * **Sélection des variables** : Choix des caractéristiques les plus pertinentes pour le clustering et la classification (`Glucose`, `BMI`, `Age`, `DiabetesPedigreeFunction`).
  * **Mise à l'échelle des variables** : Application de `StandardScaler` pour homogénéiser les échelles des variables numériques, une étape cruciale pour la plupart des algorithmes d'apprentissage automatique.

### 3\. Clustering avec K-Means

  * **Détermination du nombre optimal de clusters (`k`)** : Utilisation de la **méthode du coude** (Elbow Method) pour évaluer l'inertie pour différentes valeurs de `k` et identifier le point d'inflexion optimal.
  * **Entraînement du modèle K-Means** : Création des clusters basés sur les données prétraitées et mises à l'échelle.
  * **Assignation des clusters** : Ajout d'une colonne `Cluster` au jeu de données original, indiquant le groupe attribué à chaque patient.
  * **Interprétation des clusters** : Calcul des moyennes des caractéristiques pour chaque cluster afin de comprendre les profils de patients.
  * **Catégorisation du risque** : Création d'une colonne `risk_category` (`High Risk` / `Low Risk`) basée sur l'interprétation des clusters.

### 4\. Réduction de Dimensionnalité (PCA)

  * **Application de l'Analyse en Composantes Principales (PCA)** : Réduction des dimensions des données à 2 composantes principales (`PC1`, `PC2`) pour faciliter la visualisation.
  * **Visualisation des clusters** : Affichage des clusters dans l'espace 2D de la PCA pour une meilleure compréhension de leur répartition.

### 5\. Classification des Patients

  * **Préparation des données** :
      * Définition de la variable cible `y` (`risk_category`) et des variables explicatives `X`.
      * **Division des données** en ensembles d'entraînement (70%) et de test (30%) via `train_test_split`.
      * **Gestion du déséquilibre des classes** : Utilisation de `RandomOverSampler` sur l'ensemble d'entraînement pour équilibrer la répartition des catégories de risque.
  * **Entraînement de plusieurs modèles de classification** :
      * **Random Forest Classifier**
      * **Support Vector Machine (SVM)**
      * **Gradient Boosting Classifier**
      * **Régression Logistique**
  * **Évaluation des modèles** : Utilisation de métriques clés pour évaluer les performances de chaque modèle sur l'ensemble de test :
      * **Matrice de confusion**
      * **Rapport de classification** (précision, rappel, F1-score)
      * **Accuracy**
  * **Validation croisée** : Évaluation de la robustesse des modèles sur différentes partitions du jeu de données (5-fold cross-validation) pour obtenir une estimation plus fiable de leur performance généralisée.
  * **Optimisation des hyperparamètres** : Application de `GridSearchCV` pour affiner les hyperparamètres des modèles `RandomForestClassifier` et `GradientBoostingClassifier`, afin d'améliorer leur performance.
  * **Sélection et sauvegarde du meilleur modèle** : Comparaison des performances finales des modèles (en privilégiant le F1-score pour la classification avec déséquilibre) et sauvegarde du modèle le plus performant au format `.pkl` pour un déploiement ultérieur.

-----

## 📁 Structure des fichiers

```
.
├── diabete_data.csv
├── diabetes_prediction.ipynb
├── diabetes_risk_prediction_model.pkl
├── scaler.pkl
├── app.py                     # NOUVEAU: Votre application Streamlit
├── requirements.txt           # NOUVEAU: Vos dépendances Python
├── Dockerfile                 # NOUVEAU: Les instructions Docker
└── README.md
```

-----

## 🚀 Instructions pour l'exécuter

Pour exécuter ce projet sur votre machine locale, suivez les étapes ci-dessous :

### 1\. Prérequis

Assurez-vous d'avoir Python 3.x installé. Vous pouvez vérifier votre version de Python avec :

```bash
python --version
```

### 2\. Cloner le dépôt (si applicable)

Si ce projet est sur un dépôt Git, clonez-le :

```bash
git clone https://github.com/SalimM21/laboratoire-biom-dical-project/tree/main
cd laboratoire-biom-dical-project

```

### 3\. Créer et activer un environnement virtuel (recommandé)

```bash
python -m venv venv
# Pour Windows
.\venv\Scripts\activate
# Pour macOS/Linux
source venv/bin/activate
```

### 4\. Installer les dépendances

Installez toutes les bibliothèques nécessaires :

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn joblib
```

### 5\. Exécuter le Notebook Jupyter

Lancez Jupyter Notebook dans le répertoire du projet :

```bash
jupyter notebook
```

Ouvrez le fichier `diabetes_prediction.ipynb` et exécutez toutes les cellules séquentiellement. Le notebook contient toutes les étapes, du chargement des données à l'évaluation et la sauvegarde du modèle.

### 6\. Vérifier le modèle sauvegardé

Après l'exécution complète du notebook, un fichier `diabetes_risk_prediction_model.pkl` sera créé dans le même répertoire, représentant le modèle entraîné et prêt à l'emploi.

-----
