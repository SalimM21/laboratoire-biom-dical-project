import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import joblib # Pour charger le modèle et le scaler
from sklearn.preprocessing import StandardScaler # Pour le type hint et vérification

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Système de Prédiction du Risque de Diabète",
    page_icon="🩺",
    layout="wide"
)

# --- Fonction de chargement des ressources (modèle et scaler) ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('diabetes_risk_prediction_model.pkl')
        st.success("Modèle de prédiction chargé avec succès !", icon="✅")
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'diabetes_risk_prediction_model.pkl' est introuvable. Assurez-vous qu'il est dans le même répertoire.", icon="❌")
        st.stop()

    try:
        scaler = joblib.load('scaler.pkl')
        st.success("Scaler chargé avec succès !", icon="✅")
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'scaler.pkl' est introuvable. Veuillez vous assurer que le scaler a été sauvegardé depuis le notebook.", icon="❌")
        st.stop()
    return model, scaler

# Charger le modèle et le scaler au démarrage de l'application
model, scaler = load_resources()

# --- Chargement et prétraitement des données (simulé/simplifié pour la démo) ---
# Dans une vraie application, vous chargeriez et appliqueriez ici le même prétraitement que dans le notebook
@st.cache_data
def load_data():
    try:
        # Tente de charger votre dataset de diabète
        df = pd.read_csv('dataDiabète.csv')
        # Pour simuler la colonne 'Cluster' et 'risk_category' après clustering
        # C'est une SIMULATION, dans une vraie app, ces colonnes viendraient du vrai clustering
        if 'Cluster' not in df.columns:
            # Créer des clusters aléatoires pour la démo si la colonne n'existe pas
            st.warning("La colonne 'Cluster' n'est pas présente dans 'dataDiabète.csv'. Des clusters aléatoires seront générés pour la visualisation.")
            df['Cluster'] = np.random.randint(0, 2, size=len(df)) # 2 clusters: 0 ou 1
        if 'risk_category' not in df.columns:
            st.warning("La colonne 'risk_category' n'est pas présente. Elle sera générée à partir des clusters pour la démo.")
            df['risk_category'] = df['Cluster'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')

        # Si vous voulez montrer le DF scaled, vous devriez le sauvegarder aussi du notebook
        # et le charger ici. Pour la simplicité, nous utilisons df original.
        return df
    except FileNotFoundError:
        st.error("Fichier 'dataDiabète.csv' non trouvé. Veuillez le placer dans le même répertoire que l'application.", icon="❌")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}", icon="❌")
        st.stop()

df = load_data()
clustering_cols = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction'] # Définition des colonnes pour les analyses


# --- Barre latérale pour la navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à la page :",
    ["Accueil",
     "1. Analyse Exploratoire des Données (EDA)",
     "2. Clustering (K-Means & PCA)",
     "3. Classification des Patients",
     "4. Prédiction du Risque"
    ]
)

# --- Contenu des pages ---

# Page d'Accueil
if page == "Accueil":
    st.image("https://via.placeholder.com/800x300.png?text=Image+d'accueil+du+projet", caption="Système intelligent de prédiction du risque de diabète")
    st.title("Système intelligent de prédiction du risque de diabète")
    st.markdown("""
    Ce projet vise à développer un système intelligent capable de **prédire le risque de développer le diabète** chez un patient et de **classifier les patients** en groupes à risque.

    Utilisez le menu de navigation à gauche pour explorer les différentes étapes de l'analyse,
    du prétraitement des données à la prédiction finale du risque.
    """)
    st.info("Ce projet est une démonstration basée sur un modèle de Machine Learning entraîné sur des données de santé.")


# Page 1: Analyse Exploratoire des Données (EDA)
elif page == "1. Analyse Exploratoire des Données (EDA)":
    st.header("1. Analyse Exploratoire des Données (EDA)")
    st.markdown("""
    Cette section présente un aperçu des données utilisées, leurs distributions et les relations entre les variables.
    """)

    st.subheader("Aperçu du Jeu de Données")
    st.write("Les 5 premières lignes des données :")
    st.dataframe(df.head())
    st.write(f"**Dimensions du DataFrame :** {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    st.write("Statistiques descriptives des variables numériques :")
    st.dataframe(df.describe())

    st.subheader("Distribution des Variables Numériques (Histogrammes)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_eda_col = st.selectbox("Sélectionnez une colonne à visualiser :", numeric_cols, key="eda_hist_select")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_eda_col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {selected_eda_col}")
        ax.set_xlabel(selected_eda_col)
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)
    else:
        st.info("Aucune colonne numérique trouvée pour l'EDA.")

    st.subheader("Matrice de Corrélation")
    st.markdown("Visualisation des relations linéaires entre toutes les variables numériques.")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.select_dtypes(include=np.number).corr()
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Matrice de Corrélation")
        st.pyplot(fig)
    else:
        st.info("Aucune donnée numérique pour calculer la matrice de corrélation.")


# Page 2: Clustering (K-Means & PCA)
elif page == "2. Clustering (K-Means & PCA)":
    st.header("2. Clustering (K-Means & PCA)")
    st.markdown("""
    Cette section démontre comment les patients sont regroupés en clusters basés sur leurs caractéristiques,
    et comment la PCA aide à visualiser ces groupes.
    """)

    st.subheader("Méthode du Coude pour K-Means")
    st.image("https://via.placeholder.com/600x400.png?text=Courbe+d'inertie+du+coude", caption="Exemple de courbe d'inertie (à remplacer par votre propre graphique)")
    st.markdown("""
    La méthode du coude est utilisée pour déterminer le nombre optimal de clusters (k).
    Nous observons le point où la diminution de l'inertie commence à ralentir significativement,
    indiquant la valeur de k optimale (souvent 2 ou 3 dans notre cas pour la catégorie de risque).
    """)

    st.subheader("Répartition des Observations par Cluster")
    # Simulation de la répartition si la colonne 'Cluster' est là
    if 'Cluster' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Cluster', data=df, ax=ax)
        ax.set_title('Répartition des Observations par Cluster')
        ax.set_xlabel('Numéro de Cluster')
        ax.set_ylabel('Nombre d\'Observations')
        st.pyplot(fig)
        st.write("Comptage des observations par cluster :", df['Cluster'].value_counts().sort_index())
    else:
        st.warning("La colonne 'Cluster' n'est pas disponible pour la visualisation. Veuillez vous assurer que le clustering a été effectué.")


    st.subheader("Interprétation des Clusters (Moyennes des Caractéristiques)")
    st.markdown("Les moyennes des caractéristiques pour chaque cluster nous aident à comprendre leurs profils distincts.")
    if 'Cluster' in df.columns and all(col in df.columns for col in clustering_cols):
        # Utilisation de la version non-scaled du DF pour une meilleure interprétation des moyennes
        cluster_means = df.groupby('Cluster')[clustering_cols].mean()
        st.dataframe(cluster_means)
        st.markdown(f"""
        Dans notre cas, si **Cluster 1** montre des valeurs moyennes plus élevées pour
        `Glucose`, `BMI` et `DiabetesPedigreeFunction` par rapport au **Cluster 0**,
        cela indique que le Cluster 1 correspond aux patients à **risque élevé** de diabète,
        tandis que le Cluster 0 correspond aux patients à **faible risque**.
        """)
    else:
        st.info("Impossible de calculer les moyennes par cluster sans la colonne 'Cluster' ou les colonnes de clustering.")


    st.subheader("Visualisation des Clusters avec PCA (Réduction de Dimensionnalité)")
    st.markdown("""
    L'Analyse en Composantes Principales (PCA) réduit la complexité des données en 2 dimensions
    pour faciliter la visualisation des clusters.
    """)
    # Pour cette partie, nous avons besoin de données scaled pour la PCA.
    # Dans une vraie app, vous feriez la PCA sur df_scaled.
    # Ici, nous allons simuler un df_pca pour la démo si df n'est pas scaled
    try:
        from sklearn.decomposition import PCA
        # Re-scaling pour la démo si df n'est pas passé par le pipeline complet
        temp_df_scaled = pd.DataFrame(scaler.fit_transform(df[clustering_cols]), columns=clustering_cols, index=df.index)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(temp_df_scaled)
        df_pca = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])
        df_pca['Cluster'] = df['Cluster'] # Assurez-vous que la colonne Cluster est disponible

        chart_pca = alt.Chart(df_pca).mark_point(size=60).encode(
            x='PC1',
            y='PC2',
            color='Cluster:N', # 'N' pour Nominal (catégoriel)
            tooltip=['PC1', 'PC2', 'Cluster']
        ).properties(
            title="Clusters K-Means visualisés avec PCA"
        ).interactive() # Permet le zoom et le déplacement
        st.altair_chart(chart_pca, use_container_width=True)
        st.write(f"Variance expliquée par les 2 premières composantes : {pca.explained_variance_ratio_.sum()*100:.2f}%")
    except Exception as e:
        st.error(f"Erreur lors de la visualisation PCA : {e}. Assurez-vous que les données et la colonne 'Cluster' sont correctement préparées.", icon="❌")


# Page 3: Classification des Patients
elif page == "3. Classification des Patients":
    st.header("3. Classification des Patients")
    st.markdown("""
    Cette section présente le processus de classification pour prédire le risque de diabète.
    Nous avons entraîné et évalué plusieurs modèles de Machine Learning.
    """)

    st.subheader("Préparation des Données pour la Classification")
    st.markdown("""
    * **Variable Cible (Y) :** `risk_category` (Faible Risque / Haut Risque)
    * **Variables Explicatives (X) :** `Glucose`, `BMI`, `Age`, `DiabetesPedigreeFunction` (après mise à l'échelle)
    * **Division des données :** 70% entraînement, 30% test.
    * **Gestion du déséquilibre des classes :** Utilisation de sur-échantillonnage (`RandomOverSampler`) sur l'ensemble d'entraînement pour garantir l'équité du modèle.
    """)
    st.write(f"**Répartition des classes dans l'ensemble de données (après simulation si nécessaire) :**")
    st.write(df['risk_category'].value_counts())

    st.subheader("Modèles de Classification Testés")
    st.markdown("""
    Nous avons évalué les algorithmes suivants :
    * **Random Forest Classifier**
    * **Support Vector Machine (SVM)**
    * **Gradient Boosting Classifier**
    * **Régression Logistique**
    """)

    st.subheader("Évaluation des Modèles : Métriques Clés")
    st.markdown("Comparaison des performances des modèles sur l'ensemble de test, en privilégiant le **F1-Score** pour un problème avec déséquilibre de classes.")

    # Exemples de métriques (À REMPLACER PAR VOS VRAIS RÉSULTATS DE NOTEBOOK)
    metrics_data = {
        'Modèle': ['Random Forest (Optimisé)', 'Gradient Boosting (Optimisé)', 'SVM', 'Régression Logistique'],
        'Accuracy': [0.855, 0.842, 0.798, 0.785],
        'Precision (High Risk)': [0.805, 0.795, 0.752, 0.740],
        'Recall (High Risk)': [0.838, 0.825, 0.725, 0.710],
        'F1-Score (High Risk)': [0.821, 0.809, 0.738, 0.725],
        'F1-Score CV Moyenne': [0.840, 0.830, 0.780, 0.770]
    }
    metrics_df = pd.DataFrame(metrics_data).set_index('Modèle')
    st.dataframe(metrics_df)

    st.subheader("Matrice de Confusion du Meilleur Modèle")
    st.markdown("La matrice de confusion du modèle `Random Forest (Optimisé)` montre la précision des prédictions :")
    conf_matrix_best_model = pd.DataFrame({
        'Prédit Faible Risque': [170, 30],
        'Prédit Haut Risque': [20, 130]
    }, index=['Réel Faible Risque', 'Réel Haut Risque'])
    st.dataframe(conf_matrix_best_model)
    st.markdown("""
    * **VP (Vrais Positifs) :** 130 patients à haut risque correctement identifiés.
    * **VN (Vrais Négatifs) :** 170 patients à faible risque correctement identifiés.
    * **FP (Faux Positifs) :** 20 patients à faible risque classés par erreur comme à haut risque.
    * **FN (Faux Négatifs) :** 30 patients à haut risque classés par erreur comme à faible risque (cas critique !).
    """)

    st.success("Le modèle **Random Forest (Optimisé)** a été sélectionné comme le plus performant pour ce projet, offrant un bon équilibre entre précision et rappel pour la classe à haut risque.")


# Page 4: Prédiction du Risque
elif page == "4. Prédiction du Risque":
    st.header("4. Prédiction du Risque de Diabète")
    st.markdown("""
    Utilisez ce formulaire interactif pour entrer les informations d'un nouveau patient
    et obtenir une prédiction immédiate de son risque de diabète.
    """)

    # --- Formulaire d'entrée pour la prédiction ---
    with st.form("prediction_form"):
        st.subheader("Informations du Patient :")
        col1, col2 = st.columns(2)
        glucose = col1.number_input('Glucose (mg/dL)', min_value=0, max_value=300, value=120, help="Niveau de glucose plasmatique à 2 heures lors d'un test de tolérance au glucose.")
        bmi = col2.number_input('IMC (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, help="Indice de Masse Corporelle.")

        col3, col4 = st.columns(2)
        age = col3.number_input('Âge', min_value=0, max_value=120, value=30, help="Âge du patient en années.")
        diabetes_pedigree_function = col4.number_input('Fonction de Préd. Génétique au Diabète', min_value=0.0, max_value=2.5, value=0.5, format="%.3f", help="Un score qui indique la probabilité de diabète basée sur l'historique familial.")

        submitted = st.form_submit_button("Prédire le Risque")

    if submitted:
        # Créer un DataFrame avec les entrées utilisateur
        input_data = pd.DataFrame([[glucose, bmi, age, diabetes_pedigree_function]],
                                   columns=clustering_cols) # Utiliser clustering_cols pour s'assurer de l'ordre

        # Mettre à l'échelle les données d'entrée en utilisant le scaler chargé
        # Assurez-vous que 'scaler' est bien l'objet StandardScaler entraîné
        scaled_input_data = scaler.transform(input_data)

        # Effectuer la prédiction
        prediction = model.predict(scaled_input_data)

        st.subheader("Résultat de la Prédiction :")
        if prediction[0] == 'High Risk':
            st.error(f"Le patient est classé comme **risque élevé** de développer le diabète. Action recommandée : Consultation médicale.", icon="🚨")
        else:
            st.success(f"Le patient est classé comme **faible risque** de développer le diabète. Poursuite des bonnes habitudes de vie.", icon="👍")

        st.info("Cette prédiction est un outil d'aide à la décision et ne remplace pas l'avis d'un professionnel de la santé qualifié.")