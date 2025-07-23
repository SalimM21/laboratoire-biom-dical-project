# Dockerfile

# Utilisez une image de base Python légère
FROM python:3.9-slim-buster

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers de dépendances et installez-les
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiez le reste de l'application (y compris app.py et votre modèle)
COPY . .

# Exposez le port sur lequel Streamlit s'exécute par défaut
EXPOSE 8501

# Commande pour exécuter l'application Streamlit lorsque le conteneur démarre
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]