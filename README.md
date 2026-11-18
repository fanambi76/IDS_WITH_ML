# Détection d'intrusions réseau par apprentissage automatique

Projet de soutenance de Licence 3 TCO (ASJA Antsirabe), soutenance prévue début juillet 2027.

**Problématique** : dans quelle mesure un modèle de Machine Learning entraîné sur des données de trafic
(NSL-KDD, CICIDS2017) détecte-t-il les intrusions réseau mieux qu'une détection par règles fixes,
et à quel coût en faux positifs ?

## Structure

```
ids-ml-projet/
├── README.md
├── requirements.txt          dépendances (le projet entier)
├── .gitignore                exclut data/raw, data/processed et les modèles
│
├── data/
│   ├── raw/                  données brutes, JAMAIS modifiées
│   │   └── MachineLearningCVE/    les 8 CSV CICIDS2017 (à télécharger)
│   ├── processed/            produits par le notebook 02
│   │   ├── cicids2017_clean.parquet
│   │   └── cicids2017_sample.parquet
│   └── demo/                 CSV de démonstration pour l'app
│
├── notebooks/                à exécuter dans l'ordre
│   ├── 01_nsl_kdd_demarrage.ipynb
│   ├── 02_cicids2017_chargement_nettoyage.ipynb
│   └── 03_cicids2017_modelisation.ipynb
│
├── models/                   produits par le notebook 03
│   ├── ids_model.joblib           détection normal / attaque
│   └── ids_family_model.joblib    famille d'attaque probable
│
├── app/                      démo Streamlit
│   ├── app.py
│   └── make_demo_csv.py
│
├── reports/                  éléments pour le mémoire
│   ├── figures/              graphiques exportés (matrices de confusion, courbes...)
│   └── resultats/            tableaux de résultats (CSV)
│
└── docs/
    ├── plan_memoire.md       plan des 7 chapitres relié au code
    ├── memoire/              versions du mémoire
    ├── soutenance/           diaporama, script de présentation
    └── references/           articles, documentation des datasets
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab                        # ouvre les notebooks dans l'ordre 01, 02, 03
```

1. **Notebook 01** : télécharge NSL-KDD tout seul (connexion nécessaire la première fois).
2. **Notebook 02** : télécharge d'abord `MachineLearningCSV.zip` sur
   https://www.unb.ca/cic/datasets/ids-2017.html et décompresse-le dans `data/raw/MachineLearningCVE/`.
3. **Notebook 03** : crée les modèles dans `models/`.
4. **Démo** (depuis la racine du projet) :
   ```bash
   python app/make_demo_csv.py
   streamlit run app/app.py
   ```
   Puis charge `data/demo/demo_trafic.csv` dans l'application.

## Règles du projet

- `data/raw/` n'est jamais modifié : tout traitement passe par un notebook et écrit dans `data/processed/`.
- Les chemins sont relatifs : les notebooks s'exécutent depuis `notebooks/`, l'app depuis la racine.
- Les figures et tableaux cités dans le mémoire sont exportés dans `reports/`.
