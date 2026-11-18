# Plan du mémoire relié au projet

| Chapitre | Contenu | Où dans le projet |
|---|---|---|
| Introduction générale | contexte, problématique, objectifs, hypothèses, méthodologie | README.md (problématique) |
| 1. Réseaux et cybersécurité | protocoles, attaques, IDS/IPS, limites des règles | docs/references/ ; lien avec tes dépôts Wazuh/Sigma |
| 2. IA et Machine Learning | supervisé, non supervisé, algorithmes utilisés | notebooks 01 et 03 (choix des modèles) |
| 3. Étude et préparation des données | NSL-KDD, CICIDS2017, nettoyage, déséquilibre | notebooks 01 et 02 (tableau de bilan du nettoyage) |
| 4. Conception | architecture en deux étages (détection puis famille), critères d'évaluation | notebook 03, sections 2 et 9 |
| 5. Implémentation | environnement, entraînement, sauvegarde, démo Streamlit | notebook 03, `app/` |
| 6. Tests et résultats | protocoles A et B, métriques, erreurs, seuil, expérience sans port | notebook 03, sections 3 à 8 ; exports dans `reports/` |
| 7. Limites et perspectives | dataset de 2017, attaques inconnues, temps réel, SOC/SIEM | notebook 03, section 11 |
| Annexes | code, captures de l'app, guide d'installation | `app/`, README.md, `reports/figures/` |
