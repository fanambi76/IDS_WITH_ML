"""Démo : détection d'intrusions réseau par apprentissage automatique (CICIDS2017).

Lancer (depuis la racine du projet) :  streamlit run app/app.py
Prérequis : models/ids_model.joblib (détection) et, en option, models/ids_family_model.joblib
(type d'attaque), créés par le notebook 03.
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)

st.set_page_config(page_title="Détection d'intrusions par ML", page_icon="🛡️", layout="wide")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------- fonctions
@st.cache_resource
def load_bundle(src):
    """Charge le modèle sauvegardé (chemin ou fichier téléversé)."""
    return joblib.load(src)


def clean_label(s):
    s = re.sub(r"[^\x20-\x7E]", "-", str(s))
    return re.sub(r"\s+", " ", s).strip()


def prepare(df, feats):
    """Aligne le CSV sur les caractéristiques du modèle. Renvoie (X, df_valide, avertissements)."""
    notes = []
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} colonne(s) manquante(s) : {', '.join(missing[:8])}"
                         + (" ..." if len(missing) > 8 else ""))
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = X.isna().any(axis=1)
    if bad.any():
        notes.append(f"{int(bad.sum())} ligne(s) ignorée(s) (valeurs manquantes ou infinies).")
    ok = ~bad
    return X[ok].to_numpy("float32"), df[ok].reset_index(drop=True), notes


def score(bundle, X):
    """Renvoie un score entre 0 (normal) et 1 (suspect)."""
    if "model" in bundle:
        return bundle["model"].predict_proba(X)[:, 1], "probabilité d'attaque"
    sc, iso = bundle["scaler_iso"]                     # Isolation Forest
    an = -iso.score_samples(sc.transform(X))
    return 1 / (1 + np.exp(-15 * (an + iso.offset_))), "indice d'anomalie (non calibré)"


def to_family(l):
    """Même regroupement que dans les notebooks (étiquette CICIDS2017 -> famille)."""
    if l == "BENIGN": return "Benign"
    if l.startswith("DoS") or l in ("DDoS", "Heartbleed"): return "DoS/DDoS"
    if l == "PortScan": return "PortScan"
    if l in ("FTP-Patator", "SSH-Patator"): return "BruteForce"
    if l.startswith("Web Attack"): return "WebAttack"
    if l == "Bot": return "Botnet"
    if l == "Infiltration": return "Infiltration"
    return "Autre"


def predict_family(fbundle, d, mask, min_conf):
    """Nomme la famille des flux suspects (mask). Renvoie (famille, confiance en %)."""
    fam = np.array(["—"] * len(d), dtype=object)
    fconf = np.full(len(d), np.nan)
    if not mask.any():
        return fam, fconf
    Xf = (d.loc[mask, fbundle["features"]].apply(pd.to_numeric, errors="coerce")
          .replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy("float32"))
    proba = fbundle["model"].predict_proba(Xf)
    best = proba.argmax(axis=1)
    names = np.array(fbundle["model"].classes_, dtype=object)[best]
    c = proba.max(axis=1)
    names = np.where(c >= min_conf, names, "Indéterminée")
    fam[mask] = names
    fconf[mask] = (c * 100).round(1)
    return fam, fconf


def confidence_level(conf):
    return np.where(conf >= 0.9, "Élevée", np.where(conf >= 0.7, "Moyenne", "Faible"))


# ---------------------------------------------------------------- barre latérale
st.sidebar.header("Paramètres")
model_file = st.sidebar.file_uploader("Modèle (.joblib)", type="joblib",
                                      help="Laisse vide pour utiliser models/ids_model.joblib.")
threshold = st.sidebar.slider("Seuil de détection", 0.05, 0.95, 0.50, 0.05,
                              help="Plus le seuil est BAS, plus on détecte d'attaques "
                                   "(rappel ↑) mais plus on a de fausses alertes (précision ↓).")
family_file = st.sidebar.file_uploader("Modèle de famille d'attaque (.joblib, optionnel)", type="joblib",
                                       help="Laisse vide pour utiliser models/ids_family_model.joblib.")
min_family_conf = st.sidebar.slider("Confiance minimale pour nommer la famille", 0.30, 0.90, 0.60, 0.05,
                                    help="Sous ce seuil, la famille est affichée « Indéterminée » : "
                                         "le modèle ne la connaît peut-être pas.")
st.sidebar.caption("Seuil bas = sécurité maximale, plus de fausses alertes.\n\n"
                   "Seuil haut = moins d'alertes, mais des attaques peuvent passer.")

try:
    bundle = load_bundle(model_file if model_file is not None else str(MODELS_DIR / "ids_model.joblib"))
except FileNotFoundError:
    st.error("Modèle introuvable : lance d'abord le notebook 03 (il crée `models/ids_model.joblib`) "
             "ou téléverse le fichier à gauche.")
    st.stop()
fbundle = None
try:
    if family_file is not None:
        fbundle = load_bundle(family_file)
    elif (MODELS_DIR / "ids_family_model.joblib").exists():
        fbundle = load_bundle(str(MODELS_DIR / "ids_family_model.joblib"))
except Exception as e:
    st.sidebar.warning(f"Modèle de famille inutilisable : {e}")
feats = bundle["features"]
model_name = type(bundle["model"]).__name__ if "model" in bundle else "IsolationForest"
st.sidebar.success(f"Modèle chargé : {model_name}\n\n{len(feats)} caractéristiques")
if fbundle is None:
    st.sidebar.info("Pas de modèle de famille : seuls les verdicts normal/suspect sont affichés.")
else:
    st.sidebar.success(f"Familles connues : {', '.join(fbundle['model'].classes_)}")

# ---------------------------------------------------------------- page principale
st.title("🛡️ Détection d'intrusions réseau par apprentissage automatique")
st.write("Charge un fichier CSV de flux réseau (format CICIDS2017). Chaque flux est classé "
         "**normal** ou **suspect**, avec un niveau de confiance.")

up = st.file_uploader("Fichier CSV de trafic", type="csv")
if up is None:
    st.info("En attente d'un fichier. Pas de CSV sous la main ? Génère-en un avec `make_demo_csv.py`.")
    with st.expander("Colonnes attendues par le modèle"):
        st.write(", ".join(feats))
    st.stop()

try:
    raw = pd.read_csv(up, encoding="latin-1", low_memory=False)
    X, d, notes = prepare(raw, feats)
except Exception as e:
    st.error(f"Fichier non utilisable : {e}")
    st.stop()
for n in notes:
    st.warning(n)
if len(d) == 0:
    st.error("Aucune ligne exploitable dans ce fichier.")
    st.stop()

s, s_name = score(bundle, X)
pred = (s >= threshold).astype(int)
conf = np.where(pred == 1, s, 1 - s)          # confiance dans le verdict rendu
res = d.copy()
res.insert(0, "verdict", np.where(pred == 1, "SUSPECT", "normal"))
res.insert(1, "score", s.round(4))
res.insert(2, "confiance", (conf * 100).round(1))
res.insert(3, "niveau_confiance", confidence_level(conf))
if fbundle is not None:
    fam, fconf = predict_family(fbundle, d, pred == 1, min_family_conf)
    res.insert(4, "type_attaque_probable", fam)
    res.insert(5, "confiance_type", fconf)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flux analysés", f"{len(res):,}")
c2.metric("Flux suspects", f"{int(pred.sum()):,}")
c3.metric("Part de suspects", f"{pred.mean():.1%}")
c4.metric("Confiance moyenne", f"{conf.mean():.1%}")
st.caption(f"Score affiché : **{s_name}**, entre 0 (normal) et 1 (suspect).")

tab1, tab2, tab3 = st.tabs(["📊 Synthèse", "🔎 Flux suspects", "✅ Évaluation"])

with tab1:
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.hist(s, bins=30, color="tab:blue")
        ax.axvline(threshold, color="red", ls="--", label=f"seuil = {threshold:.2f}")
        ax.set_xlabel("Score (0 = normal, 1 = suspect)"); ax.set_ylabel("Nombre de flux")
        ax.set_yscale("log"); ax.legend()
        st.pyplot(fig)
    with right:
        cnt = res["verdict"].value_counts().reindex(["normal", "SUSPECT"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(cnt.index, cnt.values, color=["tab:green", "tab:red"])
        ax.set_ylabel("Nombre de flux"); ax.set_title("Verdicts")
        st.pyplot(fig)
    if fbundle is not None and pred.sum() > 0:
        st.write("**Types d'attaque probables parmi les flux suspects**")
        st.bar_chart(res.loc[res.verdict == "SUSPECT", "type_attaque_probable"].value_counts())
    st.write("**Niveau de confiance des verdicts**")
    st.bar_chart(res["niveau_confiance"].value_counts().reindex(["Élevée", "Moyenne", "Faible"]).fillna(0))
    st.caption("Les verdicts à confiance **faible** sont ceux qu'un analyste devrait vérifier en priorité.")

with tab2:
    sus = res[res.verdict == "SUSPECT"].sort_values("score", ascending=False)
    only_low = st.checkbox("Afficher seulement les verdicts à confiance faible ou moyenne")
    if only_low:
        sus = sus[sus.niveau_confiance != "Élevée"]
    st.write(f"{len(sus):,} flux affichables (les 500 premiers sont montrés).")
    show = [c for c in ["verdict", "score", "confiance", "niveau_confiance",
                        "type_attaque_probable", "confiance_type", "Destination Port",
                        "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Label"] if c in sus.columns]
    st.dataframe(sus[show].head(500))
    st.download_button("⬇️ Télécharger tous les résultats (CSV)",
                       res.to_csv(index=False).encode("utf-8"), "resultats_detection.csv", "text/csv")

with tab3:
    if "Label" not in res.columns:
        st.info("Ton CSV n'a pas de colonne `Label` : l'évaluation n'est pas possible. "
                "Avec un CSV étiqueté, tu verras la matrice de confusion et les métriques ici.")
    else:
        lab = res["Label"].map(clean_label)
        y_true = (lab != "BENIGN").astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Précision", f"{precision_score(y_true, pred, zero_division=0):.3f}")
        m2.metric("Rappel", f"{recall_score(y_true, pred, zero_division=0):.3f}")
        m3.metric("F1", f"{f1_score(y_true, pred, zero_division=0):.3f}")
        m4.metric("Fausses alertes / attaques ratées", f"{fp:,} / {fn:,}")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ConfusionMatrixDisplay(np.array([[tn, fp], [fn, tp]]), display_labels=["normal", "attaque"]).plot(ax=ax, colorbar=False)
        st.pyplot(fig)
        att = pd.DataFrame({"Label": lab, "détecté": pred})[y_true == 1]
        if len(att):
            st.write("**Part détectée par type d'attaque**")
            st.dataframe(att.groupby("Label")["détecté"].agg(["mean", "size"])
                         .rename(columns={"mean": "taux de détection", "size": "nb de flux"})
                         .sort_values("taux de détection").round(3))

        if fbundle is not None:
            st.write("---")
            st.write("**Type d'attaque : prédit contre réel** (attaques réellement détectées comme suspectes)")
            true_fam = lab.map(to_family)
            both = (y_true == 1) & (pred == 1)
            if both.any():
                tab = pd.crosstab(true_fam[both].rename("famille réelle"),
                                  res.loc[both, "type_attaque_probable"].rename("famille prédite"))
                named = both & (res["type_attaque_probable"] != "Indéterminée")
                acc = (true_fam[named] == res.loc[named, "type_attaque_probable"]).mean() if named.any() else float("nan")
                k1, k2 = st.columns(2)
                k1.metric("Famille correcte (flux nommés)", f"{acc:.1%}")
                k2.metric("Flux « Indéterminée »", f"{int((both & ~named).sum()):,}")
                st.dataframe(tab)
                st.caption("Une famille réelle absente des familles connues du modèle apparaîtra forcément "
                           "mal classée ou « Indéterminée » : c'est la limite du modèle supervisé.")
            else:
                st.info("Aucune attaque détectée : rien à comparer.")
