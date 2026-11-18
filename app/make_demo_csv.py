"""Crée un petit CSV de démonstration à partir du jeu nettoyé.

Usage (depuis la racine) :  python app/make_demo_csv.py [--src data/processed/cicids2017_clean.parquet] [--n 600] [--attack-share 0.25] [--no-label]
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--src", default=str(ROOT / "data/processed/cicids2017_clean.parquet"))
p.add_argument("--n", type=int, default=600)
p.add_argument("--attack-share", type=float, default=0.25)
p.add_argument("--day", default="Friday", help="jour à échantillonner (Friday = attaques inconnues du modèle)")
p.add_argument("--no-label", action="store_true", help="retire la colonne Label (cas réel : trafic non étiqueté)")
p.add_argument("--out", default=str(ROOT / "data/demo/demo_trafic.csv"))
a = p.parse_args()

df = pd.read_parquet(a.src)
df = df[df.source_file == a.day] if (df.source_file == a.day).any() else df
n_att = int(a.n * a.attack_share)
att = df[df.is_attack == 1].sample(min(n_att, int((df.is_attack == 1).sum())), random_state=1)
ben = df[df.is_attack == 0].sample(min(a.n - len(att), int((df.is_attack == 0).sum())), random_state=1)
out = pd.concat([att, ben]).sample(frac=1, random_state=1).drop(columns=["source_file", "is_attack", "family"])
if a.no_label:
    out = out.drop(columns=["Label"])
out.to_csv(a.out, index=False)
print(f"{a.out} : {len(out)} flux dont {len(att)} attaques" + (" (sans Label)" if a.no_label else ""))
