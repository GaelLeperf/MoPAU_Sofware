# MoPAU_software

[![Licence](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Description
-----------
MoPAU_Software est un petit outil Python d'importation, d'analyse cinématique et de génération de rapports PDF à partir de fichiers CSV produits par le système MoPAU.  
Le projet fournit :
- une interface graphique minimale pour saisir les informations d'un sujet et sélectionner le dossier contenant les CSV (`gui.py`),
- un module d'import/traitement et de filtrage des données (`Import_files.py`),
- une analyse cinématique et détection de mouvements (`Analysis.py`),
- une génération de rapport PDF synthétique (`report.py`),
- un point d'entrée `main.py` qui orchestre le workflow.

Prérequis
---------
- Python 3.10+ (testé sur Python 3.10/3.11)
- Système : macOS 
- Bibliothèques Python (voir `requirements.txt`) :
  - numpy, pandas, scipy, matplotlib, fpdf, tkinter (inclus avec la plupart des distributions Python), etc.

Installation
------------
1. Cloner le dépôt :
```bash
git clone https://github.com/GaelLeperf/MoPAU_software.git
cd MoPAU_software
```

2. Créer un environnement virtuel et installer les dépendances :
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Utilisation
-----------
1. Préparer le dossier CSV attendu :
   - Le dossier doit contenir exactement un fichier de chaque type :
     - `SettingsData_Simplified_*.csv`
     - `TrackerData_Simplified_*.csv`
     - `PerformanceData_Simplified_*.csv`
   - Les noms de colonnes et les formats temporels attendus sont ceux présents dans les scripts (`Recording time`, `Target spawn time`, `Target kill time`, etc.).

2. Lancer le programme :
```bash
python main.py
```
- Une petite fenêtre s'ouvre pour saisir Nom / Prénom / Âge / Pathologie et sélectionner le dossier CSV.
- À la fin, un fichier `rapport_<Nom>.pdf` est généré et (selon plate-forme) ouvert automatiquement.

Sorties
-------
- rapport_<Nom>.pdf — rapport PDF récapitulatif contenant :
  - identité du sujet,
  - paramètres de la session,
  - tableaux et graphiques de synthèse,
  - visualisations des trajectoires et détections de mouvements.
