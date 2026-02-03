# Génération d'un PDF
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
from pandas.plotting import table
import numpy as np  
import tempfile
import subprocess


def generate_report(data, SettingData, PerformanceData, filtered_data, summary_results, movements_df, detailled_results):
    # Classe PDF déjà définie
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 8, "Rapport d'Analyse Cinématique du Mouvement des Mains", 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 7, title, 0, 1, 'L')
            self.ln(5)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 6, body)
            self.ln()

        def add_figure(self, fig, width=180):
            """Ajoute une figure matplotlib au PDF"""
            tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmpfile.name, bbox_inches='tight')
            tmpfile.close()
            self.image(tmpfile.name, w=width)
            plt.close(fig)

    # Création du PDF
    pdf = PDFReport()
    pdf.add_page()

    # Chapitre 1 : Identification
    pdf.chapter_title("Identité du Sujet")
    pdf.chapter_body(
        f"Nom : {data['name']} Prénom : {data['surname']}\n"
        f"Âge : {data['age']}\n"
        f"Pathologie : {data['pathology']}\n"
    )
    # Chapitre 2 : Paramètres
    pdf.chapter_title("Paramètres")
    pdf.chapter_body(
        f"Date : {PerformanceData['Target spawn time'][0].strftime('%d/%m/%Y')}\n"
        f"Paramètres : Gain Main Droite = {SettingData['Right hand gain'].iloc[0]}, Gain Main Gauche = {SettingData['Left hand gain'].iloc[0]}\n" 
        f"Durée : {round((filtered_data['Recording time'].iloc[-1] - filtered_data['Recording time'].iloc[0]).total_seconds())} s, Nombre de Cibles : {len(PerformanceData)}\n"
        f"Taille des cibles : {PerformanceData['Target radius (m)'].iloc[0]} m, Mode : {SettingData['Target selection mode'].iloc[0]}\n"
        f"Caractéristique de la matrice de cibles : Largeur = {SettingData['Wall angle (°)'].iloc[0]}°, Hauteur = {SettingData['Wall height (m)'].iloc[0]} m, Distance = {SettingData['Wall radius (m)'].iloc[0]} m, Position vertical =  {SettingData['Wall vertical position (m)'].iloc[0]} m\n"
    )
    #Chapitre 3 : Résultats Résumés
    pdf.chapter_title("Résultats")
    # intégration du tableau résumé
 
    # ----- Fonction pour tracer un groupe horizontal -----
    def plot_group(ax, df, title, xlabel):
        df_clean = df.replace('-', np.nan).astype(float)
        means = df_clean.iloc[:, [0, 2]]
        stds  = df_clean.iloc[:, [1, 3]]

        n_rows, n_conditions = means.shape
        ind = np.arange(n_rows)
        width = 0.3

        for i in range(n_conditions):
            ax.barh(
                ind + i*width,
                means.iloc[:, i],
                xerr=stds.iloc[:, i].fillna(0).values,
                height=width
            )

        ax.set_yticks(ind + width*(n_conditions-1)/2)
        ax.set_yticklabels(df_clean.index)
        #ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

    # ----- Créer la figure -----
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 2], hspace=0.25)

    # ----- Tableau en haut -----
    summary_results.columns = [
        "Main droite",
        "Écart-type",
        "Main gauche",
        "Écart-type"
    ]

    ax_table = fig.add_subplot(gs[0])
    ax_table.axis('off')
    tbl = table(ax_table, summary_results, loc='center', cellLoc='center')
    for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.3)
            cell.set_height(0.1)
            cell.set_edgecolor('gray')
            if row == 0 or col == -1:
                cell.set_fontsize(12)
                cell.set_text_props(weight='bold')
            else:
                cell.set_fontsize(10)



    # ----- Définir les groupes -----
    percent_rows = summary_results.index.str.contains('%')
    time_rows    = summary_results.index.str.contains('Temps')
    speed_rows   = summary_results.index.str.contains('Vitesse|Pic')
    smooth_rows = summary_results.index.str.contains('Fluidité')
    # ----- Sélection des 3 dernières lignes pour les ratios -----
    df_ratios = summary_results.iloc[-4:, :].copy()
    ratio_labels = ["Fluidité","Contrôle moteur", "Efficience", "Précision (m)"]

    # ----- Tracer les 3 premiers groupes (horizontal) -----
    groups = [
        ("Pourcentages", percent_rows, "%"),
        ("Temps", time_rows, "Secondes"),
        ("Vitesses", speed_rows, "m/s")
    ]

    for i, (title, mask, xlabel) in enumerate(groups, start=1):
        ax = fig.add_subplot(gs[i])
        df = summary_results[mask]
        if df.empty:
            ax.axis('off')
            continue
        plot_group(ax, df, title, xlabel)

    # ----- Tracer les ratios différemment (bar horizontal) -----
    ax = fig.add_subplot(gs[4])

    means = df_ratios.iloc[:, [0, 2]].astype(float)
    stds  = df_ratios.iloc[:, [1, 3]].astype(float)

    n_rows, n_conditions = means.shape
    ind = np.arange(n_rows)
    height = 0.3

    for i in range(n_conditions):
        ax.barh(
            ind + i*height,
            means.iloc[:, i],
            xerr=stds.iloc[:, i].values,
            height=height,
        )

    ax.set_yticks(ind + height*(n_conditions-1)/2)
    ax.set_yticklabels(ratio_labels)
    ax.set_title("")
    ax.set_xlabel("")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # 👉 Intégration dans le PDF
    pdf.add_figure(fig)

    # Chapitre 4 : Informations sur les variables
    # Bloc interprétation avec labels en gras
    pdf.set_font("Arial", size=11)

    items = [
        ("Cibles touchées :", " Pourcentage de cibles touchées avec la main droite (orange) et la main gauche (bleu). 0% = aucune cible touchée, 100% = toutes les cibles touchées."),
        ("Utilisation :", " Pourcentage d'utilisation de la main droite (orange) et main gauche (bleu). 0% = jamais choisie, 50% = moitié des cas, 100% = toujours choisie."),
        ("Temps de réaction :", " Nombre de secondes moyennes entre l'apparition de la cible et le début du mouvement."),
        ("Temps de mouvement :", " Nombre de secondes moyennes entre le début du mouvement et la fin du mouvement."),
        ("Vitesse moyenne :", " Vitesse moyenne (m/s) pendant le mouvement."),
        ("Vitesse de pic :", " Vitesse maximale (m/s) atteinte pendant le mouvement."),
        ("Fluidité :", " Ratio entre la vitesse moyenne et la vitesse de pic. Proche de 1 = mouvement fluide."),
        ("Contrôle moteur :", " Temps (s) nécessaire pour atteindre la vitesse de pic depuis le début du mouvement."),
        ("Efficience :", " Ratio entre la distance la plus courte et la distance réellement parcourue. Proche de 1 = efficient."),
        ("Précision :", " Distance moyenne entre le point d'arrivée et la cible. Plus faible = meilleure précision."),
    ]

    x = pdf.l_margin
    w = pdf.w - pdf.l_margin - pdf.r_margin
    line_height = 6
    padding = 3

    # --- calcul hauteur totale ---
    total_h = 2 * padding
    for label, desc in items:
        pdf.set_font("Arial", "B", 11)
        label_w = pdf.get_string_width(label) + 1

        pdf.set_font("Arial", "", 11)
        # on estime le texte complet sur la largeur restante
        remaining_w = w - 2*padding - label_w
        # split_only pour compter les lignes du descriptif
        lines = pdf.multi_cell(remaining_w, line_height, desc, split_only=True)
        total_h += max(1, len(lines)) * line_height

    # saut de page si besoin
    if pdf.get_y() + total_h > pdf.page_break_trigger:
        pdf.add_page()

    y = pdf.get_y()

    # --- dessiner box (fond + cadre) ---
    pdf.set_fill_color(240, 240, 240)
    pdf.set_draw_color(120, 120, 120)
    pdf.rect(x, y, w, total_h, style="DF")

    # --- écrire contenu ---
    pdf.set_xy(x + padding, y + padding)

    for label, desc in items:
        start_x = pdf.get_x()
        start_y = pdf.get_y()

        # label en gras
        pdf.set_font("Arial", "B", 11)
        pdf.cell(pdf.get_string_width(label) + 1, line_height, label, ln=0)

        # description en normal (sur la même ligne, puis retour ligne auto)
        pdf.set_font("Arial", "", 11)
        remaining_w = w - 2*padding - (pdf.get_x() - start_x)
        pdf.multi_cell(remaining_w, line_height, desc)

    pdf.ln(20)


    # Chapitre 5 : Positons des mains avec mouvements détectés
    #pdf.chapter_title("Positions des mains au cours du temps avec mouvements détectés")

    # Visualisation des positions filtrées avec indication des mouvements détectés
    # --- Préparation du temps et des signaux ---
    dt = 1 / 50
    time = np.arange(len(filtered_data)) * dt

    # Choisir quelle main et quelle coordonnée afficher 
    right_x = filtered_data['Current position[1].x (m)']
    left_x = filtered_data['Current position[2].x (m)']
    right_y = filtered_data['Current position[1].y (m)']
    left_y = filtered_data['Current position[2].y (m)']
    right_z = filtered_data['Current position[1].z (m)']
    left_z = filtered_data['Current position[2].z (m)']
    head_x = filtered_data['Current position[0].x (m)']
    head_y = filtered_data['Current position[0].y (m)']
    head_z = filtered_data['Current position[0].z (m)']

    # --- Tracé ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time, head_x, color='tab:purple', label='Tête X')
    axs[0].plot(time, right_x, color='tab:blue', label='Droite X')
    axs[0].plot(time, left_x, color='tab:orange', label='Gauche X')
    axs[0].set_ylabel('X (m)')
    axs[0].legend()

    axs[1].plot(time, head_y, color='tab:purple', label='Tête Y')
    axs[1].plot(time, right_y, color='tab:blue', label='Droite Y')
    axs[1].plot(time, left_y, color='tab:orange', label='Gauche Y')
    axs[1].set_ylabel('Y (m)')
    axs[1].legend()

    axs[2].plot(time, head_z, color='tab:purple', label='Tête Z')
    axs[2].plot(time, right_z, color='tab:blue', label='Droite Z')
    axs[2].plot(time, left_z, color='tab:orange', label='Gauche Z')
    axs[2].set_xlabel('Temps (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()

    # === Ajout des lignes verticales sur chaque subplot ===
    for _, row in movements_df.iterrows():
        color = 'tab:blue' if row['hand'] == 'Right' else 'tab:orange'
        for ax in axs:  # Ajoute la ligne à chaque subplot
            ax.axvline(row['t0_idx'] * dt, color=color, linestyle='--', alpha=0.6)
            ax.axvline(row['tf_idx'] * dt, color=color, linestyle='-', alpha=0.6)
            #ax.axvline(row['idx_start'] * dt, color='red', linestyle='--', alpha=0.3)
            #ax.axvline(row['idx_end'] * dt, color='red', linestyle='-', alpha=0.3)
            
    plt.suptitle('Position de la Main au cours du Temps -- Detection des Mouvements')
    #plt.xlabel('Temps (s)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    pdf.add_figure(fig)

    # Sauvegarde du PDF

    pdf_file = f"rapport_{data['name']}.pdf"
    pdf.output(pdf_file)
    subprocess.run(["open", pdf_file])