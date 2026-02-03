from matplotlib.pylab import det
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_kinematic_analysis(SettingData, PerformanceData, filtered_data):
    """
    Effectue l'analyse cinématique des données importées.
    
    Parameters:
    - SettingData: DataFrame des paramètres de configuration.
    - PerformanceData: DataFrame des données de performance.
    - filtered_data: DataFrame des données de position filtrées.
    
    Returns:
    - movements_df: DataFrame des mouvements détectés et leurs paramètres cinématiques.
    """

    # Détection du début et de la fin des mouvements basés sur la vitesse et la distance parcourue
    def detect_reaching_movements(df, fps=50, vel_threshold=0.2, min_duration=0.2):
        dt = 1 / fps
        min_frames = int(min_duration * fps) # voir si toujours utile
        
        # Données temporelles (horodatage apparition et disparition des cibles)
        target_time = PerformanceData[['Target spawn time', 'Target kill time']]
        #print(target_time)
        #print(df['Recording time'])
        
        # Donnees de position
        # Colonnes pour chaque main
        cols_right = ['Recording time', 'Current position[1].x (m)', 'Current position[1].y (m)', 'Current position[1].z (m)']
        cols_left  = ['Recording time', 'Current position[2].x (m)', 'Current position[2].y (m)', 'Current position[2].z (m)']
        # Positions de la main
        pos_right = df[cols_right].to_numpy(dtype=float)
        pos_left  = df[cols_left].to_numpy(dtype=float)

        # Positions de départ
        start_pos_right = SettingData[['Start position X (m).1','Start position Y (m).1','Start position Z (m).1']].to_numpy()
        start_pos_left  = SettingData[['Start position X (m)','Start position Y (m)','Start position Z (m)']].to_numpy()

        # Fonction de calcul de la vitesse
        def compute_speed(x, y, z):
            vx = np.gradient(x, dt)
            vy = np.gradient(y, dt)
            vz = np.gradient(z, dt)
            return np.sqrt(vx**2 + vy**2 + vz**2)

        speed_right = compute_speed(df[cols_right[1]], df[cols_right[2]], df[cols_right[3]])
        speed_left  = compute_speed(df[cols_left[1]],  df[cols_left[2]],  df[cols_left[3]])


        
        def detect_bursts(speed, pos, hand_label):
            movements = []
            
            # Définition de la zone de départ en fonction de la main
            start_zone = start_pos_right if hand_label == "Right" else start_pos_left

            
            # Définition du gain en fonction de la main
            gain = float(SettingData['Right hand gain'].iloc[0]) if hand_label == "Right" else float(SettingData['Left hand gain'].iloc[0])
            
            for i, (spawn_time, kill_time) in enumerate(zip(target_time['Target spawn time'], target_time['Target kill time'])):
                
                # Position de la cible
                target_pos_x = PerformanceData.loc[len(movements)]['x (m)']
                target_pos_y = PerformanceData.loc[len(movements)]['y (m)']
                target_pos_z = PerformanceData.loc[len(movements)]['z (m)']
                target_pos = np.array([target_pos_x, target_pos_y, target_pos_z])
            
                # Fenêtre temporelle de la cible
                idx_start = np.searchsorted(df['Recording time'], spawn_time)
                idx_end   = np.searchsorted(df['Recording time'], kill_time)
                
                if idx_end <= idx_start:
                    continue

                # Vitesse dans la fenêtre
                sub_speed = speed[idx_start:idx_end]
                
                # Détection du mouvement
                moving = sub_speed > vel_threshold
                if not np.any(moving):
                    movements.append({
                        "hand": hand_label,
                        "target_id": i + 1,
                        "status": "no movement",
                        "idx_start": idx_start,
                        "idx_end": idx_end,
                        "t0_idx": np.nan,
                        "tf_idx": np.nan,
                        "reaction_time_(s)": np.nan,
                        "mvt_time_(s)": np.nan,
                        "mean_speed_(m/s)": np.nan,
                        "peak_speed_(m/s)": np.nan,
                        "Ratio_peak_mean_speed": np.nan,
                        "Time to velocity peak_(s)": np.nan,
                        "min_distance_(euclidian_dist)_(m)": np.nan,
                        "gain" : gain,
                        "VR_distance_(euclidian_dist)_(m)" : np.nan,
                        "trajectory_length_(m)" : np.nan,
                        "Efficiency" : np.nan,
                        "target_pos_x_(m)" : target_pos_x,
                        "target_pos_y_(m)" : target_pos_y,
                        "target_pos_z_(m)" : target_pos_z,
                        "distance_startZone_taget_(m)": np.linalg.norm(start_zone - target_pos),
                        "VR_pos_x_(m)" : np.nan,
                        "VR_pos_y_(m)" : np.nan,
                        "VR_pos_z_(m)" : np.nan,
                        "Accuracy" : np.nan
                    })
                    continue
                
                # t0 = premier point où la vitesse dépasse le seuil
                t0_rel = np.argmax(moving)
                t0_idx = idx_start + t0_rel
                
                valid_t0 = t0_idx
                
                # Récupération de la position de la plus éloignée de t0 dans la fenêtre (fin du mouvement)
                
                # Distance depuis le point de départ
                distances = np.sqrt(((pos[valid_t0:idx_end + 50, 1:] - pos[valid_t0, 1:]) ** 2).sum(axis=1)) # index +50 pour s'assurer de capturer le pic même lorsque la cible est atteinte juste avant la fin du mouvement
                
                # tf = point le plus éloigné dans la fenêtre
                rel_tf = np.argmax(distances)
                tf_new = valid_t0 + rel_tf
                
                if tf_new - valid_t0 < min_frames:
                    continue
                
                # Calcul de la Trajectoire réelle
                trajectory = np.sum(np.sqrt(np.sum(np.diff(pos[valid_t0:tf_new+1, 1:], axis=0)**2, axis=1)))
                
                # Position de la main virtuelle
                #VR_pos = pos[valid_t0:tf_new+1, 1:] * gain # A CORRIGER
                start = pos[valid_t0, 1:]                       # position de départ (exacte)
                segment = pos[valid_t0:tf_new+1, 1:]            # positions originales
                deltas_from_start = segment - start            # déplacement relatif à la position initiale

                VR_pos = start + deltas_from_start * gain      # start non modifié, déplacements mis à l'échelle

                #accuracy = np.linalg.norm(VR_pos[-1] - target_pos)
                
                # # Position de la cible
                target_pos_x = PerformanceData.loc[len(movements)]['x (m)']
                target_pos_y = PerformanceData.loc[len(movements)]['y (m)']
                target_pos_z = PerformanceData.loc[len(movements)]['z (m)']
                target_pos = np.array([target_pos_x, target_pos_y, target_pos_z])

                
                # Vitesse de mouvement
                segment_speed = speed[valid_t0:tf_new]
                peak_speed = segment_speed.max()
                
                # Temps de réaction
                reaction_time = (valid_t0 - idx_start) * dt
                
                # Temps de mouvement
                duration = (tf_new - valid_t0) * dt
                
                # Constitution du dictionnaire de résultats
                movements.append({
                    "hand": hand_label,
                    "target_id": i + 1,
                    "status": "movement detected",
                    "idx_start": idx_start,
                    "idx_end": idx_end,
                    "t0_idx": valid_t0,
                    "tf_idx": tf_new,
                    "reaction_time_(s)": reaction_time,
                    "mvt_time_(s)": duration,
                    "mean_speed_(m/s)": np.mean(segment_speed),
                    "peak_speed_(m/s)": peak_speed,
                    "Ratio_peak_mean_speed": np.mean(segment_speed) / peak_speed if peak_speed > 0 else np.nan, # Mesure de la fluidité (los Reyes-Guzmán et al., 2014)
                    "Time to velocity peak_(s)": (np.argmax(segment_speed) * dt), # Mesure du controle moteur (los Reyes-Guzmán et al., 2014)
                    "min_distance_(euclidian_dist)_(m)": distances[rel_tf],
                    "gain" : gain,
                    "VR_distance_(euclidian_dist)_(m)" : distances[rel_tf] * gain,
                    "trajectory_length_(m)" : trajectory,
                    "Efficiency" : distances[rel_tf] * gain / trajectory if trajectory > 0 else np.nan,
                    "target_pos_x_(m)" : target_pos_x,
                    "target_pos_y_(m)" : target_pos_y,
                    "target_pos_z_(m)" : target_pos_z,
                    "distance_startZone_taget_(m)": np.linalg.norm(start_zone - target_pos),
                    "VR_pos_x_(m)" : VR_pos[-1][0],
                    "VR_pos_y_(m)" : VR_pos[-1][1],
                    "VR_pos_z_(m)" : VR_pos[-1][2],
                    "Accuracy" : np.linalg.norm(VR_pos[-1] - target_pos)
                })
                
            return movements
                
                
            
        # Mesure des paramètres cinématiques
        moves_right = detect_bursts(speed_right, pos_right, "Right")
        moves_left  = detect_bursts(speed_left, pos_left, "Left")
        
        return pd.DataFrame(moves_right + moves_left)
                                                                
                                                                
    
    # Application de de la fonction de detection des mouvements
    movements_df = detect_reaching_movements(filtered_data, fps=50, vel_threshold=0.2, min_duration=0.1)

    # On ne garde que les mouvements validés
    movements_df = movements_df[movements_df["status"] == "movement detected"].copy()
    # Vérifier si une cible a été détectée par les deux mains
    if movements_df.duplicated(subset=['target_id']).any():
        # Pour chaque target_id, garder la détection avec la meilleure précision
        best_idx = (
        movements_df
        .groupby('target_id')['Accuracy'] #  mvt_time_(s)
        .idxmin()
            )
        # Garder uniquement les détections optimales
    movements_df = movements_df.loc[best_idx].reset_index(drop=True)

    # Affichage des résultats par cible
    detailled_results = movements_df[['target_id', 'hand', 'reaction_time_(s)', 'mvt_time_(s)', 'mean_speed_(m/s)', 'peak_speed_(m/s)', 'Ratio_peak_mean_speed', 'Time to velocity peak_(s)', 'Efficiency', 'Accuracy',]]
    detailled_results.columns = ['Target ID', 'Main utilié', 'Temps de réaction (s)', 'Temps de mouvemnt (s)', 'Vitesse moyenne (m/s)', 'Pic de vitesse (m/s)', 'Fluidité', 'Contrôle moteur', 'Efficience', 'Précision (m)']
    #detailled_results['Cible touchée ?'] = PerformanceData['Target has been hit']
    detailled_results.insert(2, 'Cible touchée ?', PerformanceData['Target has been hit'].values)


    # Résultats résumés par main
    Right_results = detailled_results[detailled_results['Main utilié'] == 'Right']
    Left_results = detailled_results[detailled_results['Main utilié'] == 'Left']

    Right_summary_results = Right_results.agg({
        'Temps de réaction (s)': ['mean', 'std'],
        'Temps de mouvemnt (s)': ['mean', 'std'],
        'Vitesse moyenne (m/s)': ['mean', 'std'],
        'Pic de vitesse (m/s)': ['mean', 'std'],
        'Fluidité': ['mean', 'std'],
        'Contrôle moteur': ['mean', 'std'],
        'Efficience': ['mean', 'std'],
        'Précision (m)': ['mean', 'std']
    }).T
    Right_summary_results.columns = ['Valeur', 'Dispersion']

    Right_summary_results.loc['Utilisation (%)'] = [len(Right_results) / len(detailled_results) * 100,np.nan]
    Right_summary_results.loc['Cibles touchées (%)'] = [Right_results['Cible touchée ?'].mean() * 100,np.nan]
    order = ['Utilisation (%)','Cibles touchées (%)']
    Right_summary_results = pd.concat([Right_summary_results.loc[order],Right_summary_results.drop(order)])


    Left_summary_results = Left_results.agg({
        'Temps de réaction (s)': ['mean', 'std'],
        'Temps de mouvemnt (s)': ['mean', 'std'],
        'Vitesse moyenne (m/s)': ['mean', 'std'],
        'Pic de vitesse (m/s)': ['mean', 'std'],
        'Fluidité': ['mean', 'std'],
        'Contrôle moteur': ['mean', 'std'],
        'Efficience': ['mean', 'std'],
        'Précision (m)': ['mean', 'std']
    }).T
    Left_summary_results.columns = ['Valeur', 'Dispersion']
    Left_summary_results.loc['Utilisation (%)'] = [len(Left_results) / len(detailled_results) * 100,np.nan]
    Left_summary_results.loc['Cibles touchées (%)'] = [Left_results['Cible touchée ?'].mean() * 100,np.nan]
    Left_summary_results = pd.concat([Left_summary_results.loc[order],Left_summary_results.drop(order)])

    summary_results = pd.concat(
        {
            "Main droite": Right_summary_results,
            "Main gauche": Left_summary_results
        },
        axis=1
    )
    summary_results = summary_results.round(2)
    summary_results = summary_results.fillna('-')
    print(summary_results)


    return summary_results, detailled_results, movements_df