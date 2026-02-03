# Import necessary libraries

import numpy as np
import pandas as pd
from datetime import datetime
import glob
import re
import time
import os
import scipy.signal as signal

def run_import_process(data):
    """
    data : dictionnaire retourné par launch_gui()
    {
        "name": ...,
        "surname": ...,
        "age": ...,
        "pathology": ...,
        "folder_path": ...
    }
    """
    folder_path = data["folder_path"]

    pattern_1 = os.path.join(folder_path, "SettingsData_Simplified_*.csv")
    pattern_2 = os.path.join(folder_path, "TrackerData_Simplified_*.csv")
    pattern_3 = os.path.join(folder_path, "PerformanceData_Simplified_*.csv")

    file_1 = glob.glob(pattern_1)
    file_2 = glob.glob(pattern_2)
    file_3 = glob.glob(pattern_3)

    if len(file_1) != 1 or len(file_2) != 1 or len(file_3) != 1:
        raise ValueError("Nombre incorrect de fichiers trouvés")

    SettingData = pd.read_csv(file_1[0], sep=";")
    SettingData = SettingData.tail(-1)
    TrackerData = pd.read_csv(file_2[0], sep=";")
    PerformanceData = pd.read_csv(file_3[0], sep=";")

    #############################################################################################

    # Correction du nom des colonnes 'Start position Y(m)' de setting data

    def clean_col(name):
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)
        name = name.replace('Y(m)', 'Y (m).1')
        return name

    SettingData.columns = [clean_col(c) for c in SettingData.columns]

            #############################################################################################

    # Rééchantillonnage et filtrage des données de capture du mouvement
    # Paramètres

    Resampling_frequency = 50  # Hz

    # Fonctions utilitaires: rééchantillonnage basé sur pandas + interpolation temporelle
    def resample_df_pandas(df, time_col='Recording time', target_freq=50, time_format='%d-%m-%y %H:%M:%S.%f', dayfirst=True, target_state_col='Target states', missing_placeholder='-'):
        """
        Rééchantillonne df sur un index régulier basé sur time_col (pd.Timestamp).
        - target_freq en Hz (ex: 50).
        - time_format: (optionnel) format strptime pour accélérer et stabiliser le parsing.
        - dayfirst: (optionnel) si True, interprète les dates comme jour/mois.
        - target_state_col: nom de la colonne à remplir par nearest lookup (ex: 'Target states').
        - missing_placeholder: valeur indiquant l'absence (par défaut '-')

        Stratégie pour target_state_col: pour chaque timestamp rééchantillonné, on regarde la valeur
        précédente et la valeur suivante dans le fichier original; si l'une des deux n'est pas
        le placeholder on la choisit (préférence pour la plus proche en temps). Si les deux
        côtés ont une valeur non-placeholder identique, on prend cette valeur.

        Retourne un DataFrame rééchantillonné.
        """
        df_orig = df.copy()

        # 1) Parse time column — spécifier format si disponible réduit les warnings et accélère
        if time_format:
            df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce')
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce', dayfirst=dayfirst)
        # juste après la conversion
        #print("sample input strings:", df_orig[time_col].iloc[:5].tolist())
        #print("parsed (df):", df[time_col].head(5))
        #print("parsed dtype:", df[time_col].dtype)

        # Drop rows sans timestamp valide
        df = df.dropna(subset=[time_col])
        if df.empty:
            return df

        # 2) Convertir les dtypes "object" en types plus précis si possible
        df = df.infer_objects()

        # 3) Indexer sur la colonne temporelle
        df = df.set_index(time_col)

        # 4) Construire le nouvel index régulier
        freq_ms = int(round(1000 / target_freq))
        new_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=f"{freq_ms}ms")

        # 5) Reindexer sur l'union (conserver originaux), trier
        tmp = df.reindex(df.index.union(new_index)).sort_index()

        # 6) Interpoler uniquement les colonnes numériques (évite le FutureWarning)
        num_cols = tmp.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            tmp[num_cols] = tmp[num_cols].interpolate(method='time')

        # --- Traitement des colonnes non numériques ---
        non_num_cols = [c for c in tmp.columns if c not in num_cols]

        # Colonnes contenant '.Device name' : on peut propager la valeur (inchangée dans un fichier)
        device_cols = [c for c in non_num_cols if '.Device name' in c]
        if device_cols:
            tmp[device_cols] = tmp[device_cols].ffill().bfill()

        # Autres colonnes non num (sauf target_state_col) : fallback simple (propagation)
        other_non_num = [c for c in non_num_cols if c not in device_cols and c != target_state_col]
        if other_non_num:
            tmp[other_non_num] = tmp[other_non_num].ffill().bfill()

        # 7) Extraire les lignes correspondant au nouvel index et remettre le time_col en colonne
        res = tmp.reindex(new_index).reset_index().rename(columns={'index': time_col})

        # 8) Remplir target_state_col à partir du fichier original en prenant la valeur d'index la plus proche
        if target_state_col in df_orig.columns:
            # préparer les séries temporelles triées
            orig_times = df_orig[[time_col, target_state_col]].copy()
            orig_times[time_col] = pd.to_datetime(orig_times[time_col], format=time_format, errors='coerce')
            orig_times = orig_times.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

            if not orig_times.empty:
                # convertir en numpy datetime64 pour searchsorted
                orig_times_np = orig_times[time_col].values.astype('datetime64[ns]')
                res_times_np = res[time_col].values.astype('datetime64[ns]')
                orig_vals = orig_times[target_state_col].astype(str).fillna(missing_placeholder).values

                # positions where each res time would be inserted in orig_times
                pos = np.searchsorted(orig_times_np, res_times_np)

                fill_vals = []
                for k, p in enumerate(pos):
                    chosen = missing_placeholder
                    # indices possibles
                    prev_idx = p - 1
                    next_idx = p
                    prev_val = None
                    next_val = None
                    prev_dt = None
                    next_dt = None
                    if prev_idx >= 0:
                        prev_val = orig_vals[prev_idx]
                        prev_dt = orig_times_np[prev_idx]
                    if next_idx < len(orig_times_np):
                        next_val = orig_vals[next_idx]
                        next_dt = orig_times_np[next_idx]

                    # Cases
                    if prev_val is None and next_val is None:
                        chosen = missing_placeholder
                    elif prev_val is None:
                        chosen = next_val
                    elif next_val is None:
                        chosen = prev_val
                    else:
                        # si l'un des deux n'est pas placeholder, préférer cette valeur
                        prev_is_missing = (str(prev_val) == str(missing_placeholder))
                        next_is_missing = (str(next_val) == str(missing_placeholder))

                        if not prev_is_missing and next_is_missing:
                            chosen = prev_val
                        elif prev_is_missing and not next_is_missing:
                            chosen = next_val
                        elif (not prev_is_missing) and (not next_is_missing):
                            # si identiques -> prendre la valeur
                            if str(prev_val) == str(next_val):
                                chosen = prev_val
                            else:
                                # choisir le plus proche en temps
                                dt = res_times_np[k]
                                # distances
                                d_prev = np.abs((dt - prev_dt).astype('timedelta64[ns]'))
                                d_next = np.abs((next_dt - dt).astype('timedelta64[ns]'))
                                if d_prev <= d_next:
                                    chosen = prev_val
                                else:
                                    chosen = next_val
                        else:
                            # both are missing_placeholder
                            chosen = missing_placeholder
                    fill_vals.append(chosen)

                res[target_state_col] = pd.Series(fill_vals)

        # 9) Retourner le DataFrame rééchantillonné
        return res



    # Exécution du rééchantillonnage 
    resampled_data = resample_df_pandas(TrackerData, time_col='Recording time', target_freq=Resampling_frequency, time_format='%d-%m-%y %H:%M:%S.%f', dayfirst=True)
    #print(f"Données rééchantillonnées: {len(TrackerData)} -> {len(resampled_data)} lignes.")
            #############################################################################################
                
    # Filtre passe-bas pour les colonnes numériques
    # Utilise scipy.signal.butter + filtfilt pour un filtrage sans déphasage
    def butter_lowpass(cutoff, fs, order=4):
        """Retourne les coefficients (b, a) d'un filtre Butterworth passe-bas."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a


    def apply_lowpass_df(df, fs, cutoff=5, order=4, time_col='Recording time'):
        """Applique un filtre passe-bas (filtfilt) sur les colonnes numériques d'un DataFrame.
        - df : DataFrame rééchantillonné (doit contenir des colonnes numériques)
        - fs : fréquence d'échantillonnage en Hz (ex: 50)
        - cutoff : fréquence de coupure en Hz (ex: 5)
        - order : ordre du filtre Butterworth

        La fonction :
        - copie le DataFrame,
        - remplit les NaN des colonnes numériques par interpolation linéaire + ffill/bfill,
        - applique signal.filtfilt pour filtrage sans déphasage,
        - retourne le DataFrame filtré.
        """
        if not isinstance(df, pd.DataFrame):
            return df
        if df.empty:
            return df

        df_f = df.copy()
        # identifier colonnes numériques (exclure colonne temps si elle est numérique par hasard)
        num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return df_f

        # remplir NaN: interpolation puis forward/backfill
        try:
            df_f[num_cols] = df_f[num_cols].interpolate(method='linear', limit_direction='both')
        except Exception:
            df_f[num_cols] = df_f[num_cols].ffill().bfill()

        # si subsistent NaN, remplir par 0 pour éviter erreurs de filtfilt (optionnel selon contexte)
        df_f[num_cols] = df_f[num_cols].fillna(0)

        # construire filtre
        b, a = butter_lowpass(cutoff, fs, order=order)

        # appliquer filtfilt colonne par colonne
        for col in num_cols:
            try:
                df_f[col] = signal.filtfilt(b, a, df_f[col].values)
            except Exception as e:
                print(f"Warning: filtrage échoué pour colonne {col}: {e}")
                # en cas d'échec, garder les données non filtrées
                df_f[col] = df[col]

        return df_f

    # Appliquer le filtre à resampled_data (si disponible) et créer filterd_data
    # Paramètres du filtre
    Filter_cutoff = 5  # Hz

    if resampled_data is not None and not resampled_data.empty:
        filtered_data = apply_lowpass_df(resampled_data, fs=Resampling_frequency, cutoff=Filter_cutoff, order=4)
    else:
        print("Aucune donnée rééchantillonnée disponible pour le filtrage.")
    # Transformation des colonnes de temps en datetime pour faciliter les comparaisons avec tracker data
    PerformanceData['Target spawn time'] = pd.to_datetime(
        PerformanceData['Target spawn time'],
        format="%d-%m-%y %H:%M:%S.%f"
    )

    PerformanceData['Target kill time'] = pd.to_datetime(
        PerformanceData['Target kill time'],
        format="%d-%m-%y %H:%M:%S.%f"
    )
    return SettingData, PerformanceData, filtered_data