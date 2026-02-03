# gui_test.py
from tkinter import Tk, StringVar, Label, Entry, Button, filedialog, messagebox

def launch_gui():
    subject_data = {}

    def select_folder():
        folder = filedialog.askdirectory(title="Sélectionner le dossier CSV")
        if folder:
            folder_path_var.set(folder)

    def validate():
        nonlocal subject_data
        subject_data = {
            "name": name_var.get(),
            "surname": surname_var.get(),
            "age": age_var.get(),
            "pathology": pathology_var.get(),
            "folder_path": folder_path_var.get()
        }
        if not all(subject_data.values()):
            messagebox.showerror("Erreur", "Tous les champs doivent être remplis.")
            subject_data = {}
            return
        root.quit()  # sortir de mainloop

    root = Tk()
    root.title("Infos Sujet")
    root.geometry("450x250")

    name_var = StringVar()
    surname_var = StringVar()
    age_var = StringVar()
    pathology_var = StringVar()
    folder_path_var = StringVar()

    Label(root, text="Nom :").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    Entry(root, textvariable=name_var).grid(row=0, column=1, padx=5, pady=5)
    Label(root, text="Prénom :").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    Entry(root, textvariable=surname_var).grid(row=1, column=1, padx=5, pady=5)
    Label(root, text="Âge :").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    Entry(root, textvariable=age_var).grid(row=2, column=1, padx=5, pady=5)
    Label(root, text="Pathologie :").grid(row=3, column=0, sticky="w", padx=5, pady=5)
    Entry(root, textvariable=pathology_var).grid(row=3, column=1, padx=5, pady=5)
    Label(root, text="Dossier CSV :").grid(row=4, column=0, sticky="w", padx=5, pady=5)
    Entry(root, textvariable=folder_path_var, width=30).grid(row=4, column=1, padx=5, pady=5)
    Button(root, text="Parcourir", command=select_folder).grid(row=4, column=2, padx=5)
    Button(root, text="Valider", command=validate).grid(row=5, column=1, pady=15)

    root.mainloop()
    root.destroy()
    return subject_data