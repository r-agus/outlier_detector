# generate_report.py

import os
import pandas as pd

def read_experiment_files(folder="experiments"):
    """
    Lee todos los archivos CSV en la carpeta 'folder' y los concatena en un DataFrame.
    """
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_latex_report(df, output_file="report.tex"):
    """
    Genera un reporte en LaTeX a partir del DataFrame y lo guarda en 'output_file'.
    """
    out_dir = "out"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, output_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Reporte generado automáticamente\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[landscape, left=1cm, right=1cm]{geometry}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\begin{document}\n")
        f.write("\\section*{Reporte de Experimentos}\n")
        f.write(df.to_latex(index=False, caption='Resultados de Experimentos', label='tab:resultados', longtable=True))
        f.write("\n\\end{document}\n")
    print(f"✅ Reporte LaTeX generado en '{output_file}'")

if __name__ == "__main__":
    df_experimentos = read_experiment_files("experiments")
    if df_experimentos.empty:
        print("No se encontraron archivos de experimentos en la carpeta 'experiments'.")
    else:
        generate_latex_report(df_experimentos, "report.tex")
