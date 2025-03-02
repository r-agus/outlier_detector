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
    Genera un reporte en LaTeX que incluye:
      - Una sección resumen con una tabla (con pocas columnas clave).
      - Una sección de detalles: para cada experimento se crea una página con la imagen generada y el informe detallado.
    """
    out_dir = "out"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("% Reporte generado automáticamente\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\usepackage{float}\n")
        f.write("\\geometry{a4paper, margin=1in}\n")
        f.write("\\begin{document}\n")
        
        # Sección resumen con tabla de resultados (columnas clave)
        f.write("\\section*{Resumen de Resultados}\n")
        summary_columns = ["Test", "Dataset", "Modelo", "Tiempo Entrenamiento (s)", "AUC-ROC"]
        df_summary = df[summary_columns]
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(df_summary.to_latex(index=False, caption=None))
        f.write("\\caption{Resumen de resultados}\n")
        f.write("\\label{tab:resumen}\n")
        f.write("\\end{table}\n")
        f.write("\n\\newpage\n")
        
        # Sección de detalles: una página por cada experimento
        f.write("\\section*{Detalles por Experimento}\n")
        for idx, row in df.iterrows():
            f.write("\\subsection*{Test: %s -- Dataset: %s -- Modelo: %s}\n" % (row["Test"], row["Dataset"], row["Modelo"]))
            f.write("\\textbf{Tiempo de entrenamiento (s):} %s \\\\ \n" % row["Tiempo Entrenamiento (s)"])
            f.write("\\textbf{AUC-ROC:} %s \\\\ \n" % row["AUC-ROC"])
            
            f.write("\\begin{center}\n")

            figura_path = "../" + row["Figura"].replace("\\", "/")
            f.write("\\includegraphics[width=0.6\\textwidth]{%s}\n" % figura_path)
            f.write("\\end{center}\n")
            f.write("\\vspace{0.5cm}\n")

            f.write("\\begin{center}\n")
            cm_path = "../" + row["Matriz Confusion"].replace("\\", "/")
            f.write("\\includegraphics[width=0.52\\textwidth]{%s}\n" % cm_path)
            f.write("\\end{center}\n")

            f.write("\\textbf{Reporte detallado:}\\\\\n")
            f.write("\\begin{verbatim}\n")
            f.write(row["Detalle"])
            f.write("\n\\end{verbatim}\n")

            f.write("\\newpage\n")
        f.write("\\end{document}\n")
    print(f"✅ Reporte LaTeX generado en '{output_file}'")

if __name__ == "__main__":
    df_experimentos = read_experiment_files("experiments")
    if df_experimentos.empty:
        print("No se encontraron archivos de experimentos en la carpeta 'experiments'.")
    else:
        generate_latex_report(df_experimentos, "report.tex")
