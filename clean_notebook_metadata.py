import nbformat

# Ruta del archivo de entrada y salida
input_path = "Salary_Prediction.ipynb"
output_path = "Salary_Prediction_CLEAN.ipynb"

# Cargar notebook
with open(input_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Eliminar metadatos globales
nb.metadata = {}

# Eliminar metadatos por celda
for cell in nb.cells:
    cell.metadata = {}

# Guardar notebook limpio
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook cleaned and saved to {output_path}")
