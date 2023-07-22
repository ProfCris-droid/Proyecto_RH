import pandas as pd

def guardar_csv_como_txt(archivo_csv, archivo_txt):
    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(archivo_csv)

    # Convertir el DataFrame en una cadena con formato de tabla
    tabla_formateada = df.to_string(index=False)

    # Guardar la tabla formateada en un archivo .txt
    with open(archivo_txt, 'w') as file:
        file.write(tabla_formateada)

# Nombre del archivo CSV que quieres leer
nombre_archivo_csv = 'C:\soporte\RRHH\employee_data.csv'

# Nombre del archivo .txt en el que quieres guardar la información
nombre_archivo_txt = 'datos_tabla.txt'

# Llamar a la función para guardar el contenido del CSV en un archivo .txt
guardar_csv_como_txt(nombre_archivo_csv, nombre_archivo_txt)
