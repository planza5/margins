import os
from PIL import Image


def clear_output_directory(output_dir):
    """
    Borra todas las imágenes del directorio de salida antes de comenzar el procesamiento.

    :param output_dir: Directorio de salida.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return

    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        if file_name.lower().endswith('.jpg'):
            os.remove(file_path)
            print(f"Borrado: {file_name}")


def resize_and_crop_multiple(input_dirs, output_dir, resize_size, crop_size):
    """
    Procesa imágenes de varios directorios de entrada, redimensiona y recorta,
    y guarda todas las imágenes procesadas en un único directorio de salida.

    :param input_dirs: Lista de directorios de entrada con imágenes originales.
    :param output_dir: Directorio donde se guardarán las imágenes procesadas.
    :param resize_size: Tamaño para redimensionar (ancho, alto) antes del recorte.
    :param crop_size: Tamaño final después del recorte central (ancho, alto).
    """
    # Limpiar el directorio de salida
    clear_output_directory(output_dir)

    counter = 1  # Contador para nombres de archivo

    for input_dir in input_dirs:
        print(f"Procesando directorio: {input_dir}")
        for file_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, file_name)

            # Procesar solo archivos .jpg
            if not file_name.lower().endswith('.jpg'):
                print(f"Saltando archivo no soportado: {file_name}")
                continue

            try:
                with Image.open(input_path) as img:
                    # Rotar si el ancho es mayor que la altura
                    if img.width > img.height:
                        img = img.rotate(90, expand=True)

                    # Redimensionar a 512x1138
                    img_resized = img.resize(resize_size, Image.LANCZOS)

                    # Recortar a 512x1024
                    width, height = resize_size
                    target_width, target_height = crop_size
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height
                    img_cropped = img_resized.crop((left, top, right, bottom))

                    # Crear el nuevo nombre de archivo
                    output_file_name = f"IMG_{counter:04d}.jpg"
                    output_path = os.path.join(output_dir, output_file_name)

                    # Guardar la imagen procesada
                    img_cropped.save(output_path, format="JPEG")
                    print(f"Procesada: {output_file_name}")

                    # Incrementar el contador
                    counter += 1

            except Exception as e:
                print(f"Error procesando {file_name}: {e}")


# Parámetros
input_directories = [
    r"C:\Users\planz\Pictures\BackupMovil\Camera 2023-04-12",
    r"C:\Users\planz\Pictures\BackupMovil\Camera 2025-12-12"
]
output_directory = "./Images_512x1024"  # Directorio de salida único
resize_dimensions = (512, 1138)  # Ancho x Alto para el redimensionado
crop_dimensions = (512, 1024)  # Ancho x Alto para el recorte central

# Ejecutar
resize_and_crop_multiple(input_directories, output_directory, resize_dimensions, crop_dimensions)
