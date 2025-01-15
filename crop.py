import os
from PIL import Image

# Directorios de origen y destino
input_dir = "Images_512x1024"
output_dir = "Images_512x1024_Margin"

# Crear el directorio de destino si no existe
os.makedirs(output_dir, exist_ok=True)

# Eliminar todas las imágenes existentes en el directorio de salida
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    if os.path.isfile(file_path) and filename.endswith((".jpg", ".jpeg", ".png")):
        os.remove(file_path)
        print(f"Eliminada: {filename}")


# Función para recortar y redimensionar imágenes
def crop_and_resize(image, crop_fraction=0.85, target_size=(512, 1024)):
    """
    Recorta un porcentaje central de la imagen y redimensiona al tamaño objetivo.

    :param image: Imagen PIL
    :param crop_fraction: Fracción central a conservar (0.9 = conservar 90%)
    :param target_size: Tamaño objetivo (alto, ancho)
    :return: Imagen recortada y redimensionada
    """
    # Obtener dimensiones originales
    width, height = image.size

    # Calcular dimensiones del recorte
    crop_width = int(width * crop_fraction)
    crop_height = int(height * crop_fraction)

    # Calcular los márgenes para el recorte central
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Recortar la imagen
    cropped_image = image.crop((left, top, right, bottom))

    # Redimensionar manteniendo la proporción
    resized_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

    return resized_image


# Procesar todas las imágenes en el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Ruta de la imagen de entrada
        input_path = os.path.join(input_dir, filename)

        # Abrir la imagen
        with Image.open(input_path) as img:
            # Recortar y redimensionar
            processed_image = crop_and_resize(img)

            # Guardar en el directorio de salida
            output_path = os.path.join(output_dir, filename)
            processed_image.save(output_path)

            print(f"Procesada: {filename}")

print("Proceso completado. Todas las imágenes han sido procesadas y guardadas.")
