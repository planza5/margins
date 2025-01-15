from PIL import Image, ImageDraw
import os

# Directorios de origen y destino
input_dir = "Images_512x1024"
output_dir = "Images_512x1024_Margin"

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)


# Función para pintar márgenes blancos directamente sobre la imagen
def paint_white_margins(image, margin_size=50):
    """
    Pinta márgenes blancos directamente sobre la imagen original.

    :param image: Imagen PIL
    :param margin_size: Tamaño del margen en píxeles.
    :return: Imagen con márgenes blancos.
    """
    # Crear una copia de la imagen para modificar
    image_with_margins = image.copy()
    draw = ImageDraw.Draw(image_with_margins)

    # Pintar los márgenes blancos (superior, inferior, izquierdo, derecho)
    width, height = image.size
    draw.rectangle([(0, 0), (width, margin_size)], fill="white")  # Margen superior
    draw.rectangle([(0, height - margin_size), (width, height)], fill="white")  # Margen inferior
    draw.rectangle([(0, 0), (margin_size, height)], fill="white")  # Margen izquierdo
    draw.rectangle([(width - margin_size, 0), (width, height)], fill="white")  # Margen derecho

    return image_with_margins


# Procesar todas las imágenes en el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Ruta de la imagen de entrada
        input_path = os.path.join(input_dir, filename)

        # Abrir la imagen
        with Image.open(input_path) as img:
            # Pinta los márgenes blancos directamente sobre la imagen
            img_with_margins = paint_white_margins(img)

            # Guardar la imagen en el directorio de salida
            output_path = os.path.join(output_dir, filename)
            img_with_margins.save(output_path)

            print(f"Procesada: {filename}")

print("Todas las imágenes han sido procesadas y guardadas en el directorio de salida.")
