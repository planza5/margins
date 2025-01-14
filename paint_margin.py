import os
from PIL import Image

def paint_white_margin_fixed_pixels(
    source_dir: str,
    dest_dir: str,
    margin_px: int = 50,
    margin_color=(255, 255, 255)
):
    """
    Recorre las imágenes .jpg de source_dir, quita 'margin_px' píxeles en cada lado
    y pinta esa zona de margen con margin_color (por defecto, blanco (255,255,255)),
    manteniendo la misma resolución original. Guarda las imágenes resultantes en dest_dir.

    :param source_dir:  Directorio con las imágenes originales.
    :param dest_dir:    Directorio donde se guardarán las imágenes con margen pintado.
    :param margin_px:   Número de píxeles a quitar/pintar en cada borde.
    :param margin_color: Color en RGB para el margen, por defecto blanco (255,255,255).
    """
    # Crea el directorio de destino si no existe
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.jpg'):
            source_path = os.path.join(source_dir, filename)

            try:
                with Image.open(source_path) as img:
                    width, height = img.size

                    # Calcular los bordes a recortar
                    left_cut   = margin_px
                    right_cut  = width - margin_px
                    top_cut    = margin_px
                    bottom_cut = height - margin_px

                    # Verificamos que no sea inválido el recorte
                    if right_cut > left_cut and bottom_cut > top_cut:
                        # 1) Recortamos la sección central
                        cropped_img = img.crop((left_cut, top_cut, right_cut, bottom_cut))

                        # 2) Creamos una nueva imagen del mismo tamaño, rellena de color blanco
                        new_img = Image.new("RGB", (width, height), margin_color)

                        # 3) Pegamos la parte recortada en su posición original
                        new_img.paste(cropped_img, (left_cut, top_cut))

                        # 4) Guardamos con el mismo nombre en el directorio de destino
                        dest_path = os.path.join(dest_dir, filename)
                        new_img.save(dest_path, format="JPEG")

                        print(f"Imagen procesada: {filename}")
                    else:
                        print(
                            f"Saltando {filename}: la imagen es demasiado pequeña "
                            f"para quitar {margin_px}px en cada borde."
                        )

            except Exception as e:
                print(f"Error al procesar {filename}: {e}")


if __name__ == "__main__":
    # Directorio de origen y de destino
    directorio_origen = "./Images_512x1024"
    directorio_destino = "./Images_512x1024_Margin"

    # Cantidad de píxeles a quitar/pintar en cada lado
    margen_pixeles = 50

    paint_white_margin_fixed_pixels(
        source_dir=directorio_origen,
        dest_dir=directorio_destino,
        margin_px=margen_pixeles,
        margin_color=(255, 255, 255)  # Blanco
    )
