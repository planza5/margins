import torch
from PIL import Image
from torchvision import transforms

# Cargar el modelo entrenado
model_path = "checkpoints/gen_final_complete.pth"
gen = torch.load(model_path)
gen.eval()  # Establecer en modo evaluación


# Función para procesar una imagen
def add_borders(input_image_path, output_image_path):
    """
    Añade bordes a una imagen de entrada utilizando el modelo entrenado.

    :param input_image_path: Ruta de la imagen de entrada.
    :param output_image_path: Ruta para guardar la imagen de salida.
    """
    # Cargar la imagen
    image = Image.open(input_image_path).convert("RGB")
    original_size = image.size  # Guardar el tamaño original (ancho, alto)

    # Preprocesar la imagen
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizar a [-1, 1]
    ])
    image_tensor = preprocess(image).unsqueeze(0).cuda()  # Añadir batch dimension y mover a GPU si está disponible

    # Pasar la imagen por el modelo
    with torch.no_grad():
        output_tensor = gen(image_tensor)

    # Postprocesar la imagen generada
    postprocess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Desnormalizar de [-1, 1] a [0, 1]
        transforms.ToPILImage()
    ])
    output_image = postprocess(output_tensor.squeeze(0).cpu())

    # Redimensionar a las dimensiones originales, si es necesario
    output_image = output_image.resize(original_size)

    # Guardar la imagen de salida
    output_image.save(output_image_path)
    print(f"Imagen guardada en: {output_image_path}")


# Ejemplo de uso
add_borders("input_image.jpg", "output_image_with_borders.jpg")
