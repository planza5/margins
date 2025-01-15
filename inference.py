import torch
import torchvision.transforms as transforms
from PIL import Image
from generator import UNetGenerator

######################################
# Configuración inicial
######################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Ruta del archivo de pesos
weights_path = "checkpoints/gen_epoch_100.pth"

# Crear instancia del generador
try:
    # Si guardaste solo los pesos:
    gen = UNetGenerator(in_channels=3, out_channels=3, features=64).to(device)
    gen.load_state_dict(torch.load(weights_path, map_location=device))
    print("Pesos cargados correctamente.")
except RuntimeError as e:
    # Si guardaste todo el modelo:
    print("Cargando el modelo completo...")
    gen = torch.load(weights_path, map_location=device)
    gen.to(device)
    print("Modelo cargado correctamente.")

# Poner el modelo en modo evaluación
gen.eval()

######################################
# Funciones de normalización
######################################
def normalize(image):
    return transforms.ToTensor()(image).unsqueeze(0) * 2 - 1

def denormalize(tensor):
    return transforms.ToPILImage()((tensor.squeeze(0) + 1) / 2)

######################################
# Cargar y procesar la imagen
######################################
image_path = "Images_512x1024_Margin/IMG_0001.jpg"
output_path = "result.jpg"

try:
    input_image = Image.open(image_path).convert("RGB")
    print("Imagen cargada correctamente.")
except Exception as e:
    print(f"Error al cargar la imagen: {e}")
    exit()

# Normalizar la imagen
input_tensor = normalize(input_image).to(device)

######################################
# Generar la imagen con márgenes
######################################
with torch.no_grad():
    output_tensor = gen(input_tensor)

# Desnormalizar la imagen de salida y guardarla
output_image = denormalize(output_tensor.cpu())
output_image.save(output_path)
print(f"Imagen generada guardada en: {output_path}")
