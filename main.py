import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from margin_dataset import MarginsDataset
import generator
import discriminator


######################################
# Función auxiliar para center crop  #
######################################
def center_crop_tensor(tensor, target_size):
    """
    Recorta el tensor 'tensor' para que sus dimensiones espaciales (alto y ancho)
    sean iguales a target_size, mediante un recorte centrado.

    :param tensor: Tensor de forma (B, C, H, W)
    :param target_size: Tuple (target_h, target_w)
    :return: Tensor recortado
    """
    _, _, h, w = tensor.shape
    target_h, target_w = target_size
    crop_top = (h - target_h) // 2
    crop_left = (w - target_w) // 2
    return tensor[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]


######################################
# Configuración inicial              #
######################################
# Definir dispositivo (GPU o CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Crear directorio para checkpoints si no existe
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Definir funciones de pérdida
criterion = nn.BCEWithLogitsLoss()
l1_loss_fn = nn.L1Loss()

# Instanciar generador y discriminador.
# Nota: Para una cGAN tipo Pix2Pix se trabaja con pares concatenados (6 canales),
# así que se configura el discriminador para in_channels=6.
gen = generator.getGenerator(in_channels=3, out_channels=3, features=64).to(device)
dis = discriminator.getDiscriminator(in_channels=6, features=64).to(device)

# Crear dataset y DataLoader (usar el dataset de entrenamiento)
dataset = MarginsDataset(
    do_normalize=True,
    mode="train",
    split_ratio=0.8,
    seed=42,
    augment_rotation=True  # Habilitar imágenes rotadas
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Definir optimizadores para generador y discriminador
optim_gen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_dis = torch.optim.Adam(dis.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Número de épocas de entrenamiento
num_epochs = 1

######################################
# Ciclo de entrenamiento             #
######################################
for epoch in range(num_epochs):
    epoch_dis_loss = 0.0
    epoch_gen_loss = 0.0

    for i, (imgs_no_margin, imgs_real) in enumerate(dataloader):
        # Mover imágenes al dispositivo
        imgs_no_margin = imgs_no_margin.to(device)
        imgs_real = imgs_real.to(device)

        #########################################
        # ENTRENAR EL DISCRIMINADOR
        #########################################
        # Generar imagen fake a partir de imgs_no_margin
        gen_out = gen(imgs_no_margin)

        # --- Pareja real ---
        # Concatenar la imagen sin margen con la imagen real (dim=1 -> 3+3=6 canales)
        real_pair = torch.cat([imgs_no_margin, imgs_real], dim=1)
        out_real = dis(real_pair)
        labels_real = torch.ones_like(out_real, device=imgs_real.device)
        dis_loss_real = criterion(out_real, labels_real)

        # --- Pareja falsa ---
        # Antes de concatenar, verificar que imgs_no_margin tenga la misma resolución que gen_out
        target_size = gen_out.shape[2:]  # (target_height, target_width)
        if imgs_no_margin.shape[2:] != target_size:
            imgs_no_margin_cropped = center_crop_tensor(imgs_no_margin, target_size)
        else:
            imgs_no_margin_cropped = imgs_no_margin

        fake_pair = torch.cat([imgs_no_margin_cropped, gen_out], dim=1)
        out_fake = dis(fake_pair)
        labels_fake = torch.zeros_like(out_fake, device=gen_out.device)
        dis_loss_fake = criterion(out_fake, labels_fake)

        # Combinar pérdidas del discriminador (promedio)
        dis_loss_total = 0.5 * (dis_loss_real + dis_loss_fake)

        optim_dis.zero_grad()
        dis_loss_total.backward()
        optim_dis.step()

        #########################################
        # ENTRENAR EL GENERADOR
        #########################################
        # Regenerar el par falso para entrenamiento del generador
        if imgs_no_margin.shape[2:] != target_size:
            imgs_no_margin_cropped = center_crop_tensor(imgs_no_margin, target_size)
        else:
            imgs_no_margin_cropped = imgs_no_margin
        fake_pair = torch.cat([imgs_no_margin_cropped, gen_out], dim=1)
        out_fake_for_gen = dis(fake_pair)
        labels_real_for_gen = torch.ones_like(out_fake_for_gen,
                                              device=gen_out.device)  # Quiere que se clasifique como real.
        gen_loss_adv = criterion(out_fake_for_gen, labels_real_for_gen)

        # Pérdida de reconstrucción L1 para forzar similitud con imgs_real
        gen_loss_l1 = l1_loss_fn(gen_out, imgs_real)
        lambda_l1 = 100.0
        gen_loss_total = gen_loss_adv + lambda_l1 * gen_loss_l1

        optim_gen.zero_grad()
        gen_loss_total.backward()
        optim_gen.step()

        # Acumular pérdidas para el promedio de la época
        epoch_dis_loss += dis_loss_total.item()
        epoch_gen_loss += gen_loss_total.item()

        # Depuración: imprimir cada 50 batches
        if (i + 1) % 50 == 0:
            print(
                f"[Epoch {epoch + 1} - Batch {i + 1}/{len(dataloader)}] D Loss: {dis_loss_total.item():.4f} | G Loss: {gen_loss_total.item():.4f}")

    # Promedio de pérdidas por época
    avg_dis_loss = epoch_dis_loss / len(dataloader)
    avg_gen_loss = epoch_gen_loss / len(dataloader)
    print(
        f"***** Epoch [{epoch + 1}/{num_epochs}] - Avg D Loss: {avg_dis_loss:.4f} | Avg G Loss: {avg_gen_loss:.4f} *****")

    # Guardar checkpoints cada 10 épocas
    if (epoch + 1) % 10 == 0:
        torch.save(gen.state_dict(), f"checkpoints/gen_epoch_{epoch + 1}.pth")
        torch.save(dis.state_dict(), f"checkpoints/dis_epoch_{epoch + 1}.pth")

# Guardar los modelos finales
torch.save(gen.state_dict(), "checkpoints/gen_final.pth")
torch.save(dis.state_dict(), "checkpoints/dis_final.pth")

torch.save(gen, "checkpoints/gen_final_complete.pth")



print("Entrenamiento finalizado.")