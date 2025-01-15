import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from margin_dataset import MarginsDataset
import generator
import discriminator


######################################
# Función auxiliar para center crop  #
######################################
def center_crop_tensor(tensor, target_size):
    _, _, h, w = tensor.shape
    target_h, target_w = target_size
    crop_top = (h - target_h) // 2
    crop_left = (w - target_w) // 2
    return tensor[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]


######################################
# Función auxiliar para graficar pérdidas #
######################################
def plot_losses(dis_losses, gen_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(dis_losses, label='Discriminator Loss')
    plt.plot(gen_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('losses.png')  # Guardar como imagen
    plt.close()


######################################
# Configuración inicial              #
######################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
if not os.path.exists("images"):
    os.makedirs("images")

criterion = nn.BCEWithLogitsLoss()
l1_loss_fn = nn.L1Loss()

gen = generator.getGenerator(in_channels=3, out_channels=3, features=64).to(device)
dis = discriminator.getDiscriminator(in_channels=6, features=64).to(device)

dataset = MarginsDataset(
    do_normalize=True,
    mode="train",
    split_ratio=0.8,
    seed=42
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optim_gen = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_dis = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))  # Ajuste la tasa de aprendizaje aquí

num_epochs = 100

# Listas para almacenar las pérdidas
dis_losses = []
gen_losses = []

######################################
# Ciclo de entrenamiento             #
######################################
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"Comenzando la época {epoch + 1}/{num_epochs}")

    epoch_dis_loss = 0.0
    epoch_gen_loss = 0.0

    for i, (imgs_no_margin, imgs_real) in enumerate(dataloader):
        imgs_no_margin = imgs_no_margin.to(device)
        imgs_real = imgs_real.to(device)

        #########################################
        # ENTRENAR EL DISCRIMINADOR
        #########################################
        gen_out = gen(imgs_no_margin)

        real_pair = torch.cat([imgs_no_margin, imgs_real], dim=1)
        out_real = dis(real_pair)
        labels_real = torch.ones_like(out_real, device=imgs_real.device)
        dis_loss_real = criterion(out_real, labels_real)

        target_size = gen_out.shape[2:]
        if imgs_no_margin.shape[2:] != target_size:
            imgs_no_margin_cropped = center_crop_tensor(imgs_no_margin, target_size)
        else:
            imgs_no_margin_cropped = imgs_no_margin

        fake_pair = torch.cat([imgs_no_margin_cropped, gen_out], dim=1)
        out_fake = dis(fake_pair)
        labels_fake = torch.zeros_like(out_fake, device=gen_out.device)
        dis_loss_fake = criterion(out_fake, labels_fake)

        dis_loss_total = 0.5 * (dis_loss_real + dis_loss_fake)

        optim_dis.zero_grad()
        dis_loss_total.backward()
        optim_dis.step()

        #########################################
        # ENTRENAR EL GENERADOR
        #########################################
        for _ in range(3):  # Actualizar generador varias veces
            gen_out = gen(imgs_no_margin)
            if imgs_no_margin.shape[2:] != target_size:
                imgs_no_margin_cropped = center_crop_tensor(imgs_no_margin, target_size)

            fake_pair = torch.cat([imgs_no_margin_cropped, gen_out], dim=1)
            out_fake_for_gen = dis(fake_pair)
            labels_real_for_gen = torch.ones_like(out_fake_for_gen, device=gen_out.device)
            gen_loss_adv = criterion(out_fake_for_gen, labels_real_for_gen)

            gen_loss_l1 = l1_loss_fn(gen_out, imgs_real)
            lambda_l1 = 100.0
            gen_loss_total = gen_loss_adv + lambda_l1 * gen_loss_l1

            optim_gen.zero_grad()
            gen_loss_total.backward(retain_graph=True)
            optim_gen.step()

        epoch_dis_loss += dis_loss_total.item()
        epoch_gen_loss += gen_loss_total.item()

    avg_dis_loss = epoch_dis_loss / len(dataloader)
    avg_gen_loss = epoch_gen_loss / len(dataloader)
    dis_losses.append(avg_dis_loss)
    gen_losses.append(avg_gen_loss)

    epoch_time = time.time() - epoch_start_time
    total_time = time.time() - start_time
    remaining_time = epoch_time * (num_epochs - (epoch + 1))

    print(f"***** Epoch [{epoch + 1}/{num_epochs}] - Avg D Loss: {avg_dis_loss:.4f} | Avg G Loss: {avg_gen_loss:.4f} *****")
    print(f"Tiempo de la época: {epoch_time:.2f}s | Tiempo total transcurrido: {total_time:.2f}s | Tiempo restante estimado: {remaining_time:.2f}s")

    if (epoch + 1) % 10 == 0:
        torch.save(gen.state_dict(), f"checkpoints/gen_epoch_{epoch + 1}.pth")
        torch.save(dis.state_dict(), f"checkpoints/dis_epoch_{epoch + 1}.pth")

        # Guardar imágenes
        gen_out = gen(imgs_no_margin)
        comparison = torch.cat([imgs_no_margin, gen_out], 0)
        save_image(comparison, f"images/comparison_epoch_{epoch + 1}.png", nrow=imgs_no_margin.size(0), normalize=True)
        print(f"Imagen guardada para la época {epoch + 1}")

# Graficar pérdidas
plot_losses(dis_losses, gen_losses)

torch.save(gen.state_dict(), "checkpoints/gen_final.pth")
torch.save(dis.state_dict(), "checkpoints/dis_final.pth")
torch.save(gen, "checkpoints/gen_final_complete.pth")

print("Entrenamiento finalizado.")
