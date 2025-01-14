import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    Discriminador PatchGAN típico de Pix2Pix.
    Se encarga de distinguir si una imagen es real o falsa (generated)
    evaluándola en parches.

    - 'in_channels' es el número de canales de la imagen de entrada.
      (Para RGB: 3; si concatenas imagen original + imagen generada, puede ser 6, etc.).
    - 'features' controla la cantidad base de filtros;
      cada capa la multiplica en potencia de 2 (p. ej., 64, 128, 256...).
    """

    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()

        # 1. Primera capa sin BatchNorm, según la convención Pix2Pix
        #   (conv + LeakyReLU)
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 2. A partir de la segunda capa, añadimos BatchNorm
        self.down2 = self._block(features, features * 2, stride=2)
        self.down3 = self._block(features * 2, features * 4, stride=2)

        # 3. Cuarta capa con stride=1 para reducir menos la resolución
        #    y aumentar la capacidad discriminativa
        self.down4 = self._block(features * 4, features * 8, stride=1)

        # 4. Capa final: conv con stride=1 que mapea a un solo canal
        #    (representación de real/fake). No le ponemos activación
        #    porque usaremos una pérdida (ej. BCEWithLogitsLoss)
        #    que aplica sigmoide internamente.
        self.final = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

    def _block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """
        Bloque de (Conv2d + BatchNorm2d + LeakyReLU).
        Se usa para la 'bajada' típica en el PatchGAN.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """
        Forward pass:
        x (B, in_channels, H, W) -> salida (B, 1, H/2^n, W/2^n)
        donde H/2^n, W/2^n dependen de las strides.
        """
        x = self.down1(x)  # Conv(stride=2) -> (B, features, H/2, W/2)
        x = self.down2(x)  # Conv(stride=2) -> (B, features*2, H/4, W/4)
        x = self.down3(x)  # Conv(stride=2) -> (B, features*4, H/8, W/8)
        x = self.down4(x)  # Conv(stride=1) -> (B, features*8, H/8, W/8)
        x = self.final(x)  # Conv(stride=1) -> (B, 1, H/8, W/8)

        return x


def getDiscriminator(in_channels=3, features=64):
    """
    Devuelve una instancia del discriminador PatchGAN.

    :param in_channels: canales de entrada (3 para imagen RGB simple).
                        Si en Pix2Pix concatenas la imagen original con la generada,
                        suele ser 3+3=6 canales.
    :param features: número base de filtros en la primera capa.
    :return: Instancia de PatchGANDiscriminator (nn.Module).
    """
    return PatchGANDiscriminator(in_channels, features)
