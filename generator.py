import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """
    Generador de tipo U-Net que recibe una imagen de entrada
    (por ejemplo, con los márgenes recortados) y produce la misma
    imagen pero con márgenes rellenos.

    - 'in_channels' es el número de canales de la imagen de entrada.
    - 'out_channels' es el número de canales de la imagen de salida.
      Normalmente coincide con in_channels (por ejemplo, 3 para RGB).
    - 'features' controla la cantidad de filtros en la primera capa;
      se suele doblar en cada bajada (downsampling).
    """

    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # 1. Definimos el "downsampling" (encoder).
        # Cada bloque reduce la resolución y aumenta el número de canales.
        # Usamos convoluciones con stride=2 para reducir el tamaño.

        self.down1 = self._block(in_channels, features, name="down1", batchnorm=False)
        self.down2 = self._block(features, features * 2, name="down2")
        self.down3 = self._block(features * 2, features * 4, name="down3")
        self.down4 = self._block(features * 4, features * 8, name="down4")
        self.down5 = self._block(features * 8, features * 8, name="down5")

        # 2. Definimos capas de "bottleneck" (la parte más profunda).
        # Es donde la resolución es mínima, pero tenemos más canales.
        self.bottleneck = self._block(features * 8, features * 8, name="bottleneck")

        # 3. Definimos el "upsampling" (decoder).
        # Usamos ConvTranspose2d para duplicar la resolución.
        # Y concatenamos las skip connections con la salida del upsampling.

        self.up1 = self._up_block(features * 8, features * 8, name="up1")
        self.up2 = self._up_block(features * 8 * 2, features * 8, name="up2")
        self.up3 = self._up_block(features * 8 * 2, features * 4, name="up3")
        self.up4 = self._up_block(features * 4 * 2, features * 2, name="up4")
        self.up5 = self._up_block(features * 2 * 2, features, name="up5")

        # 4. Capa final para mapear a out_channels, habitualmente con tanh ([-1,1]) o sigmoid ([0,1]).
        # Aquí pondremos tanh.
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, name, kernel_size=4, stride=2, padding=1, batchnorm=True):
        """
        Bloque de convolución (downsampling) con activación LeakyReLU y batchnorm opcional.
        """
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batchnorm)
        )
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def _up_block(self, in_channels, out_channels, name, kernel_size=4, stride=2, padding=1):
        """
        Bloque de deconvolución (upsampling) con activación ReLU y batchnorm.
        """
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder (Downsampling)
        d1 = self.down1(x)  # (B, features,   H/2,   W/2)
        d2 = self.down2(d1)  # (B, features*2, H/4,   W/4)
        d3 = self.down3(d2)  # (B, features*4, H/8,   W/8)
        d4 = self.down4(d3)  # (B, features*8, H/16,  W/16)
        d5 = self.down5(d4)  # (B, features*8, H/32,  W/32)

        # Bottleneck
        bottleneck = self.bottleneck(d5)  # (B, features*8, H/64, W/64)

        # Decoder (Upsampling) con skip connections
        up1 = self.up1(bottleneck)  # (B, features*8, H/32, W/32)
        up1 = torch.cat([up1, d5], dim=1)  # Concatenate skip connection

        up2 = self.up2(up1)  # (B, features*8, H/16, W/16)
        up2 = torch.cat([up2, d4], dim=1)

        up3 = self.up3(up2)  # (B, features*4, H/8, W/8)
        up3 = torch.cat([up3, d3], dim=1)

        up4 = self.up4(up3)  # (B, features*2, H/4, W/4)
        up4 = torch.cat([up4, d2], dim=1)

        up5 = self.up5(up4)  # (B, features, H/2, W/2)
        up5 = torch.cat([up5, d1], dim=1)

        # Capa final para llegar a la resolución original
        out = self.final_conv(up5)  # (B, out_channels, H, W)

        return out


def getGenerator(in_channels=3, out_channels=3, features=64):
    """
    Devuelve una instancia del generador (U-Net) con los parámetros indicados.

    :param in_channels: canales de entrada (3 para RGB).
    :param out_channels: canales de salida (3 para RGB).
    :param features: número base de filtros en la primera capa.
    :return: Instancia de UNetGenerator (nn.Module).
    """
    return UNetGenerator(in_channels, out_channels, features)
