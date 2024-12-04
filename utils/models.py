import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv2(self.sigmoid(self.conv1(x))))
        return x * attention


class ResNetUNetWithAttention(nn.Module):
    def __init__(
        self,
        output_channels=1,
        freeze_entire_backbone=False,
        freeze_initial_layers=False,
    ):
        super(ResNetUNetWithAttention, self).__init__()

        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())

        if freeze_entire_backbone:
            # Freeze the entire backbone
            for param in self.base_model.parameters():
                param.requires_grad = False

        if freeze_initial_layers:
            # Freeze only the initial layers
            for param in self.layer0.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # Initial Conv layer
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # MaxPool and Layer1
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn4 = AttentionBlock(512)
        self.decoder4 = self.conv_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn3 = AttentionBlock(256)
        self.decoder3 = self.conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn2 = AttentionBlock(128)
        self.decoder2 = self.conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionBlock(128)
        self.decoder1 = self.conv_block(128, 64)

        # New layers for final upsampling
        self.upconv0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_decoder = self.conv_block(64, 64)
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        # Decoder with skip connections and attention
        d4 = torch.cat((self.upconv4(e4), e3), dim=1)
        d4 = self.attn4(d4)
        d4 = self.decoder4(d4)

        d3 = torch.cat((self.upconv3(d4), e2), dim=1)
        d3 = self.attn3(d3)
        d3 = self.decoder3(d3)

        d2 = torch.cat((self.upconv2(d3), e1), dim=1)
        d2 = self.attn2(d2)
        d2 = self.decoder2(d2)

        d1 = torch.cat((self.upconv1(d2), e0), dim=1)
        d1 = self.attn1(d1)
        d1 = self.decoder1(d1)

        d0 = self.upconv0(d1)
        d0 = self.final_decoder(d0)
        out = self.final_conv(d0)

        return out


class UnetNoPretraining(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, initial_filters=64):
        super(UnetNoPretraining, self).__init__()

        self.encoder1 = self.conv_block(input_channels, initial_filters)
        self.encoder2 = self.conv_block(initial_filters, initial_filters * 2)
        self.encoder3 = self.conv_block(initial_filters * 2, initial_filters * 4)
        self.encoder4 = self.conv_block(initial_filters * 4, initial_filters * 8)
        self.encoder5 = self.conv_block(initial_filters * 8, initial_filters * 16)

        self.pool = nn.MaxPool2d(2)

        # Adjusting channels to match attention input
        self.upconv5 = nn.ConvTranspose2d(
            initial_filters * 16, initial_filters * 8, kernel_size=2, stride=2
        )
        self.attn5 = AttentionBlock(initial_filters * 16)  # Updated input channels
        self.decoder5 = self.conv_block(initial_filters * 16, initial_filters * 8)

        self.upconv4 = nn.ConvTranspose2d(
            initial_filters * 8, initial_filters * 4, kernel_size=2, stride=2
        )
        self.attn4 = AttentionBlock(initial_filters * 8)  # Updated input channels
        self.decoder4 = self.conv_block(initial_filters * 8, initial_filters * 4)

        self.upconv3 = nn.ConvTranspose2d(
            initial_filters * 4, initial_filters * 2, kernel_size=2, stride=2
        )
        self.attn3 = AttentionBlock(initial_filters * 4)  # Updated input channels
        self.decoder3 = self.conv_block(initial_filters * 4, initial_filters * 2)

        self.upconv2 = nn.ConvTranspose2d(
            initial_filters * 2, initial_filters, kernel_size=2, stride=2
        )
        self.attn2 = AttentionBlock(initial_filters * 2)  # Updated input channels
        self.decoder2 = self.conv_block(initial_filters * 2, initial_filters)

        self.final_conv = nn.Conv2d(initial_filters, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))

        # Decoding path with skip connections and attention
        d5 = torch.cat((self.upconv5(e5), e4), dim=1)
        d5 = self.attn5(d5)  # Apply attention
        d5 = self.decoder5(d5)

        d4 = torch.cat((self.upconv4(d5), e3), dim=1)
        d4 = self.attn4(d4)  # Apply attention
        d4 = self.decoder4(d4)

        d3 = torch.cat((self.upconv3(d4), e2), dim=1)
        d3 = self.attn3(d3)  # Apply attention
        d3 = self.decoder3(d3)

        d2 = torch.cat((self.upconv2(d3), e1), dim=1)
        d2 = self.attn2(d2)  # Apply attention
        d2 = self.decoder2(d2)

        # Final output layer
        out = self.final_conv(d2)

        return out  # Assuming binary segmentation


class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
