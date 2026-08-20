# shapeft/encoders/clay_encoder.py
# Wraps the Clay Foundation Model (https://github.com/Clay-foundation/model)
# to match the ShapingFT / PANGAEA-style Encoder interface used by
# CROMA_OPTICAL_Encoder and SSL4EO_DINO_Encoder.
#
# Requires the `claymodel` package importable, e.g.:
#   pip install git+https://github.com/Clay-foundation/model.git
# and the v1.5 checkpoint:
#   wget https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt

import math
from logging import Logger
from pathlib import Path

import torch
from einops import rearrange, repeat

from shapeft.encoders.base import Encoder

# Wavelengths (nanometers) for the Sentinel-2 L2A bands in the exact order
# used by PASTIS-HD's `bands.optical` list: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
S2_WAVES_NM = [493.0, 560.0, 665.0, 704.0, 740.0, 783.0, 842.0, 865.0, 1610.0, 2190.0]
S2_GSD = 10.0


class Clay_Optical_Encoder(Encoder):
    """
    Clay v1.5 encoder adapted to ShapingFT's single-timestep, dict-based
    encoder interface. Clay's own spatiotemporal position encoding needs
    `time` and `latlon`, which ShapingFT's decoder does not forward to the
    encoder per call. We supply fixed placeholder time/latlon (see NOTE
    below) — wire real PASTIS acquisition dates / patch centroids through
    if you want Clay's temporal/geo encoding to actually be informative.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        size: str = "large",
    ):
        dims = {
            "tiny": (192, 12, 3, 64),
            "small": (384, 12, 6, 64),
            "base": (768, 12, 12, 64),
            "large": (1024, 24, 16, 64),
        }
        dim, depth, heads, dim_head = dims[size]

        super().__init__(
            model_name="clay_optical",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=dim,
            output_dim=output_dim,
            output_layers=output_layers,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers
        self.output_layers_set = set(output_layers)
        self.img_size = input_size
        self.patch_size = 8
        self.dim = dim

        try:
            from claymodel.model import Encoder as ClayEncoderBase
        except ImportError as exc:
            raise ImportError(
                "Clay encoder requires the claymodel package. Install it with "
                "`pip install git+https://github.com/Clay-foundation/model.git`."
            ) from exc

        self.clay_encoder = ClayEncoderBase(
            mask_ratio=0.0,
            patch_size=self.patch_size,
            shuffle=False,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=4.0,
        )

        self.register_buffer(
            "waves", torch.tensor(S2_WAVES_NM, dtype=torch.float32), persistent=False
        )

    def forward(self, image):
        cube = image["optical"].squeeze(2)  # (B, C, H, W)
        B = cube.shape[0]
        device = cube.device

        # NOTE: placeholder metadata — see class docstring.
        time = torch.zeros(B, 4, device=device)
        latlon = torch.zeros(B, 4, device=device)
        gsd = torch.tensor(S2_GSD, device=device)

        patches, _ = self.clay_encoder.to_patch_embed(cube, self.waves.to(device))
        patches = self.clay_encoder.add_encodings(patches, time, latlon, gsd)

        cls_tokens = repeat(self.clay_encoder.cls_token, "1 1 D -> B 1 D", B=B)
        patches = torch.cat((cls_tokens, patches), dim=1)

        hidden_states = []
        x = patches
        for i, (attn, ff) in enumerate(self.clay_encoder.transformer.layers):
            x = attn(x) + x
            x = ff(x) + x
            if i in self.output_layers:
                hidden_states.append(x)

        grid = self.img_size // self.patch_size
        output = [
            h[:, 1:, :]  # drop cls token
            .permute(0, 2, 1)
            .reshape(B, -1, grid, grid)
            .contiguous()
            for h in hidden_states
        ]
        return output

    def load_encoder_weights(self, logger: Logger, from_scratch: bool = False) -> None:
        if from_scratch:
            return
        ckpt = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        pretrained = {
            k.replace("model.encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.encoder.")
        }
        missing, incompatible_shape, matched = {}, {}, {}
        for name, param in self.clay_encoder.named_parameters():
            if name not in pretrained:
                missing[name] = param.shape
            elif pretrained[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained[name].shape)
            else:
                matched[name] = pretrained[name]
        self.clay_encoder.load_state_dict(matched, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
