from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints


        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        ##raise NotImplementedError
        x = torch.cat([track_left, track_right], dim=1)
        x = x.flatten(start_dim=1)
        x = self.net(x)
        return x.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        self.d_model = d_model

        self.input_proj = nn.Linear(2, d_model)

        self.side_embed = nn.Embedding(2, d_model)

        self.pos_embed = nn.Embedding(n_track * 2, d_model)

        self.query_embed = nn.Embedding(n_waypoints, d_model)



        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=3,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )


    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        ##raise NotImplementedError
        batch_size = track_left.shape[0]
        device = track_left.device

        tracks = torch.cat([track_left, track_right], dim=1)
        memory = self.input_proj(tracks)

        # side ids: first n_track tokens are left, next n_track are right
        side_ids = torch.cat(
            [
                torch.zeros(self.n_track, dtype=torch.long, device=device),
                torch.ones(self.n_track, dtype=torch.long, device=device),
            ],
            dim=0,
        )

        pos_ids = torch.arange(self.n_track * 2, device=device)

        memory = (
            memory
            + self.side_embed(side_ids)[None, :, :]
            + self.pos_embed(pos_ids)[None, :, :]
        )

        query_ids = torch.arange(self.n_waypoints, device=device)
        queries = self.query_embed(query_ids)[None, :, :].expand(batch_size, -1, -1)

        decoded = self.decoder(tgt=queries, memory=memory)
        waypoints = self.output_head(decoded)

        return waypoints



class CNNPlanner(torch.nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            in_channels: int, 
            out_channels: int, 
            stride: int
        ):
            super().__init__()

            self.n_waypoints = n_waypoints

            self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
            self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)



            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
                nn.GroupNorm(8, 32),
                nn.ReLU(),

                self.Block(32, 64, stride=2),
                self.Block(64, 128, stride=2),
                self.Block(128, 256, stride=2),
                self.Block(256, 256, stride=1),
            )

            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_waypoints * 2),
            )

            self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        ##raise NotImplementedError
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)

        return x.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
