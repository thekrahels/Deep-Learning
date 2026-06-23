import torch
from typing import Any

class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.self_att = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4 * embed_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(4 * embed_dim, embed_dim)
        )
        self.in_norm = torch.nn.LayerNorm(embed_dim)
        self.mlp_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x) -> Any:
        x_norm = self.in_norm(x)
        x = x + self.self_att(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Embedding(128, embed_dim),
            *[
                TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
            ],
            torch.nn.Linear(embed_dim, 128)
        )

    def forward(self, x) -> Any:
        return self.network(x)

def train() -> None:
    with open(__file__) as f:
        code = f.read()

    tokens = torch.as_tensor([ord(c) for c in code])
    
    net = Transformer(128, 8, 0)
    optim = torch.optim.Adam(net.parameters(), lr=.001)
    for it in range(100):
        pred = net(tokens[None, :-1])[0]
        loss = torch.nn.functional.cross_entropy(pred, tokens[1:])
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(float(loss))
        break

    #print(net(tokens[None, :10]))
    #print(code)

##net = Transformer(128, 8, 4)
##net(torch.rand(16, 10, 128)).shape

if __name__ == "__main__":
    train()