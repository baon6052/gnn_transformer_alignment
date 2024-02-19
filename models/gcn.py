import lightning as L
from torch import nn, optim
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(L.LightningModule):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_hidden: int = 4,
        num_hidden_layers: int = 1,
        visualise: bool = False,
        learning_rate: float = 0.01,
    ):
        super().__init__()

        self.visualise = visualise
        self.learning_rate = learning_rate

        self.conv1 = GCNConv(num_features, num_hidden)
        self.hidden_layers = []

        for _ in range(num_hidden_layers):
            self.hidden_layers.append(GCNConv(num_hidden, num_hidden))

        self.hidden_layers = nn.ModuleList(self.hidden_layers)

        self.conv3 = GCNConv(num_hidden, 2)
        self.classifier = Linear(2, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()

        for hidden_layer in self.hidden_layers:
            h = hidden_layer(h, edge_index)
            h = h.tanh()

        h = self.conv3(h, edge_index)
        h = h.tanh()

        out = self.classifier(h)
        return out, h

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, "test")

    def _evaluate(self, batch, stage: str):
        out, h = self.forward(batch.x, batch.edge_index)
        out = out.cpu()
        batch = batch.cpu()

        node_mask = batch.val_mask
        label_mask = batch.val_mask

        if stage == "test":
            node_mask = batch.test_mask
            label_mask = batch.test_mask

        pred = out.argmax(1)

        # loss = self.criterion(out[node_mask], batch.y[label_mask])
        acc = (pred[node_mask] == batch.y[label_mask]).float().mean()
        self.log(f"{stage}_accuracy", acc, prog_bar=False)
