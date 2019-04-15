from .autoencoder import Autoencoder


class LinAutoencoder(Autoencoder):
    def build_module(self):
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=out.shape[1], out_features = self.feature_layer_size, bias=self.use_bias),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.feature_layer_size , out_features = out.shape[1],bias=self.use_bias),
            nn.Tanh())
