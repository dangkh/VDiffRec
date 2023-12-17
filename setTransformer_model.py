from setTransformer_module import *

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_items = 1000):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, num_outputs*dim_output))
        
        self.predictItem = nn.Linear(dim_output, num_items)
        self.activateF = nn.Tanh()

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

    def predict(self, X):
        return self.activateF(self.predictItem(X))

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, num_items = 1000, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))
        self.predictItem = nn.Linear(dim_output, num_items)
        self.activateF = nn.Sigmoid()

    def forward(self, X):
        # print(X[0], X[0].shape)
        enc = self.enc(X)
        # print(enc[0], enc[0].shape)
        dec = self.dec(enc)
        # print(dec[0], dec[0].shape)
        return dec

    def predict(self, X):
        # print(X[0], X[0].shape)
        return self.activateF(self.predictItem(X))