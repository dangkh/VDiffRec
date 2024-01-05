from setTransformer_module import *

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_items = 1000):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dropout = nn.Dropout(0.1)
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden))
        self.dec = nn.Sequential(
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))
        
        self.predictItem = nn.Linear(dim_output, num_items)
        self.activateF = nn.Sigmoid()

    def forward(self, X, label):
        X = self.enc(X)
        elabel = label.unsqueeze(-1)
        X = X * elabel
        X = X.sum(-2)
        numI = label.sum(1).reshape(-1, 1)
        X = X / numI
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        X = self.dropout(X)
        return X

    def predict(self, X):
        return self.activateF(self.predictItem(X))

class AttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionPooling, self).__init__()

        # Linear layers for attention scoring
        self.size = input_size
        self.V = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        # Calculate attention scores
        scores = self.tanh(self.V(input_features)) 
        scores = self.w(scores)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)

        # Apply attention weights to input features
        pooled_features = torch.sum(weights * input_features, dim=1)

        return pooled_features, weights


class GateAttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GateAttentionPooling, self).__init__()

        # Linear layers for attention scoring
        self.size = input_size
        self.V = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        self.sigd = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        # Calculate attention scores
        v = self.tanh(self.V(input_features)) 
        u = self.sigd(self.U(input_features)) 
        scores = self.w(v*u)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)

        # Apply attention weights to input features
        pooled_features = torch.sum(weights * input_features, dim=1)

        return pooled_features, weights

class selfAttentionSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128, num_items = 1000):
        super(selfAttentionSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.dropout = nn.Dropout(0.1)
        self.enc = GateAttentionPooling(dim_input, dim_input)
        self.dec = nn.Sequential(
                nn.ReLU(),
                nn.Linear(dim_input, num_outputs*dim_output))
        
        self.predictItem = nn.Linear(dim_output, num_items)
        self.activateF = nn.Sigmoid()

    def forward(self, X, label):
        X,_ = self.enc(X)
        X = self.dec(X)
        X = self.dropout(X)
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
        enc = self.enc(X)
        dec = self.dec(enc)
        return dec

    def predict(self, X):
        # print(X[0], X[0].shape)
        return self.activateF(self.predictItem(X))