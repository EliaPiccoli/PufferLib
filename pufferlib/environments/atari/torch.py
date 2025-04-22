import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(pufferlib.models.Convolutional):
    def __init__(self, env, input_size=512, hidden_size=512, output_size=512,
            framestack=1, flat_size=64*6*9):
        super().__init__(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            framestack=framestack,
            flat_size=flat_size,
        )

class WSAPolicy(pufferlib.models.WSA):
    def __init__(self, env, emb_size=256, input_size=256, hidden_size=256,
                 output_size=512, device='cuda:0'):
        super().__init__(
            env=env,
            emb_size=emb_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device
        )

class EnsemblePolicy(pufferlib.models.Ensemble):
    def __init__(self, env, emb_size=256, input_size=256, hidden_size=256,
                 output_size=512, device='cuda:0'):
        super().__init__(
            env=env,
            emb_size=emb_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device
        )

class WSAFTPolicy(pufferlib.models.WSAFT):
    def __init__(self, env, emb_size=256, input_size=256, hidden_size=256,
                 output_size=512, device='cuda:0'):
        super().__init__(
            env=env,
            emb_size=emb_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device
        )

class WSASinglePolicy(pufferlib.models.WSASingle):
    def __init__(self, env, emb_size=256, input_size=256, hidden_size=256,
                 output_size=512, device='cuda:0'):
        super().__init__(
            env=env,
            emb_size=emb_size,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            device=device
        )