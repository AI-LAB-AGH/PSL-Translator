import wandb


sweep_configuration = {
    'method': 'bayes',
    'metric': {'name': 'acc_mean', 'goal': 'maximize'},
    'parameters': {
        'model': {'values': ['LSTM', 'GRU']},
        'hidden_dim': {'values': [64, 128, 192, 256, 320, 384, 448, 512]},
        'learning_rate': {'values': [0.0001, 0.001, 0.01, 0.1]}
    }
}
