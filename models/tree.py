import torch


def create_lag_features_tensor(data, column_index, lag=10):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    column_data = data[:, column_index]

    lagged_features = [column_data]
    for i in range(1, lag + 1):
        lagged_feature = torch.cat((torch.full((i,), float('nan')), column_data[:-i]))
        lagged_features.append(lagged_feature.view(-1, 1))

    return torch.cat([data] + lagged_features[1:], dim=1)

def create_tabular_dataset_tensor(data, lag=10):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    lagged_data = data
    for col in range(data.shape[1]):
        lagged_data = create_lag_features_tensor(lagged_data, column_index=col, lag=lag)

    lagged_data = lagged_data[~torch.isnan(lagged_data).any(dim=1)]
    return lagged_data
