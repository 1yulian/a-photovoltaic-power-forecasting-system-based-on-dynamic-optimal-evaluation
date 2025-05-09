import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import sLSTM, mLSTM, xLSTM, TransformerModel, GRUModel, NBeats, BiLSTMAttention, EnhancedBiLSTMAttention
import pandas as pd
import cvxpy as cp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib
matplotlib.use('TkAgg')

from visualization import plot_comparison


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


input_size = 3
head_size = 32
num_heads = 2
hidden_size = 64
num_layers = 2
output_size = input_size
conv_channels = 16
kernel_size = 3
batch_size = 8000
seq_len = 8




class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.8):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.mean(torch.max((self.quantile - 1) * errors,
                                    self.quantile * errors))
        return loss


# Load data
data = pd.read_excel(r'D:\桌面\实验代码\peerj代码\xlstm\data\宇明10.10-11.15（九天预测）.xlsx', usecols=[0, 4, 5])
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
dataset = data.values.astype('float32')

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split dataset
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_dataset(dataset, seq_len=8):
    dataX, dataY = [], []
    for i in range(len(dataset) - seq_len):
        a = dataset[i:(i + seq_len - 1)]
        dataX.append(a)
        dataY.append(dataset[i + seq_len - 1])
    return torch.Tensor(dataX).to(device), torch.Tensor(dataY).to(device)


trainX, trainY = create_dataset(train, seq_len)
testX, testY = create_dataset(test, seq_len)

# DataLoader
train_dataset = TensorDataset(trainX, trainY)
test_dataset = TensorDataset(testX, testY)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


models = {
    "xLSTM": xLSTM(input_size, head_size, num_heads, batch_first=True, layers='msm').to(device),
    "LSTM": nn.LSTM(input_size, head_size, batch_first=True, proj_size=input_size).to(device),
    "sLSTM": sLSTM(input_size, head_size, num_heads, batch_first=True).to(device),
    "mLSTM": mLSTM(input_size, head_size, num_heads, batch_first=True).to(device),
    "Transformer": TransformerModel(input_size, num_heads, hidden_size, num_layers, output_size).to(device),
    "GRU": GRUModel(input_size, hidden_size, output_size, num_layers=num_layers).to(device),
    "NBeats": NBeats(input_size=input_size, output_size=output_size).to(device),
    "BiLSTMAttention": BiLSTMAttention(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                       output_size=output_size).to(device),
    "EnhancedBiLSTMAttention": EnhancedBiLSTMAttention(input_size=input_size, conv_channels=conv_channels,
                                                       kernel_size=kernel_size, hidden_size=hidden_size,
                                                       num_layers=num_layers, output_size=output_size).to(device)
}



def train_model(model, model_name, epochs=100, learning_rate=0.001):
    criterion = QuantileLoss(quantile=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    for epoch in tqdm(range(epochs), desc=f'Training {model_name}'):
        model.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            preds = outputs[:, -1, :] if model_name not in ["Transformer", "GRU", "BiLSTMAttention",
                                                            "EnhancedBiLSTMAttention"] else outputs
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
    return model, train_losses


# Train all models
trained_models = {}
all_train_losses = {}
for model_name, model in models.items():
    trained_models[model_name], all_train_losses[model_name] = train_model(model, model_name)


def evaluate_model(model, data_loader, model_name):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            preds = outputs[:, -1, :] if model_name not in ["Transformer", "GRU", "BiLSTMAttention",
                                                            "EnhancedBiLSTMAttention"] else outputs
            predictions.extend(preds.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    return np.array(predictions), np.array(actuals)


# Obtain prediction results for all models
test_predictions = {}
test_actuals = {}
for model_name, model in trained_models.items():
    test_predictions[model_name], test_actuals[model_name] = evaluate_model(model, test_loader, model_name)



def calculate_scores(test_actuals, test_predictions, model_names):
    metrics = {}

    for name in model_names:
        actuals_denorm = scaler.inverse_transform(test_actuals[name])
        preds_denorm = scaler.inverse_transform(test_predictions[name])
        actuals = actuals_denorm[:, -1]
        preds = preds_denorm[:, -1]

        metrics[name] = {
            'mae': mean_absolute_error(actuals, preds),
            'rmse': np.sqrt(mean_squared_error(actuals, preds)),
            'r2': r2_score(actuals, preds),
            'pred_var': np.var(preds),
            'max_dev': np.max(np.abs(actuals - preds) / (np.abs(actuals) + 1e-6))
        }

    # Calculate score
    scores = {}
    for name in model_names:
        mae_score = 20 * (1 - metrics[name]['mae'] / sum(m['mae'] for m in metrics.values()))
        rmse_score = 20 * (1 - metrics[name]['rmse'] / sum(m['rmse'] for m in metrics.values()))
        r2_score_val = 20 * (metrics[name]['r2'] / max(m['r2'] for m in metrics.values()))
        acc_score = mae_score + rmse_score + r2_score_val

        var_score = 20 * (1 - metrics[name]['pred_var'] / sum(m['pred_var'] for m in metrics.values()))
        max_dev_score = 20 * (1 - metrics[name]['max_dev'] / sum(m['max_dev'] for m in metrics.values()))
        stab_score = var_score + max_dev_score

        total_score = 0.6 * acc_score + 0.4 * stab_score

        scores[name] = {
            'total_score': total_score,
            'acc_score': acc_score,
            'stab_score': stab_score,
            'details': {
                'MAE': metrics[name]['mae'],
                'RMSE': metrics[name]['rmse'],
                'R2': metrics[name]['r2'],
                'Variance': metrics[name]['pred_var'],
                'MaxDev': metrics[name]['max_dev']
            }
        }
    return scores



# def dynamic_ensemble(test_predictions, scores, model_names, scaler, beta=2.0):
#     actuals_denorm = scaler.inverse_transform(test_actuals[model_names[0]])
#     target_actual = actuals_denorm[:, -1]
#
#     sorted_models = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:2]
#     top_models = [m[0] for m in sorted_models]
#     print(f"\nSelected Top2 Models: {top_models}")
#
#     weights = {}
#     total = sum(np.exp(beta * scores[m]['total_score'] / 100) for m in top_models)
#     for m in top_models:
#         weights[m] = np.exp(beta * scores[m]['total_score'] / 100) / total

#     ensemble_pred = np.zeros_like(test_predictions[top_models[0]][:, -1])
#     for m in top_models:
#         preds_denorm = scaler.inverse_transform(test_predictions[m])
#         ensemble_pred += weights[m] * preds_denorm[:, -1]
#
#     ensemble_rmse = np.sqrt(mean_squared_error(target_actual, ensemble_pred))
#
#     print(f"\nEvaluation Metrics:")
#     print(f"MAE: {mean_absolute_error(target_actual, ensemble_pred):.4f}")
#     print(f"RMSE: {ensemble_rmse:.4f
#     print(f"R²: {r2_score(target_actual, ensemble_pred):.4f}")
#
#     return ensemble_pred, weights, target_actual




def dynamic_ensemble(test_predictions, scores, model_names, scaler):
    actuals_denorm = scaler.inverse_transform(test_actuals[model_names[0]])
    y = actuals_denorm[:, -1]

    # Select the Top 2 models
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)[:2]
    top_models = [m[0] for m in sorted_models]
    print(f"\nSelected Top2 Models: {top_models}")

    # Obtain predicted values
    p1 = scaler.inverse_transform(test_predictions[top_models[0]])[:, -1]
    p2 = scaler.inverse_transform(test_predictions[top_models[1]])[:, -1]

    # Convex Optimization Modeling
    w = cp.Variable(2)

    # Objective function: minimize L2 norm
    objective = cp.Minimize(cp.sum_squares(y - (w[0] * p1 + w[1] * p2)))
    constraints = [w >= 0,  w <= 1,  cp.sum(w) == 1  ]


    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError("Optimization failed to converge")

        w_opt = w.value.round(4)
    except Exception as e:
        print(f"Optimization failed: {str(e)}, using equal weights")
        w_opt = np.array([0.5, 0.5])  #Use average weights when failing

    ensemble_pred = w_opt[0] * p1 + w_opt[1] * p2

    # Evaluation Metrics
    metrics = {
        'MAE': mean_absolute_error(y, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y, ensemble_pred)),
        'R2': r2_score(y, ensemble_pred),
        'Weights': {top_models[0]: w_opt[0], top_models[1]: w_opt[1]}
    }

    print(f"\nOptimization Results:")
    print(f"- Optimal Weights: {top_models[0]}: {w_opt[0]:.4f}, {top_models[1]}: {w_opt[1]:.4f}")
    print(f"- MAE: {metrics['MAE']:.4f}")
    print(f"- RMSE: {metrics['RMSE']:.4f}")
    print(f"- R²: {metrics['R2']:.4f}")

    return ensemble_pred, metrics['Weights'], y



model_names = list(models.keys())
scores = calculate_scores(test_actuals, test_predictions, model_names)
beta = 2.0
ensemble_pred, weights, target_actual = dynamic_ensemble(test_predictions, scores, model_names, scaler)


epsilon = 1e-6

ensemble_mae = mean_absolute_error(target_actual, ensemble_pred)
safe_actual = np.where(target_actual == 0, epsilon, target_actual)
ensemble_mape = np.mean(np.abs((target_actual - ensemble_pred) / safe_actual)) * 100
ensemble_r2 = r2_score(target_actual, ensemble_pred)



print("\n=== Each model’s inverse normalization metric ===")
for name in model_names:
    print(f"""
    {name}:
        MAE  : {scores[name]['details']['MAE']:.4f}
        R²   : {scores[name]['details']['R2']:.4f}
    """)

ensemble_rmse = np.sqrt(mean_squared_error(target_actual, ensemble_pred))


print("\n=== Detailed scores for each model (sorted by total score)===")
sorted_models = sorted(scores.items(), key=lambda x: x[1]['total_score'], reverse=True)


print(f"{'Model Name':<25} | {'Total Score':<7} | {'Accuracy Score':<7} | {'Stability Score':<7} | {'MAE':<8} | {'RMSE':<8} | {'R²':<6} ")
print("-" * 115)


for model_name, score_data in sorted_models:
    details = score_data['details']
    print(
        f"{model_name:<25} | {score_data['total_score']:>7.2f} | "
        f"{score_data['acc_score']:>9.2f} | {score_data['stab_score']:>9.2f} | "
        f"{details['MAE']:>7.4f} | {details['RMSE']:>7.4f} | "
        f"{details['R2']:>5.4f}"
    )


print(f"\nEvaluation Metrics:")
print(f"MAE: {mean_absolute_error(target_actual, ensemble_pred):.4f}")
print(f"RMSE: {ensemble_rmse:.4f}")
print(f"R²: {r2_score(target_actual, ensemble_pred):.4f}")


print("\n\n=== Final metrics of the integrated model ===")
print(f"MAE          : {ensemble_mae:.4f}")
print(f"R²           : {ensemble_r2:.4f}")
print(f"Models used and their weights: {', '.join([f'{k} ({v:.2f})' for k, v in weights.items()])}")


for name in model_names:
    print(f"""
    {name.ljust(20)}:
    MAE={scores[name]['details']['MAE']:.4f}
    R²={scores[name]['details']['R2']:.4f}
    """)


actuals_denorm = scaler.inverse_transform(test_actuals[model_names[0]])
target_actual = actuals_denorm[:, -1]


plot_data = {
    'actual': target_actual,
    'ensemble_pred': ensemble_pred,
    'model_predictions': {
        model: scaler.inverse_transform(test_predictions[model])[:, -1]
        for model in models.keys()
    },
    'top_models': list(weights.keys()),
    'weights': weights,
    'beta': beta
}


plot_comparison(plot_data)


plot_comparison(plot_data, xlim=(150, 250))