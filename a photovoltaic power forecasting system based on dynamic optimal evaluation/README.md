# Photovoltaic Power Forecasting System Based on Dynamic Optimal Evaluation

## Description  
This project implements a dynamic ensemble framework for photovoltaic power forecasting using multiple deep learning models. The system evaluates model performance through accuracy and stability metrics, then optimizes ensemble weights via convex programming.  

**Code Structure**:  
- `main.py`: Data preprocessing, model training, evaluation, and dynamic ensemble logic.  
- `model.py`: Definitions for all forecasting models (LSTM variants, Transformer, GRU, etc.).  
- `visualization.py`: Functions for plotting prediction comparisons.  

## Dataset Information  
The dataset used in this study contains six key features for analyzing and comparing photovoltaic forecast data from different sources and their relationship with actual observations. The dataset includes timestamps, centralized endpoint predictions of solar power generation, European Center for Medium-Range Weather Forecasts (ECMWF) predicted solar irradiance, self-developed large model (GDFS) predicted solar irradiance, actual measured solar irradiance, and actual measured power generation.

### Preprocessing Steps  
1. **Missing Values**: Filled with `0` using `pandas.DataFrame.fillna`.  
2. **Normalization**: Applied `MinMaxScaler` from scikit-learn (scale: 0–1).  
3. **Train/Test Split**: 90% training, 10% testing.  
4. **Sliding Window**: Sequence length = 8 timesteps.  

## Requirements  
```python
torch==2.5.1+cu121
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
cvxpy==1.6.5  
matplotlib==3.10.1
tqdm==4.67.1  
```

## Usage Instructions  
### 1. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 2. Run the Code  
```bash
python main.py
```

## Methodology  
### Model Training  
- **Quantile Loss**: Trained with τ=0.8 to emphasize over-prediction robustness.  
- **Hyperparameters**:  
  ```python
  seq_len = 8       
  hidden_size = 64  
  epochs = 100       
  ```

### Dynamic Ensemble  
1. **Model Selection**: Top 2 models by total score (accuracy + stability).  
2. **Weight Optimization**:  
   ```python
   minimize ‖y - (w₁p₁ + w₂p₂)‖²  
   subject to: w₁ + w₂ = 1, w ≥ 0
   ```  
   Solved via CVXPY with the ECOS solver.  



