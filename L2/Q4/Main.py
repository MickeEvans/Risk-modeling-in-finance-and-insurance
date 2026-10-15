import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# ---------------------------------------------------------
# 1. DATA PREPARATION & PARAMETERS
# ---------------------------------------------------------

# Automatically resolves the path to market_data.csv in the parent L2 folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'market_data.csv')

df = pd.read_csv(data_path, sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

stocks = ['BA', 'CAT', 'KO', 'DIS', 'JPM', 'MMM', 'MSFT', 'PFE', 'WMT', 'XOM']
rates = ['Y05', 'Y10', 'Y15', 'Y20', 'Y30']
maturities = np.array([5.0, 10.0, 15.0, 20.0, 30.0])

n_scenarios = 5000
n_steps = 480

log_returns = np.log(df[stocks] / df[stocks].shift(1)).dropna()
rate_diffs = (df[rates] - df[rates].shift(1)).dropna() # Rates are often modeled in differences

# ---------------------------------------------------------
# 2. DYNAMIC NELSON-SIEGEL (DNS) CALIBRATION
# ---------------------------------------------------------
# Lambda determines where the curvature is maximized (typically around 2-3 years for weekly data)
lam = 0.0609 

# Nelson-Siegel factor loadings matrix for our 5 maturities
# [1, (1-exp(-lam*t))/(lam*t), (1-exp(-lam*t))/(lam*t) - exp(-lam*t)]
H = np.zeros((5, 3))
H[:, 0] = 1.0
H[:, 1] = (1 - np.exp(-lam * maturities)) / (lam * maturities)
H[:, 2] = H[:, 1] - np.exp(-lam * maturities)

# Extract Level, Slope, Curvature (L, S, C) for every historical week via pseudo-inverse
rate_levels = df[rates].iloc[1:].values
dns_factors = rate_levels @ np.linalg.pinv(H).T

# Fit VAR(1) to DNS factors
X_dns = dns_factors[:-1]
Y_dns = dns_factors[1:]
dns_var_coef = np.linalg.inv(X_dns.T @ X_dns) @ X_dns.T @ Y_dns
dns_residuals = Y_dns - X_dns @ dns_var_coef

# ---------------------------------------------------------
# 3. EQUITY MARKET FACTOR CALIBRATION
# ---------------------------------------------------------
# Create equally weighted market factor
market_return = log_returns.mean(axis=1).values
# Fit AR(1) to market factor
X_mkt = np.vstack([np.ones(len(market_return)-1), market_return[:-1]]).T
Y_mkt = market_return[1:]
mkt_coef = np.linalg.inv(X_mkt.T @ X_mkt) @ X_mkt.T @ Y_mkt
mkt_residuals = Y_mkt - X_mkt @ mkt_coef

# Stock Betas to Market Factor
stock_betas = np.zeros(10)
stock_alphas = np.zeros(10)
stock_residuals = np.zeros((len(Y_mkt), 10))

for i, stock in enumerate(stocks):
    y_stock = log_returns[stock].values[1:]
    coef = np.linalg.inv(X_mkt.T @ X_mkt) @ X_mkt.T @ y_stock
    stock_alphas[i], stock_betas[i] = coef
    stock_residuals[:, i] = y_stock - X_mkt @ coef

# Combine all residuals into one system (3 rates, 1 market, 10 stocks = 14 dimensions)
all_residuals = np.hstack([dns_residuals, mkt_residuals.reshape(-1,1), stock_residuals])

# ---------------------------------------------------------
# 4. GARCH(1,1) CALIBRATION (AI Optimization)
# ---------------------------------------------------------
def garch_likelihood(params, returns):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return 1e10 # Penalize unstable parameters
    
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        
    log_lik = -0.5 * np.sum(np.log(sigma2) + (returns**2) / sigma2)
    return -log_lik

garch_params = np.zeros((14, 3))
std_residuals = np.zeros_like(all_residuals)
historical_vars = np.var(all_residuals, axis=0)

print("Calibrating GARCH models (this takes a few seconds)...")
for i in range(14):
    res = all_residuals[:, i]
    # Initial guesses: omega, alpha, beta
    init_guess = [historical_vars[i]*0.05, 0.1, 0.85] 
    opt = minimize(garch_likelihood, init_guess, args=(res,), method='Nelder-Mead')
    garch_params[i] = opt.x
    
    # Calculate standardized residuals for DCC
    s2 = np.zeros(len(res))
    s2[0] = historical_vars[i]
    for t in range(1, len(res)):
        s2[t] = opt.x[0] + opt.x[1]*res[t-1]**2 + opt.x[2]*s2[t-1]
    std_residuals[:, i] = res / np.sqrt(s2)

# DCC baseline correlation matrix (Unconditional)
Q_bar = np.corrcoef(std_residuals.T)

# ---------------------------------------------------------
# 5. MONTE CARLO SIMULATION WITH DCC-GARCH
# ---------------------------------------------------------
print(f"Simulating {n_scenarios} paths over {n_steps} weeks...")

# We will simulate the assets iteratively
sim_rates = np.zeros((n_scenarios, n_steps, 5))
sim_stocks = np.zeros((n_scenarios, n_steps, 10))

# DCC Parameters (typically highly persistent)
dcc_a, dcc_b = 0.05, 0.90 

# Last known values
last_rates = df[rates].iloc[-1].values
last_stocks = df[stocks].iloc[-1].values
last_dns = dns_factors[-1]
last_mkt = market_return[-1]
current_var = historical_vars.copy()
Q_t = Q_bar.copy()

for s in range(n_scenarios):
    dns_t = last_dns.copy()
    mkt_t = last_mkt
    var_t = current_var.copy()
    q_t = Q_t.copy()
    
    r_path = np.zeros((n_steps, 5))
    s_path = np.zeros((n_steps, 10))
    current_prices = last_stocks.copy()
    current_yields = last_rates.copy()
    
    # Generate standard normal shocks for the entire path
    Z = np.random.multivariate_normal(np.zeros(14), np.eye(14), size=n_steps)
    
    for t in range(n_steps):
        # 1. Dynamic Correlation Update (DCC)
        if t > 0:
            eps_prev = std_shocks
            q_t = (1 - dcc_a - dcc_b)*Q_bar + dcc_a * np.outer(eps_prev, eps_prev) + dcc_b * q_t
            # Normalize to valid correlation matrix
            d = np.diag(1 / np.sqrt(np.diag(q_t)))
            R_t = d @ q_t @ d
        else:
            R_t = Q_bar
            
        # Cholesky decomposition of dynamic correlation
        try:
            chol = np.linalg.cholesky(R_t)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(Q_bar) # Fallback if slightly non-pos-def
            
        std_shocks = chol @ Z[t]
        
        # 2. GARCH Volatility Update
        shock_t = np.zeros(14)
        for i in range(14):
            # omega + alpha * prev_shock^2 + beta * prev_var
            # For t=0 we just use the steady state var_t
            shock_t[i] = std_shocks[i] * np.sqrt(var_t[i])
            var_t[i] = garch_params[i, 0] + garch_params[i, 1]*(shock_t[i]**2) + garch_params[i, 2]*var_t[i]
            
        # 3. Step forward the state variables
        dns_shock = shock_t[0:3]
        mkt_shock = shock_t[3]
        stock_shock = shock_t[4:14]
        
        dns_t = dns_t @ dns_var_coef + dns_shock
        mkt_t = mkt_coef[0] + mkt_coef[1]*mkt_t + mkt_shock
        
        # 4. Map back to real rates and stocks
        current_yields = H @ dns_t
        r_path[t] = current_yields
        
        stock_returns = stock_alphas + stock_betas * mkt_t + stock_shock
        current_prices = current_prices * np.exp(stock_returns)
        s_path[t] = current_prices
        
    sim_rates[s] = r_path
    sim_stocks[s] = s_path

print("Simulation complete.")

# ---------------------------------------------------------
# 6. COMPARATIVE SIMULATIONS & PLOTTING (Deliverable 4b)
# ---------------------------------------------------------
print("Running baseline models for comparison...")

# --- A. Historical Simulation (Fast Vectorized) ---
np.random.seed(42)
hs_indices = np.random.randint(0, len(log_returns), size=(n_scenarios, n_steps))
hs_sim_stocks = last_stocks * np.exp(np.cumsum(log_returns.values[hs_indices], axis=1))
hs_sim_rates = last_rates + np.cumsum(rate_diffs.values[hs_indices], axis=1)

# --- B. VAR(1) Simulation ---
import sys
from pathlib import Path

# 1. Point Python to the main L2 folder where common.py and var1.py live
l2_dir = Path(__file__).resolve().parent.parent
if str(l2_dir) not in sys.path:
    sys.path.insert(0, str(l2_dir))

# 2. Import Michael's custom functions
from common import load_data, log_returns
from var1 import simulate_var1, returns_to_levels

# 3. Load Michael's calibrated matrices
var_A = pd.read_csv(l2_dir / 'Q2' / 'var1_A.csv', index_col=0).values.astype(float)
var_c = pd.read_csv(l2_dir / 'Q2' / 'var1_c.csv', index_col=0).values.astype(float).flatten()
var_Sigma = pd.read_csv(l2_dir / 'Q2' / 'var1_Sigma.csv', index_col=0).values.astype(float)

# 4. Load the exact starting states his model expects
m_prices = load_data()
m_R = log_returns(m_prices).to_numpy()
m_R_last = m_R[-1]
m_start_levels = m_prices.iloc[-1].to_numpy()

# 5. Run Michael's simulation engine
rng = np.random.default_rng(42)
m_sim_returns = simulate_var1(var_c, var_A, var_Sigma, m_R_last, n_scenarios, n_steps, rng)
m_levels = returns_to_levels(m_sim_returns, m_start_levels)

# 6. Split his output to feed directly into your plotting loop
var_sim_stocks = m_levels[:, :, :10] 
var_sim_rates = m_levels[:, :, 10:]

percentiles = [5, 10, 25, 50, 75, 90, 95]

def get_pctls(sim_data, asset_idx):
    return np.percentile(sim_data[:, :, asset_idx], percentiles, axis=0)
def get_pctls(sim_data, asset_idx):
    return np.percentile(sim_data[:, :, asset_idx], percentiles, axis=0)

# Indices based on market data: CAT=1, WMT=8, Y05=0, Y30=4
models = {
    'VAR(1)': {'stocks': var_sim_stocks, 'rates': var_sim_rates, 'style': '-', 'alpha': 0.6},
    'HS': {'stocks': hs_sim_stocks, 'rates': hs_sim_rates, 'style': '--', 'alpha': 0.8},
    'AI ESG': {'stocks': sim_stocks, 'rates': sim_rates, 'style': ':', 'alpha': 1.0}
}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
assets = [('CAT', 1, 'stocks'), ('WMT', 8, 'stocks'), ('5Y Rate', 0, 'rates'), ('30Y Rate', 4, 'rates')]

for ax, (name, idx, asset_type) in zip(axes.flatten(), assets):
    time_steps = np.arange(1, n_steps + 1)
    
    for model_name, data in models.items():
        pctls = get_pctls(data[asset_type], idx)
        
        for p_idx, p_val in enumerate(percentiles):
            label = f"{model_name}" if p_val == 50 else None
            ax.plot(time_steps, pctls[p_idx], linestyle=data['style'], alpha=data['alpha'], 
                    color='black' if model_name == 'VAR(1)' else 'blue' if model_name == 'HS' else 'red', 
                    label=label)
            
    ax.set_title(f'Model Comparison: {name}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()