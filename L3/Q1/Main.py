import numpy as np
import matplotlib.pyplot as plt

months = 600 # 50 years into the future

# Initialize arrays with zeros for t=0 to t=600
pop_pp = np.zeros(months + 1)
pop_po = np.zeros(months + 1)
pop_de = np.zeros(months + 1)
pop_tr = np.zeros(months + 1)

t_pp_de = np.zeros(months + 1)
t_po_de = np.zeros(months + 1)
t_pp_tr = np.zeros(months + 1)
t_pp_po = np.zeros(months + 1)

pop_pp[0] = 1.0 # Initial state

#S_m = 1-(1-S_t)**(1/12)
S_m = 0.00168214 # 2% yearly transision rate

# Makeham parameters (Women)
alpha = 0.000739
beta = 5.691932e-07
gamma = 0.13631

# Define Makeham survival function
def L(x):
    return np.exp(-alpha * x - (beta / gamma) * (np.exp(gamma * x) - 1))
qx = np.zeros(months + 1)

# Pre-calculate monthly death probabilities for the next 600 months
starting_age = 60 # From Assignment 2
for t in range(1, months + 1):
    age_end = starting_age + (t / 12)
    age_start = starting_age + ((t - 1) / 12)
    
    qx[t] = 1 - (L(age_end) / L(age_start))

# Calculate the exact integer month the policyholder retires here
retirement_month = (65 - starting_age) * 12 + 1

for t in range(1, months + 1):
    # 1. Death transitions (using your qx[t] array)
    t_pp_de[t] = pop_pp[t-1] * qx[t]
    t_po_de[t] = pop_po[t-1] * qx[t]
    
    # 2. Transfer transitions
    t_pp_tr[t] = (pop_pp[t-1] - t_pp_de[t]) * S_m
    
    # 3. Payout transitions
    if t == retirement_month:
        t_pp_po[t] = pop_pp[t-1] - t_pp_de[t] - t_pp_tr[t]
    else:
        t_pp_po[t] = 0
        
    # 4. Update total populations
    pop_pp[t] = pop_pp[t-1] - t_pp_de[t] - t_pp_tr[t] - t_pp_po[t]
    pop_po[t] = pop_po[t-1] - t_po_de[t] + t_pp_po[t]
    pop_de[t] = pop_de[t-1] + t_pp_de[t] + t_po_de[t]
    pop_tr[t] = pop_tr[t-1] + t_pp_tr[t]

time_axis = np.arange(months + 1)

plt.figure(figsize=(10, 6))
plt.stackplot(time_axis, pop_pp, pop_po, pop_tr, pop_de, 
              labels=['Premium (PP)', 'Payout (PO)', 'Transfer (TR)', 'Death (DE)'])
plt.legend(loc='upper right')
plt.title('State Model Populations Over 50 Years')
plt.xlabel('Months (t)')
plt.ylabel('Population Fraction')
plt.margins(x=0, y=0)
plt.show()