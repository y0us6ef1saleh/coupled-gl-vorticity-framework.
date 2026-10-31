import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class GLVorticityModel:
    def __init__(self, gamma=1.0, beta_0=1.0, beta_VS=2.5):
        self.gamma = gamma
        self.beta_0 = beta_0
        self.beta_VS = beta_VS

    def _odes(self, t, y):
        A, omega = y
        beta_eff = self.beta_0 + self.beta_VS * (omega**2)
        dA_dt = self.gamma * A - beta_eff * (A**3)
        domega_dt = -0.5 * omega 
        return [dA_dt, domega_dt]

    def solve(self, t_span=(0, 10), A0=0.1, omega0=0.5, num_points=200):
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        sol = solve_ivp(self._odes, t_span, [A0, omega0], t_eval=t_eval, method='RK45')
        return sol.t, sol.y[0], sol.y[1]

# بيانات NACA 0012 التجريبية (تقريبية من تقرير Becker 1940)
NACA_X = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
NACA_AMP = np.array([1.25, 1.15, 1.10, 1.05, 0.95, 0.85, 0.75, 0.65, 0.55, 0.48])

def calibrate_model(x_data, y_data):
    def residuals(params):
        beta_0, beta_VS = params
        # استخدام (-) لضمان أن التأثير يخفض المنحنى
        predicted = np.exp(-beta_0 * x_data) - beta_VS * (x_data**2) + 0.2
        return predicted - y_data

    initial_guess = [1.0, 0.1]
    bounds = ([0.0, 0.0], [10.0, 5.0]) 
    
    result = least_squares(residuals, initial_guess, bounds=bounds)
    beta_0_cal, beta_VS_cal = result.x
    
    ss_res = np.sum(result.fun**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return beta_0_cal, beta_VS_cal, r_squared
