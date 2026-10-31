import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# 1. إعدادات الشبكة والمحاكاة (GPU Version)
# ============================================================
N = 256             # حجم الشبكة الكبير للجامعة
L = 2 * cp.pi
dx = L / N
dt = 0.0005         # خطوة زمنية أصغر لضمان الاستقرار للشبكات الكبيرة
N_steps = 1000

x = cp.linspace(0, L, N, endpoint=False)
X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

kx = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2
K2[0,0,0] = 1.0

gamma = 1.0
nu_eff = 1e-4
nu_mol = 1e-4
beta_0 = 1.0
beta_VS = 2.5

# ============================================================
# 2. دوال مؤثر WENO5 للحمل (GPU)
# ============================================================
def weno5_derivative_gpu(f, axis, dx):
    f_m = cp.moveaxis(f, axis, 0)
    shape = f_m.shape
    N_x = shape[0]
    
    f_ext = cp.zeros((N_x + 4,) + shape[1:], dtype=f.dtype)
    f_ext[2:-2] = f_m
    f_ext[0:2] = f_m[-2:]
    f_ext[-2:] = f_m[0:2]
    
    fm2, fm1, f0, fp1, fp2 = f_ext[0:N_x], f_ext[1:N_x+1], f_ext[2:N_x+2], f_ext[3:N_x+3], f_ext[4:N_x+4]
    
    eps = 1e-6
    q0 = (2*fm2 - 7*fm1 + 11*f0) / 6.0
    q1 = (-fm1 + 5*f0 + 2*fp1) / 6.0
    q2 = (2*f0 + 5*fp1 - fp2) / 6.0
    
    b0 = 13/12*(fm2 - 2*fm1 + f0)**2 + 1/4*(fm2 - 4*fm1 + 3*f0)**2
    b1 = 13/12*(fm1 - 2*f0 + fp1)**2 + 1/4*(fm1 - fp1)**2
    b2 = 13/12*(f0 - 2*fp1 + fp2)**2 + 1/4*(3*f0 - 4*fp1 + fp2)**2
    
    g0, g1, g2 = 1/10, 6/10, 3/10
    a0 = g0 / (eps + b0)**2
    a1 = g1 / (eps + b1)**2
    a2 = g2 / (eps + b2)**2
    asum = a0 + a1 + a2
    
    w0, w1, w2 = a0/asum, a1/asum, a2/asum
    f_plus = (w0*q0 + w1*q1 + w2*q2)
    deriv = (cp.roll(f_plus, -1, axis=0) - f_plus) / dx
    
    return cp.moveaxis(deriv, 0, axis)

def laplacian_spectral_gpu(f):
    return cp.fft.ifftn(-K2 * cp.fft.fftn(f)).real

def compute_velocity_gpu(omega):
    u = cp.zeros_like(omega)
    ox_hat, oy_hat, oz_hat = cp.fft.fftn(omega[0]), cp.fft.fftn(omega[1]), cp.fft.fftn(omega[2])
    K2_safe = cp.where(K2 == 0, 1.0, K2)
    u[0] = cp.fft.ifftn(1j * (KY * oz_hat - KZ * oy_hat) / K2_safe).real
    u[1] = cp.fft.ifftn(1j * (KZ * ox_hat - KX * oz_hat) / K2_safe).real
    u[2] = cp.fft.ifftn(1j * (KX * oy_hat - KY * ox_hat) / K2_safe).real
    return u

# ============================================================
# 3. دالة حساب المشتقات (RHS - GPU)
# ============================================================
def compute_rhs_gpu(A, omega):
    u = compute_velocity_gpu(omega)
    
    dAdx = weno5_derivative_gpu(A, axis=0, dx=dx)
    dAdy = weno5_derivative_gpu(A, axis=1, dx=dx)
    dAdz = weno5_derivative_gpu(A, axis=2, dx=dx)
    adv_A = -(u[0]*dAdx + u[1]*dAdy + u[2]*dAdz)
    
    lap_A = laplacian_spectral_gpu(A)
    omega_mag2 = omega[0]**2 + omega[1]**2 + omega[2]**2
    beta_eff = beta_0 + beta_VS * omega_mag2
    nonlinear = -beta_eff * (A**3)
    
    RHS_A = gamma * A + nonlinear + adv_A + nu_eff * lap_A
    
    RHS_omega = cp.zeros_like(omega)
    for i in range(3):
        dw_dx = weno5_derivative_gpu(omega[i], axis=0, dx=dx)
        dw_dy = weno5_derivative_gpu(omega[i], axis=1, dx=dx)
        dw_dz = weno5_derivative_gpu(omega[i], axis=2, dx=dx)
        adv_w = -(u[0]*dw_dx + u[1]*dw_dy + u[2]*dw_dz)
        
        du_dx = weno5_derivative_gpu(u[i], axis=0, dx=dx)
        du_dy = weno5_derivative_gpu(u[i], axis=1, dx=dx)
        du_dz = weno5_derivative_gpu(u[i], axis=2, dx=dx)
        vs = omega[0]*du_dx + omega[1]*du_dy + omega[2]*du_dz
        
        lap_w = laplacian_spectral_gpu(omega[i])
        RHS_omega[i] = adv_w + vs + nu_mol * lap_w
        
    return RHS_A, RHS_omega

# ============================================================
# 4. الشروط الابتدائية (GPU)
# ============================================================
A = 0.1 * cp.cos(X + Y + Z) + 0.05 * cp.random.randn(N, N, N)

omega = cp.zeros((3, N, N, N))
omega[0] = cp.sin(X) * cp.cos(Y) * cp.cos(Z)
omega[1] = -cp.cos(X) * cp.sin(Y) * cp.cos(Z)
omega[2] = 0.0

# ============================================================
# 5. الحلقة الرئيسية (RK2 المُصححة)
# ============================================================
print(f"Starting GPU-Accelerated Simulation on {N}^3 grid...")
cp.cuda.Stream.null.synchronize()
start_time = time.time()

for step in range(N_steps):
    # المرحلة الأولى (Predictor)
    RHS_A1, RHS_omega1 = compute_rhs_gpu(A, omega)
    A_1 = A + dt * RHS_A1
    omega_1 = omega + dt * RHS_omega1
    
    # المرحلة الثانية (Corrector)
    RHS_A2, RHS_omega2 = compute_rhs_gpu(A_1, omega_1)
    
    # التحديث النهائي
    A = A + 0.5 * dt * (RHS_A1 + RHS_A2)
    omega = omega + 0.5 * dt * (RHS_omega1 + RHS_omega2)
    
    if step % 100 == 0:
        A_max = cp.max(A).item()
        omega_max = cp.max(cp.abs(omega)).item()
        cp.cuda.Stream.null.synchronize()
        print(f"Step {step:4d} | A_max = {A_max:.4f} | omega_max = {omega_max:.4f}")

cp.cuda.Stream.null.synchronize()
end_time = time.time()
print(f"Simulation complete in {end_time - start_time:.2f} seconds.")

# ============================================================
# 6. عرض النتائج (نقل البيانات للـ CPU للرسم)
# ============================================================
print("Generating plots...")
A_cpu = cp.asnumpy(A)
omega_mag_cpu = cp.asnumpy(cp.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im1 = axes[0].imshow(A_cpu[:, :, N//2].T, origin='lower', cmap='hot')
axes[0].set_title('Amplitude A (Midplane)')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(omega_mag_cpu[:, :, N//2].T, origin='lower', cmap='viridis')
axes[1].set_title('Vorticity Magnitude |ω| (Midplane)')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
