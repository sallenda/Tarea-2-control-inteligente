import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize



N0, NS = 1.0, 1.52
NH, NL = 2.35, 1.46
N = 100

LOW, HIGH = 10.0, 300.0

WAVELENGTHS = np.arange(500.0, 601.0)
PASS = (540.0 <= WAVELENGTHS) & (WAVELENGTHS <= 560.0)
STOP = ~PASS

def indices(d):
    return np.where(np.arange(d) % 2 == 0, NH, NL)

def transmittance(t):
    t = np.asarray(t, dtype=float)

    B = np.ones_like(WAVELENGTHS, dtype=complex)
    C = NS * B

    for n, thickness in zip(indices(len(t)), t):
        delta = 2 * np.pi * n * thickness / WAVELENGTHS

        c = np.cos(delta)
        s = np.sin(delta)

        B, C = (
            c * B + 1j * s * C / n,
            1j * n * s * B + c * C,
        )

    R = np.abs((N0 * B - C) / (N0 * B + C))**2

    return 1 - R

def objective(t):
    T = transmittance(t)

    J_pass = np.mean((1 - T[PASS])**2)
    J_stop = np.mean(T[STOP]**2)

    return float(J_pass + J_stop)

def noisy_objective(t, rng, sigma=0.03):
    T = transmittance(t)
    T_measured = T + sigma * rng.standard_normal(T.size)

    J_pass = np.mean((1 - T_measured[PASS])**2)
    J_stop = np.mean(T_measured[STOP]**2)

    return float(J_pass + J_stop)

if __name__ == "__main__":
    T0 = 4 * N0 * NS / (N0 + NS)**2

    assert np.allclose(transmittance([]), T0, atol=1e-12)
    assert np.all(np.isfinite(transmittance(np.full(13, 100.0))))

    print(f"PASS: d=0 T={T0:.12f}; P={PASS.sum()}, S={STOP.sum()}")




    ### Parte a

    ### Verificacion de que en ausencia de capas modelo coincide con T0

    T_sincapas = transmittance([])          
    print("T0 analítico:", T0)
    print("T modelo (primeros 5):", T_sincapas[:5])
    print("Diferencia máxima:", np.max(np.abs(T_sincapas - T0)))
    print("Son iguales:", np.allclose(T_sincapas, T0, atol=1e-12))

    plt.plot(WAVELENGTHS, T_sincapas, label="Modelo (0 capas)")
    plt.axhline(T0, ls="--", color="k", label="T0 analítico")
    plt.xlabel("λ [nm]")
    plt.ylabel("T")
    plt.legend()
    plt.show()


    ###Filtro de 2 capas con espesores t1 y t2, y busqueda de los espesores que minimizan J(t1, t2)

    J = np.zeros((151, 151))

    t1_values = np.linspace(LOW, HIGH, 151)
    t2_values = np.linspace(LOW, HIGH, 151)

    for i, t1 in enumerate(t1_values):
        for j, t2 in enumerate(t2_values):
            J[i, j] = objective([t1, t2])


    T1, T2 = np.meshgrid(t1_values, t2_values, indexing="ij")


    
    plt.contourf(T1, T2, J, levels=50, cmap="viridis")
    plt.colorbar(label="J")
    plt.xlabel("$t_1$ [nm]")
    plt.ylabel("$t_2$ [nm]")
    plt.title("J(t1, t2) para filtro de 2 capas")
    plt.show()


    ##Al ojo, se ve que los puntos (t1, t2) = (50,10),(170,10),(60,190),(170,190) 
    # son puntos de bajo costo.

    puntos_bc = [(50,10),(170,10),(60,190),(170,190)]

    for t1_bc, t2_bc in puntos_bc:
        # Punto de la cuadrícula más cercano
        i = np.argmin(np.abs(t1_values - t1_bc))
        j = np.argmin(np.abs(t2_values - t2_bc))
        t0 = np.array([t1_values[i], t2_values[j]])
        J0 = J[i, j]

        res = minimize(objective, x0=t0, method="L-BFGS-B", bounds=[(LOW, HIGH)] * 2)

        print(f"Antes:   t = ({t0[0]:7.2f}, {t0[1]:7.2f})  J = {J0:.6f}")
        print(f"Después: t = ({res.x[0]:7.2f}, {res.x[1]:7.2f})  J = {res.fun:.6f}  ({res.message})")
        print()



    










