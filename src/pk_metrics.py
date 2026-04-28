import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Modelo
# =========================
def simulate(g):
    dose_mg = 100
    tau = 8
    t_max = 48
    dt = 0.01

    alpha_cl = 0.3
    alpha_q = 0.7
    alpha_vc = 0.1
    alpha_vp = 0.2

    Vc_base = 10.0
    Vp_base = 30.0
    CL_base = 5.0
    Q_base = 8.0

    Vc = Vc_base * (g ** (-alpha_vc))
    Vp = Vp_base * (g ** (-alpha_vp))
    CL = CL_base * (g ** alpha_cl)
    Q = Q_base * (g ** alpha_q)

    k10 = CL / Vc
    k12 = Q / Vc
    k21 = Q / Vp

    t = np.arange(0, t_max + dt, dt)

    A1 = np.zeros_like(t)
    A2 = np.zeros_like(t)

    next_dose_time = 0

    for i in range(1, len(t)):
        if t[i - 1] >= next_dose_time:
            A1[i - 1] += dose_mg
            next_dose_time += tau

        a1 = A1[i - 1]
        a2 = A2[i - 1]

        dA1_dt = -k10 * a1 - k12 * a1 + k21 * a2
        dA2_dt =  k12 * a1 - k21 * a2

        A1[i] = a1 + dA1_dt * dt
        A2[i] = a2 + dA2_dt * dt

    C1 = A1 / Vc
    C2 = A2 / Vp

    return t, C1, C2, {
        "g": g,
        "Vc_L": Vc,
        "Vp_L": Vp,
        "CL_L_h": CL,
        "Q_L_h": Q,
        "k10_1_h": k10,
        "k12_1_h": k12,
        "k21_1_h": k21,
    }


# =========================
# Métricas PK
# =========================
def compute_metrics(g):
    t, C1, C2, params = simulate(g)

    auc_central = np.trapezoid(C1, t)
    auc_peripheral = np.trapezoid(C2, t)
    
    cmax_central = np.max(C1)
    cmax_peripheral = np.max(C2)

    tmax_central = t[np.argmax(C1)]
    tmax_peripheral = t[np.argmax(C2)]

    # Valle final: concentración inmediatamente antes de la última dosis.
    # Dosis cada 8 h: 0, 8, 16, 24, 32, 40, 48.
    # Tomamos justo antes de 40 h como valle representativo dentro del periodo.
    trough_time = 40 - 0.01
    trough_idx = np.argmin(np.abs(t - trough_time))

    trough_central = C1[trough_idx]
    trough_peripheral = C2[trough_idx]

    ratio_auc_tissue_plasma = auc_peripheral / auc_central

    row = {
        **params,
        "Cmax_central_mg_L": cmax_central,
        "Tmax_central_h": tmax_central,
        "AUC_central_mg_h_L": auc_central,
        "Ctrough_central_mg_L": trough_central,
        "Cmax_peripheral_mg_L": cmax_peripheral,
        "Tmax_peripheral_h": tmax_peripheral,
        "AUC_peripheral_mg_h_L": auc_peripheral,
        "Ctrough_peripheral_mg_L": trough_peripheral,
        "AUC_tissue_plasma_ratio": ratio_auc_tissue_plasma,
    }

    return row


# =========================
# Escenarios
# =========================
g_values = [0.01, 0.16, 0.38, 1.0, 2, 5, 10, 100]

rows = [compute_metrics(g) for g in g_values]
df = pd.DataFrame(rows)

# Orden útil para paper
df = df[
    [
        "g",
        "Vc_L",
        "Vp_L",
        "CL_L_h",
        "Q_L_h",
        "k10_1_h",
        "k12_1_h",
        "k21_1_h",
        "Cmax_central_mg_L",
        "Tmax_central_h",
        "AUC_central_mg_h_L",
        "Ctrough_central_mg_L",
        "Cmax_peripheral_mg_L",
        "Tmax_peripheral_h",
        "AUC_peripheral_mg_h_L",
        "Ctrough_peripheral_mg_L",
        "AUC_tissue_plasma_ratio",
    ]
]

df.to_csv("pk_metrics_by_gravity.csv", index=False)

print("\n=== Métricas PK por gravedad ===")
print(df.round(3).to_string(index=False))


# =========================
# Figura 1: exposición central y tisular
# =========================
plt.figure(figsize=(8, 5))

plt.plot(df["g"], df["AUC_central_mg_h_L"], marker="o", label="AUC plasma / central")
plt.plot(df["g"], df["AUC_peripheral_mg_h_L"], marker="o", label="AUC tejido / periférico")

plt.xscale("log")
plt.xlabel("Gravedad relativa (g)")
plt.ylabel("AUC (mg·h/L)")
plt.title("Exposición sistémica y tisular según gravedad")
plt.grid(True, alpha=0.35)
plt.legend()

plt.tight_layout()
plt.savefig("metric_auc_vs_gravity.png", dpi=300, bbox_inches="tight")
plt.savefig("metric_auc_vs_gravity.pdf", bbox_inches="tight")
plt.show()


# =========================
# Figura 2: razón tejido/plasma
# =========================
plt.figure(figsize=(8, 5))

plt.plot(
    df["g"],
    df["AUC_tissue_plasma_ratio"],
    marker="o",
)

plt.xscale("log")
plt.xlabel("Gravedad relativa (g)")
plt.ylabel("Razón AUC tejido / AUC plasma")
plt.title("Desacoplamiento relativo tejido-plasma según gravedad")
plt.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig("metric_auc_ratio_vs_gravity.png", dpi=300, bbox_inches="tight")
plt.savefig("metric_auc_ratio_vs_gravity.pdf", bbox_inches="tight")
plt.show()


# =========================
# Figura 3: Cmax y valle central
# =========================
plt.figure(figsize=(8, 5))

plt.plot(df["g"], df["Cmax_central_mg_L"], marker="o", label="Cmax plasma")
plt.plot(df["g"], df["Ctrough_central_mg_L"], marker="o", label="Cvalle plasma")

plt.xscale("log")
plt.xlabel("Gravedad relativa (g)")
plt.ylabel("Concentración (mg/L)")
plt.title("Concentraciones plasmáticas máximas y valle según gravedad")
plt.grid(True, alpha=0.35)
plt.legend()

plt.tight_layout()
plt.savefig("metric_cmax_trough_vs_gravity.png", dpi=300, bbox_inches="tight")
plt.savefig("metric_cmax_trough_vs_gravity.pdf", bbox_inches="tight")
plt.show()