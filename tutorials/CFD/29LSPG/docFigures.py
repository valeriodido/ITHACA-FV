"""Figures of StackedVsSegregatedPOD.md (Sections 6.3, 9 and 10).

Usage: python3 docFigures.py   ->   images/doc/*.png
All numbers are those of the model problem of Section 10.4.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "doc")
os.makedirs(OUT, exist_ok=True)

SEG = "#CC0000"     # segregated
STK = "#0066CC"     # stacked
REF = "#222222"     # best approximation / FOM
GREY = "#999999"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#E5E5E5", "figure.dpi": 130})

# Model problem of Section 10.4
alpha, gamma, eps, ell = 1.0, 0.05, 0.01, 1.0
rhoU, rhoP, g = 0.005, 0.002, 0.3
J = np.array([[alpha, gamma], [0.0, eps]])
r0 = np.array([rhoU, rhoP])


def segregated():
    return -np.linalg.solve(J, r0)


def stacked(delta=0.0, epsilon=eps):
    Jl = np.array([[alpha, gamma], [0.0, epsilon]])
    j = Jl @ np.array([1.0, ell])
    r0p = r0 - Jl @ np.array([0.0, delta])
    e = -(j @ r0p) / (j @ j)
    return np.array([e, ell * e - delta]), r0p + j * e


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.join(OUT, name))


# ---------------------------------------------------------------------------------------------------------------------
# Fig. 1 - Residual space: what each trial space can do to the residual
# ---------------------------------------------------------------------------------------------------------------------
eSeg = segregated()
eStk, rStk = stacked()
jStk = J @ np.array([1.0, ell])

fig, ax = plt.subplots(figsize=(7.2, 4.6))
s = 1e3  # residuals in units of 1e-3
ax.plot(0, 0, "o", color=REF, ms=7, zorder=5)
ax.annotate("r = 0", (0, 0), (6, -14), textcoords="offset points")
ax.plot(*(s * r0), "s", color=REF, ms=7, zorder=5)
ax.annotate(r"$\mathbf{r}_0$" "\n(best approx.)", s * r0, (-72, -30), textcoords="offset points", fontsize=10)

# Segregated: r0 + e_u J_u + e_p J_p = 0, drawn as two legs
leg1 = r0 + eSeg[0] * J[:, 0]
ax.annotate("", s * leg1, s * r0, arrowprops=dict(arrowstyle="->", color=SEG, lw=2))
ax.annotate("", (0, 0), s * leg1, arrowprops=dict(arrowstyle="->", color=SEG, lw=2, ls="--"))
ax.text(s * (r0[0] + leg1[0]) / 2, s * r0[1] + 0.2, rf"$e_u\,\mathbf{{J}}_u$,  $e_u={eSeg[0]:+.3f}$",
        color=SEG, ha="center", va="bottom", fontsize=10)
ax.text(s * leg1[0] / 2 + 0.8, s * leg1[1] / 2 - 0.35, r"$e_p\,\mathbf{J}_p$, $e_p=$" f"{eSeg[1]:+.2f}" "\n"
        r"($\mathbf{J}_p$ is tiny: a 40$\times$ larger coefficient)", color=SEG, fontsize=10, va="top")

# Stacked: the line r0 + t j and the closest point to 0
t = np.linspace(-0.0105, 0.0065, 2)
line = r0[:, None] + jStk[:, None] * t
ax.plot(s * line[0], s * line[1], color=STK, lw=1.2, alpha=0.6)
ax.text(-5.8, 2.9, r"reachable with the stacked coordinate: $\mathbf{r}_0+t\,\mathbf{j}$ (blue line)",
        color=STK, fontsize=10, va="bottom")
ax.annotate("", s * rStk, s * r0, arrowprops=dict(arrowstyle="->", color=STK, lw=2))
ax.plot(*(s * rStk), "o", color=STK, ms=7, zorder=5)
ax.plot([0, s * rStk[0]], [0, s * rStk[1]], ":", color=STK)
ax.annotate("stacked LSPG: closest point,\nresidual left = " r"$(\approx 0,\ \rho_p)$",
            s * rStk, (-150, -42), textcoords="offset points", color=STK, fontsize=10)

ax.set_xlabel(r"momentum residual $r_1$  [$10^{-3}$]")
ax.set_ylabel(r"pressure-eq. residual $r_2$  [$10^{-3}$]")
ax.set_title("Residual space: segregated cancels $\\mathbf{r}_0$, stacked minimises it")
ax.set_aspect("equal")
ax.set_xlim(-6, 12)
ax.set_ylim(-2.5, 3.6)
save(fig, "fig_residual_space.png")

# ---------------------------------------------------------------------------------------------------------------------
# Fig. 2 - Coordinate (error) space: residual contours, drag contours, the two solutions
# ---------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
eu = np.linspace(-0.03, 0.03, 400)
ep = np.linspace(-0.26, 0.06, 400)
EU, EP = np.meshgrid(eu, ep)
R1 = r0[0] + J[0, 0] * EU + J[0, 1] * EP
R2 = r0[1] + J[1, 1] * EP
Rn = np.sqrt(R1**2 + R2**2)
cs = ax.contour(EU, EP, Rn, levels=[0.0005, 0.001, 0.002, 0.004, 0.008, 0.016], colors=GREY, linewidths=0.8)
ax.clabel(cs, fmt=lambda v: f"{v:.4f}", fontsize=8)
ax.axhspan(-0.0333, 0.0333, color="#2ca02c", alpha=0.08)
ax.text(0.029, 0.036, r"$|\delta C_d|<0.01$", color="#2ca02c", ha="right", fontsize=9)

ax.plot(0, 0, "s", color=REF, ms=8, zorder=5, label="best approximation (projection)")
ax.plot(eu, ell * eu, color=STK, lw=1.5, label=r"stacked trial space $e_p=\ell e_u-\delta$")
ax.plot(*eSeg, "o", color=SEG, ms=8, zorder=5, label=f"segregated LSPG: residual 0, $\\delta C_d={g * eSeg[1]:+.3f}$")
ax.plot(*eStk, "o", color=STK, ms=8, zorder=5, label=f"stacked LSPG: residual {np.linalg.norm(rStk):.4f}, "
        f"$\\delta C_d={g * eStk[1]:+.4f}$")
ax.annotate("long, flat valley:\n" r"$\|\mathbf{r}\|$ barely changes with $e_p$", (0.013, -0.06), fontsize=9, color="#555555")
ax.set_xlabel(r"velocity error $e_u$")
ax.set_ylabel(r"pressure error $e_p$   (note the 8$\times$ larger axis range)")
ax.set_title("Coordinate space: residual contours (grey) and the two LSPG solutions")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
save(fig, "fig_error_space.png")

# ---------------------------------------------------------------------------------------------------------------------
# Fig. 3 - Drag error vs. epsilon (pressure-row sensitivity) and vs. delta (violation of the training relation)
# ---------------------------------------------------------------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
epss = np.logspace(-4, 0, 200)
seg_e = [abs(g * rhoP / e) for e in epss]
stk_e = [abs(g * stacked(0.0, e)[0][1]) for e in epss]
a1.loglog(epss, seg_e, color=SEG, lw=2, label=r"segregated: $g\,\rho_p/\varepsilon$")
a1.loglog(epss, stk_e, color=STK, lw=2, label="stacked")
a1.axvline(eps, color=GREY, ls=":")
a1.text(eps * 1.15, 1e-3, r"$\varepsilon$ of Sec. 10.4", color=GREY, fontsize=9)
a1.set_xlabel(r"pressure-row sensitivity $\varepsilon\sim(\kappa h)^2$")
a1.set_ylabel(r"$|\delta C_d|$")
a1.set_title("Smoother pressure error (smaller $\\varepsilon$) $\\to$ larger segregated error", fontsize=10)
a1.legend(fontsize=9)

deltas = np.linspace(0, 0.4, 200)
a2.plot(deltas, [abs(g * eSeg[1])] * len(deltas), color=SEG, lw=2, label="segregated (independent of $\\delta$)")
a2.plot(deltas, [abs(g * stacked(d)[0][1]) for d in deltas], color=STK, lw=2, label="stacked")
dx = deltas[np.argmin(np.abs(np.array([abs(g * stacked(d)[0][1]) for d in deltas]) - abs(g * eSeg[1])))]
a2.axvline(dx, color=GREY, ls=":")
a2.text(dx + 0.01, 0.02, f"crossover $\\delta\\approx{dx:.2f}$", color=GREY, fontsize=9)
a2.axvspan(0, 0.05, color="#2ca02c", alpha=0.08)
a2.text(0.005, 0.105, "in-sample:\n$\\delta$ small", color="#2ca02c", fontsize=9, va="top")
a2.set_xlabel(r"violation of the learned relation $\delta=p^\star-\ell u^\star$ (training RMS units)")
a2.set_ylabel(r"$|\delta C_d|$")
a2.set_title("Stacked wins while the learned $p$–$U$ relation holds", fontsize=10)
a2.legend(fontsize=9)
fig.tight_layout()
save(fig, "fig_drag_error.png")

# ---------------------------------------------------------------------------------------------------------------------
# Fig. 4 - Projection onto one stacked mode (Section 6.3): per-field fits and weighted least squares
# ---------------------------------------------------------------------------------------------------------------------
phiU, phiP = 1.0, 0.6          # the U and p parts of one stacked mode (one value each, for the picture)
dU, dP = 0.8, 0.9              # the deviation to project; not on the mode's line
cU, cP = dU / phiU, dP / phiP  # per-field fits

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, wP, title in [(a1, 1.0, r"weighted coordinates $(\sqrt{w_U}\,U,\ \sqrt{w_p}\,p)$, $w_U=w_p=1$"),
                      (a2, None, r"effect of the weight $w_p$ (physical coordinates)")]:
    t = np.linspace(-0.1, 1.9, 2)
    ax.plot(t * phiU, t * phiP, color=STK, lw=1.5, label=r"stacked mode: $(\hat\Phi^U,\hat\Phi^p)\,c$")
    ax.plot(dU, dP, "s", color=REF, ms=8, zorder=5, label=r"deviation $(d^U,d^p)$")
    ax.plot(cU * phiU, cU * phiP, "^", color="#ff7f0e", ms=9, zorder=5, label=rf"fit $U$ only: $c^U={cU:.2f}$")
    ax.plot(cP * phiU, cP * phiP, "v", color="#7733AA", ms=9, zorder=5, label=rf"fit $p$ only: $c^p={cP:.2f}$")
    ax.plot([dU, cU * phiU], [dP, cU * phiP], ":", color="#ff7f0e")
    ax.plot([dU, cP * phiU], [dP, cP * phiP], ":", color="#7733AA")
    ax.set_xlabel("U component")
    ax.set_ylabel("p component")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.1, 1.3)

# (a) orthogonal projection in the weighted inner product = weighted average of the per-field fits
gU, gP = phiU**2, phiP**2
a = (gU * cU + gP * cP) / (gU + gP)
a1.plot(a * phiU, a * phiP, "o", color=STK, ms=9, zorder=6, label=rf"weighted LS: $\hat a={a:.2f}$")
a1.plot([dU, a * phiU], [dP, a * phiP], "-", color=STK, lw=1)
a1.text(0.02, 1.22, rf"$\hat a=\frac{{\gamma_U c^U+\gamma_p c^p}}{{\gamma_U+\gamma_p}}$, "
        rf"$\gamma_U={gU:.2f}$, $\gamma_p={gP:.2f}$", fontsize=10, va="top")
a1.legend(loc="lower right", fontsize=8.5)

# (b) the weighted-LS point slides from c^U (w_p -> 0) to c^p (w_p -> inf)
for wp, mk, off in [(0.05, "o", (8, -14)), (0.3, "o", (-60, 18)), (1.0, "o", (8, -12)), (3.0, "o", (8, -12)),
                    (20.0, "o", (8, -12))]:
    gUw, gPw = phiU**2, wp * phiP**2
    aw = (gUw * cU + gPw * cP) / (gUw + gPw)
    a2.plot(aw * phiU, aw * phiP, mk, color=STK, ms=6, zorder=6)
    a2.annotate(f"$w_p={wp:g}$", (aw * phiU, aw * phiP), off, textcoords="offset points", fontsize=8.5,
                color=STK, arrowprops=dict(arrowstyle="-", color=STK, lw=0.6))
a2.text(0.02, 1.22, "the projection always lies between the two\nper-field fits; $w_f$ decides where",
        fontsize=10, va="top")
a2.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_projection.png")

# ---------------------------------------------------------------------------------------------------------------------
# Fig. 5 - How the residual rows see a pressure error of wavenumber kappa (Section 9)
# ---------------------------------------------------------------------------------------------------------------------
theta = np.linspace(1e-3, np.pi, 400)            # kappa h
fig, ax = plt.subplots(figsize=(7.2, 4.2))
lap = 1 - np.cos(theta)                           # Jacobi-scaled 1-D Laplacian: (2 - 2cos)/2
ax.semilogy(theta / np.pi, lap, color=SEG, lw=2, label=r"pressure equation, Jacobi-scaled: $1-\cos\kappa h\approx(\kappa h)^2/2$")
grad = np.sin(theta) / 3                          # (2 dt / 3) * sin(kappa h)/h, with dt/h ~ 1/2 (Co ~ 0.5)
ax.semilogy(theta / np.pi, grad, color="#ff7f0e", lw=2, label=r"momentum, through $r_{AU}\nabla p$: $\propto\sin\kappa h$")
ax.semilogy(theta / np.pi, np.ones_like(theta), color=STK, lw=2, ls="--",
            label=r"momentum, through $\partial_t U$ (BDF2 diagonal): $\mathcal{O}(1)$")
ax.axvspan(0, 0.1, color="#2ca02c", alpha=0.08)
ax.text(0.005, 2e-4, "smooth errors\n(base pressure, drag)", color="#2ca02c", fontsize=9)
ax.text(0.75, 3, "checkerboard", fontsize=9, color=GREY)
ax.set_xlabel(r"wavenumber of the error, $\kappa h/\pi$")
ax.set_ylabel("sensitivity of the weighted residual")
ax.set_ylim(1e-5, 10)
ax.set_title("Who can see a pressure error? (1-D model, unit amplitude)", fontsize=11)
ax.legend(fontsize=8.5, loc="lower right")
save(fig, "fig_sensitivity.png")
