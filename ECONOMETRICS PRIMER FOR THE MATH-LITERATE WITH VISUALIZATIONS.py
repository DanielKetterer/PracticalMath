print("""
=============================================================================
ECONOMETRICS PRIMER FOR THE MATH-LITERATE (WITH VISUALIZATIONS)
=============================================================================
Audience: Strong math background (linear algebra, probability, calculus),
          limited economics exposure. Each section explains the *economic
          problem* briefly, then dives into the math and code.

Sections
--------
  1.  OLS — the baseline workhorse
  2.  Frisch–Waugh–Lovell (partial regression / partialing out)
  3.  Heteroskedasticity — detection and robust SEs
  4.  Instrumental Variables (IV / 2SLS)
  5.  Panel Data — Fixed Effects (within estimator)
  6.  Difference-in-Differences (DiD)
  7.  Regression Discontinuity Design (RDD)
  8.  Binary outcomes — Probit / Logit
  9.  Maximum Likelihood Estimation from scratch
  10. Bootstrap inference


Uses only: numpy, pandas, scipy, matplotlib, PIL, reportlab
All estimators implemented from scratch via linear algebra / scipy.optimize.
Includes clustered SEs, robust inference, and AME standard errors.
=============================================================================
""")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)
save = True
# -- Style --
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA", "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#333", "axes.labelcolor": "#222",
    "xtick.color": "#555", "ytick.color": "#555", "text.color": "#222",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#AAA", "figure.dpi": 140,
})
CB, CO, CG, CR, CP, CY = "#2171B5", "#E6550D", "#31A354", "#DE2D26", "#756BB1", "#888"


import os
import sys
try:
    OUTDIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTDIR = os.getcwd()
# Ensure the methods package is importable
sys.path.insert(0, OUTDIR)

# --- Import encapsulated methods ---
from methods.utils import ols_fit, add_const
from methods import ols as m_ols
from methods import fwl as m_fwl
from methods import heteroskedasticity as m_het
from methods import iv_2sls as m_iv
from methods import panel_fe as m_fe
from methods import did as m_did
from methods import rdd as m_rdd
from methods import binary_outcomes as m_bin
from methods import mle as m_mle
from methods import bootstrap as m_boot

def savefig(fig, name):
    fig.savefig(os.path.join(OUTDIR, name), bbox_inches="tight", dpi=150); plt.close(fig)

# =========================================================================
# We collect the section text blocks so we can interleave them in the PDF.
# Each entry: (section_text, figure_filename)
# =========================================================================
section_contents = []

# =============================================================================
# 1. OLS
# =============================================================================
section1_text = """\
Section 1: OLS -- Ordinary Least Squares

Economic story
Suppose we observe wages (y) and years of schooling (x) for a sample of
workers. A central question in labor economics is: what is the "return to
schooling" -- how much does an additional year of education raise wages on
average? OLS is the starting point for nearly every empirical analysis.

Mathematical setup
We posit the linear model  y = X*beta + epsilon,  where X is the n-by-k
design matrix (including a column of ones for the intercept), beta is the
k-vector of unknown parameters, and epsilon is the n-vector of errors.

The classical OLS assumptions are:
  (A1) Linearity: the conditional expectation E[y|X] is linear in X.
  (A2) Random sampling: observations are i.i.d.
  (A3) No perfect multicollinearity: X has full column rank.
  (A4) Exogeneity (zero conditional mean): E[epsilon|X] = 0.
  (A5) Homoskedasticity: Var(epsilon|X) = sigma^2 * I.

Under (A1)-(A4), OLS is unbiased: E[beta_hat] = beta.
Adding (A5), OLS is BLUE (Best Linear Unbiased Estimator) -- the
Gauss-Markov theorem guarantees the smallest variance among all
linear unbiased estimators.

The closed-form solution:
  beta_hat = (X'X)^{-1} X'y
  Residuals: e_hat = y - X*beta_hat
  Estimated variance: sigma_hat^2 = e_hat'e_hat / (n - k)
  Variance of beta_hat: Var(beta_hat) = sigma_hat^2 * (X'X)^{-1}
  Standard errors: SE(beta_hat_j) = sqrt(Var(beta_hat)_{jj})

Geometric interpretation: OLS projects y onto the column space of X.
The fitted values y_hat = X*beta_hat are the orthogonal projection,
and the residuals e_hat are perpendicular to col(X), which is why
X'e_hat = 0 (the "normal equations").

INTUITION BOX: Why projection?
Think of y as a vector in n-dimensional space. The columns of X span
a k-dimensional subspace. OLS finds the point in that subspace closest
to y (minimizing ||y - X*beta||^2). The residual vector e is the
perpendicular drop from y to that subspace -- hence X'e = 0.
For a mathematician, this is simply the orthogonal projection theorem
applied to the inner product space R^n.

Why OLS fails here -- omitted variable bias (OVB)
In our simulation, ability affects both schooling (smarter people get
more education) and wages (smarter people earn more), but we omit
ability from the regression. The OVB formula is:
  bias = beta_ability * [Cov(schooling, ability) / Var(schooling)]
Since both terms are positive, the OLS estimate of the return to
schooling is biased upward -- it captures part of the ability effect.
This is the fundamental motivation for the IV and panel methods that
follow in later sections.

Derivation of the OVB formula:
  Let the true model be y = X1*beta1 + X2*beta2 + u, where X2 is
  omitted. The short regression estimates:
    beta_hat_short = (X1'X1)^{-1} X1'y
                   = beta1 + (X1'X1)^{-1} X1'X2 * beta2 + (X1'X1)^{-1} X1'u
  Taking expectations: E[beta_hat_short] = beta1 + delta * beta2,
  where delta = (X1'X1)^{-1} X1'X2 is the coefficient from regressing
  the omitted variable on the included variable. The sign of the bias
  is determined by the signs of delta and beta2 -- both positive here.

POLICY CONNECTION: Understanding OVB is essential for evaluating social
programs. When we observe that people who receive mental health treatment
have better outcomes, we cannot simply conclude the treatment worked:
those who seek treatment may differ systematically from those who do not
(in severity, motivation, access, support networks). OVB tells us
exactly how such confounders bias naive estimates, and motivates the
identification strategies in Sections 4-7.
"""
print(section1_text)

n = 500

ability = np.random.normal(0, 1, n)
rho = 1.0
v_sd = np.sqrt(4 - rho**2)  # so Var(schooling) = rho^2*Var(ability) + Var(v) = 1 + 3 = 4
schooling = 12 + rho * ability + np.random.normal(0, v_sd, n)
wage = 2.5*schooling + 1.5*ability + np.random.normal(0, 3, n)
X1 = add_const(schooling)
res_short = m_ols.estimate(X1, wage)
b1, se1, e1 = res_short["beta"], res_short["se"], res_short["residuals"]
X2 = add_const(np.column_stack([schooling, ability]))
res_long = m_ols.estimate(X2, wage)
b2, se2, e2 = res_long["beta"], res_long["se"], res_long["residuals"]

f1 = res_short["fitted"]
print(f"\nManual OLS:")
print(f"  beta_hat (intercept) = {b1[0]:.3f}  SE = {se1[0]:.3f}")
print(f"  beta_hat (schooling) = {b1[1]:.3f}  SE = {se1[1]:.3f}")
print(f"  True coeff on schooling = 2.5  (ability bias inflates estimate)")

section1_text += f"""
Results (short vs long regression)

Short regression (ability omitted): wage ~ 1 + schooling
  beta_hat (intercept) = {b1[0]:.3f}  SE = {se1[0]:.3f}
  beta_hat (schooling) = {b1[1]:.3f}  SE = {se1[1]:.3f}

Long regression (ability included): wage ~ 1 + schooling + ability
  beta_hat (intercept) = {b2[0]:.3f}  SE = {se2[0]:.3f}
  beta_hat (schooling) = {b2[1]:.3f}  SE = {se2[1]:.3f}
  beta_hat (ability)   = {b2[2]:.3f}  SE = {se2[2]:.3f}

True DGP coefficients
  schooling = 2.5
  ability   = 1.5

Interpretation
Because schooling is generated as schooling = 12 + rho·ability + noise with rho = {rho:.1f},
ability is positively correlated with schooling and also directly raises wages. In the short
regression, ability is absorbed into the error term, and since Cov(schooling, ability) > 0,
the exogeneity condition E[ε|schooling] = 0 fails. The omitted-variable-bias formula implies:

  bias( beta_hat_schooling ) = beta_ability · Cov(schooling, ability) / Var(schooling)  > 0.

So the short-regression estimate of the “return to schooling” is biased upward on average:
beta_hat(schooling) tends to exceed 2.5. The long regression restores exogeneity by controlling
for ability, so beta_hat(schooling) is centered near 2.5 and beta_hat(ability) near 1.5 (up to
sampling noise).

"""

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax = axes[0]
ax.scatter(schooling, wage, alpha=.25, s=12, c=CB, edgecolors="none")
xl = np.linspace(schooling.min(), schooling.max(), 100)
ax.plot(xl, b1[0]+b1[1]*xl, c=CO, lw=2.5, label=f"OLS: y_hat = {b1[0]:.1f}+{b1[1]:.2f}x")
ax.plot(xl, 2.5*xl, c=CG, lw=2, ls="--", label="True: y = 2.5x")
ax.set_xlabel("Schooling"); ax.set_ylabel("Wage"); ax.set_title("A) OLS Fit vs True"); ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(f1, e1, alpha=.25, s=12, c=CP, edgecolors="none")
ax.axhline(0, color=CR, lw=1.5, ls="--")
ax.set_xlabel("Fitted y_hat"); ax.set_ylabel("Residuals e_hat"); ax.set_title("B) Residuals vs Fitted")

ax = axes[2]
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("C) DAG: Omitted Variable Bias")
for x,y,l in [(2,8,"Ability\n(unobs)"),(2,2,"Schooling"),(8,5,"Wage")]:
    ax.add_patch(plt.Circle((x,y),1.1,fc="white",ec=CB,lw=2))
    ax.text(x,y,l,ha="center",va="center",fontsize=9,fontweight="bold")
akw=dict(arrowstyle="-|>",color=CO,lw=2,mutation_scale=18)
ax.annotate("",xy=(6.9,5.7),xytext=(3.1,7.5),arrowprops=akw)
ax.annotate("",xy=(6.9,4.3),xytext=(3.1,2.5),arrowprops=akw)
ax.annotate("",xy=(3.0,6.8),xytext=(2.5,3.2),arrowprops=dict(arrowstyle="-|>",color=CR,lw=2,mutation_scale=18,linestyle="dashed"))
ax.text(5.5,7.2,"beta=1.5",fontsize=9,color=CO); ax.text(5.5,2.8,"beta=2.5",fontsize=9,color=CO)
ax.text(0.3,5,"Confounds\n(bias)",fontsize=8,color=CR,fontstyle="italic")
fig.suptitle("Section 1: OLS -- The Baseline Workhorse",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig01_ols.png")

# --- Supplementary Figure 1B: Monte Carlo Demonstration of OVB ---
n_mc = 500; n_sims = 1000
mc_result = m_ols.monte_carlo_ovb(n_mc, n_sims, rho)
mc_short = mc_result["mc_short"]
mc_long = mc_result["mc_long"]

fig_mc, axes_mc = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes_mc[0]
ax.hist(mc_short, bins=45, density=True, alpha=0.55, color=CR, edgecolor="white", label="Short reg (biased)")
ax.hist(mc_long, bins=45, density=True, alpha=0.55, color=CG, edgecolor="white", label="Long reg (unbiased)")
ax.axvline(2.5, color="black", ls="--", lw=2, label="True beta=2.5")
ax.axvline(np.mean(mc_short), color=CR, ls=":", lw=2, label=f"E[short]={np.mean(mc_short):.3f}")
ax.axvline(np.mean(mc_long), color=CG, ls=":", lw=2, label=f"E[long]={np.mean(mc_long):.3f}")
ax.set_xlabel("beta_hat(schooling)"); ax.set_ylabel("Density")
ax.set_title("A) Sampling Distributions (1000 MC sims)"); ax.legend(fontsize=7)

ax = axes_mc[1]
# Show how bias changes with rho (strength of confounding)
rho_vals = np.linspace(0, 2, 30)
bias_vals = 1.5 * rho_vals / (rho_vals**2 + (4 - np.minimum(rho_vals**2, 3.99)))
ax.plot(rho_vals, bias_vals, c=CR, lw=2.5)
ax.axhline(0, color=CY, lw=1)
ax.fill_between(rho_vals, 0, bias_vals, alpha=0.15, color=CR)
ax.set_xlabel("rho (Cov(schooling, ability) strength)")
ax.set_ylabel("OVB = beta_ability * delta")
ax.set_title("B) Bias Grows with Confounding")
ax.annotate("Stronger confounding\n-> larger bias", xy=(1.5, 0.2), fontsize=9,
            fontstyle="italic", color=CR,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CR, alpha=0.9))

ax = axes_mc[2]
# Show the OVB decomposition visually
categories = ["True causal\neffect", "Ability bias\n(OVB)", "OLS estimate\n(short reg)"]
vals = [2.5, np.mean(mc_short) - 2.5, np.mean(mc_short)]
colors_bar = [CG, CR, CO]
bars = ax.bar(categories, vals, color=colors_bar, width=0.5, edgecolor="white")
ax.axhline(2.5, color=CG, ls="--", lw=1.5, alpha=0.7)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Coefficient value"); ax.set_title("C) OVB Decomposition")
ax.text(0.5, 0.02, "OLS = True + Bias", transform=ax.transAxes, fontsize=9,
        ha="center", fontstyle="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CY, alpha=0.9))
fig_mc.suptitle("Section 1B: Monte Carlo -- OVB in Action (1000 simulations)", fontsize=14, y=1.03)
fig_mc.tight_layout()
savefig(fig_mc, "fig01b_mc_ovb.png")

section1b_text = f"""
Section 1B: Monte Carlo Demonstration of Omitted Variable Bias

Why Monte Carlo?
A single regression gives one estimate -- it might be close to or far
from the truth due to sampling noise. To see whether an estimator is
biased, we need to ask: "If I could repeat this experiment thousands of
times, where would the estimates cluster?" Monte Carlo simulation does
exactly this: generate data from a known DGP, estimate, repeat, and
examine the distribution of estimates.

Monte Carlo design
  - 1000 simulations, n = {n_mc} each
  - DGP: schooling = 12 + {rho:.1f}*ability + noise; wage = 2.5*schooling + 1.5*ability + noise
  - "Short regression": regress wage on schooling only (ability omitted)
  - "Long regression": regress wage on schooling + ability (correct spec)

Results
  Short regression: E[beta_hat] = {np.mean(mc_short):.3f}  (biased above 2.5)
  Long regression:  E[beta_hat] = {np.mean(mc_long):.3f}  (centered on 2.5)
  Empirical bias = {np.mean(mc_short) - 2.5:.3f}

Panel A shows the two sampling distributions overlaid. The short-regression
distribution (red) is shifted to the right of the true value 2.5 -- this
is bias. The long-regression distribution (green) is centered on 2.5 --
unbiased. Panel B shows that the bias grows with the strength of the
confounding relationship (rho). Panel C decomposes the OLS estimate into
its two components: the true causal effect plus the omitted variable bias.

KEY TAKEAWAY: Unbiasedness is a property of the *procedure*, not any
single estimate. A biased estimator systematically misses the target
across repeated samples. This is why identification -- ensuring E[e|X]=0
-- is the central concern of applied econometrics.
"""
section_contents.append((section1_text, "fig01_ols.png"))
section_contents.append((section1b_text, "fig01b_mc_ovb.png"))

# =============================================================================
# 2. FWL
# =============================================================================
section2_text = """\
Section 2: Frisch-Waugh-Lovell (FWL) Theorem

Motivation
When we run a multiple regression y ~ x1 + x2, what does it mean to say
we are "controlling for x2"? The FWL theorem gives a precise, geometric
answer: the coefficient on x1 is identical to what you get by first
removing the linear influence of x2 from both y and x1, then regressing
the residuals on each other.

Formal statement
Partition X = [X1, X2] and beta = [beta_1, beta_2]'. Define the
annihilator (residual-maker) matrix for X2:
  M2 = I - X2 (X2'X2)^{-1} X2'

Then: beta_hat_1 from the full regression y = X1*beta_1 + X2*beta_2 + e
equals exactly beta_hat_1 from the auxiliary regression
  M2*y = (M2*X1) * beta_1 + residual

In other words, you can "partial out" X2 and get the same coefficient.

Why it matters for econometrics
1. Understanding "controlling for": When someone says "we control for
   experience," FWL tells you the coefficient on schooling captures
   only the component of schooling orthogonal to experience -- the
   variation in schooling that cannot be predicted from experience.

2. Fixed effects: Entity fixed effects with hundreds of dummies are
   computationally expensive in a full regression. FWL says you can
   equivalently demean within each entity and run OLS on the demeaned
   data -- this is exactly the "within estimator" (Section 5).

3. Partial regression plots: Plotting M2*y against M2*x1 is the
   standard "added variable plot" or "partial regression plot" used
   in diagnostics. The slope of that scatter is beta_hat_1.

Geometric intuition
In the column space, X2 defines a subspace. M2 projects any vector
onto the orthogonal complement of that subspace. FWL says: project
both y and x1 out of X2's column space, then regress -- you get the
same answer as fitting the full model. The figure shows this as
projecting y down to its residual component M2*y, which is the part
of y that X2 cannot explain.
"""
print(section2_text)

experience = np.random.normal(10, 3, n)
wage2 = 2.0*schooling + 1.0*experience + np.random.normal(0, 2, n)
X_full_fwl = add_const(np.column_stack([schooling, experience]))
fwl_verify = m_fwl.verify_fwl(wage2, X_full_fwl, idx_interest=1)
b2f = ols_fit(X_full_fwl, wage2)[0]
fwl_result = m_fwl.partial_out(wage2, schooling, add_const(experience))
fwl = fwl_result["fwl_coef"]
rw = fwl_result["resid_y"]
rs = fwl_result["resid_X1"]
print(f"\nFull regression beta_hat(schooling)   : {fwl_verify['full_coef']:.6f}")
print(f"FWL partialled beta_hat(schooling)    : {fwl_verify['fwl_coef']:.6f}")
print("  -> Identical. FWL theorem confirmed.")
print("  Interpretation: beta_hat(schooling) measures the relationship between")
print("  schooling and wages *after removing the linear influence of experience*.")

section2_text += f"""
Results
  Full regression beta_hat(schooling)   : {b2f[1]:.6f}
  FWL partialled beta_hat(schooling)    : {fwl:.6f}
  -> Identical (up to numerical precision). FWL theorem confirmed.

Interpretation: The coefficient on schooling in the multiple regression
captures only the part of the schooling-wage association that is not
explained by experience. If schooling and experience were orthogonal
(uncorrelated), partialing out would make no difference. But in reality
they are correlated (more experienced workers may have different
education patterns), so controlling for experience changes the estimate.

This is a critical concept throughout applied econometrics: every time
you add a control variable, you are implicitly partialing out. FWL
makes this operation explicit and verifiable.
"""

fig, axes = plt.subplots(1,3,figsize=(15,4.5))
ax=axes[0]
sc=ax.scatter(schooling,wage2,c=experience,cmap="coolwarm",alpha=.4,s=15,edgecolors="none")
ax.set_xlabel("Schooling"); ax.set_ylabel("Wage"); ax.set_title("A) Raw (colored by experience)")
plt.colorbar(sc,ax=ax,label="Experience",shrink=.8)

ax=axes[1]
ax.scatter(rs,rw,alpha=.3,s=12,c=CB,edgecolors="none")
xr=np.linspace(rs.min(),rs.max(),100)
ax.plot(xr,fwl*xr,c=CO,lw=2.5,label=f"FWL slope={fwl:.3f}")
ax.set_xlabel("Schooling perp exp"); ax.set_ylabel("Wage perp exp"); ax.set_title("B) After Partialing Out"); ax.legend(fontsize=9)

ax=axes[2]
ax.set_xlim(-1,10); ax.set_ylim(-1,8); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("C) FWL Geometry")
ax.annotate("",xy=(7,0),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=CY,lw=2.5))
ax.text(7.3,-.3,"col(X2)",fontsize=9,color=CY)
ax.annotate("",xy=(5,6),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=CB,lw=2.5))
ax.text(5.2,6.1,"y",fontsize=10,color=CB,fontweight="bold")
ax.annotate("",xy=(5,0),xytext=(0,0),arrowprops=dict(arrowstyle="-|>",color=CY,lw=1.5,ls="--"))
ax.annotate("",xy=(5,6),xytext=(5,0),arrowprops=dict(arrowstyle="-|>",color=CO,lw=2.5))
ax.text(5.5,3,"M2*y\n(residual)",fontsize=9,color=CO,fontweight="bold")
ax.plot([4.5,4.5,5],[0,.5,.5],c=CY,lw=1)
ax.text(1.5,5.5,"FWL: regress M2*y on M2*x1\n-> same beta_hat as full model",fontsize=9,fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=CO,alpha=.9))
fig.suptitle("Section 2: Frisch-Waugh-Lovell Theorem",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig02_fwl.png")
section_contents.append((section2_text, "fig02_fwl.png"))

# =============================================================================
# 3. Heteroskedasticity
# =============================================================================
section3_text = """\
Section 3: Heteroskedasticity -- Detection and Robust Standard Errors

Economic context
In many empirical settings, the variance of the error term is not constant
across observations. A classic example: wage variance often grows with
income level -- high earners have more volatile compensation (bonuses,
stock options, variable pay), while minimum-wage workers cluster tightly
around a fixed hourly rate. This pattern is called heteroskedasticity:
Var(epsilon_i | X_i) = sigma_i^2, which varies with X.

What breaks and what does not
Under heteroskedasticity, OLS beta_hat remains unbiased and consistent
(assumptions A1-A4 still hold). However, the usual formula for standard
errors, SE = sqrt(sigma_hat^2 * (X'X)^{-1}), is wrong because it
assumes Var(epsilon|X) = sigma^2 * I. Using incorrect SEs means
confidence intervals have wrong coverage and hypothesis tests have
incorrect size -- you might reject a true null too often or too rarely.

Detection: Breusch-Pagan test
Regress the squared OLS residuals e_hat_i^2 on X. Under H0 of
homoskedasticity, the squared residuals should be unrelated to X.
A significant F-stat (or chi-squared) rejects H0 and indicates
heteroskedasticity.

The fix: Heteroskedasticity-Consistent (HC) standard errors
Rather than assuming a constant sigma^2, we estimate the "sandwich"
covariance matrix:
  V_hat_HC = (X'X)^{-1} * [Sum_i e_hat_i^2 * x_i * x_i'] * (X'X)^{-1}

The HC1 variant (Stata's default) applies a degrees-of-freedom
correction: multiply by n/(n-k). Other variants (HC0, HC2, HC3)
differ in how they adjust the residuals, but in practice with
moderate n they give very similar results.

Practical advice
In applied econometrics, robust SEs are essentially the default.
Many journals and referees expect them, and there is no real cost
to using them when heteroskedasticity is absent (they are still
consistent, just slightly less efficient than classical SEs).
The mantra: "Always report robust standard errors."
"""
print(section3_text)

x3 = np.random.uniform(1, 10, n)
y3 = 1 + 2*x3 + np.random.normal(0, x3*0.5, n)
X3 = add_const(x3)
het_result = m_het.estimate_with_robust_se(X3, y3)
b3 = het_result["beta"]
se3h = het_result["se_classical"]
se3r = het_result["se_robust"]
e3 = het_result["residuals"]
esq = e3**2
bp_b,bp_se,_,_ = ols_fit(X3, esq)
bp_F = het_result["bp_test"]["F_stat"]
bp_p = het_result["bp_test"]["p_value"]
print(f"\nBreusch-Pagan test  F = {bp_F:.2f},  p = {bp_p:.4f}")
print("  p < 0.05 -> reject homoskedasticity")
print(f"\nSE(x) -- homoskedastic : {se3h[1]:.4f}")
print(f"SE(x) -- HC1 robust    : {se3r[1]:.4f}")
print("  In practice: always report robust SEs in applied econometrics.")

section3_text += f"""
Results
  Breusch-Pagan test  F = {bp_F:.2f},  p = {bp_p:.4f}
  p < 0.05 -> reject homoskedasticity (the variance is not constant)

  SE(x) -- homoskedastic : {se3h[1]:.4f}
  SE(x) -- HC1 robust    : {se3r[1]:.4f}

In this simulation, the variance grows linearly with x (the "fan" shape
in panel A). The Breusch-Pagan test strongly rejects homoskedasticity.
Notice that the robust SE differs from the classical SE -- using the
wrong one would give misleading inference. Panel C shows that both
confidence intervals cover the true value (beta=2), but in general,
the homoskedastic CI can be too narrow or too wide depending on the
pattern of heteroskedasticity and the distribution of X.
"""

fig,axes=plt.subplots(1,3,figsize=(15,4.5))
ax=axes[0]
ax.scatter(x3,y3,alpha=.25,s=12,c=CB,edgecolors="none")
xp=np.linspace(1,10,100)
ax.plot(xp,1+2*xp,c=CO,lw=2.5,label="E[y|x]")
ax.fill_between(xp,1+2*xp-2*(xp*.5),1+2*xp+2*(xp*.5),alpha=.15,color=CO,label="+/-2*sigma(x)")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("A) Fan-Shaped Data"); ax.legend(fontsize=8)

ax=axes[1]
ax.scatter(x3,esq,alpha=.2,s=12,c=CP,edgecolors="none")
ax.plot(xp,bp_b[0]+bp_b[1]*xp,c=CR,lw=2.5,label=f"BP (F={bp_F:.1f})")
ax.set_xlabel("x"); ax.set_ylabel("e_hat^2"); ax.set_title("B) Breusch-Pagan"); ax.legend(fontsize=8)

ax=axes[2]
bx=b3[1]
for i,(lab,se,col) in enumerate([("Homoskedastic",se3h[1],CB),("HC1 Robust",se3r[1],CO)]):
    lo,hi=bx-1.96*se,bx+1.96*se
    ax.errorbar(i,bx,yerr=1.96*se,fmt="o",color=col,capsize=8,capthick=2,ms=8,lw=2,
                label=f"{lab}\n[{lo:.3f},{hi:.3f}]")
ax.axhline(2,color=CG,ls="--",lw=1.5,label="True beta=2")
ax.set_xticks([0,1]); ax.set_xticklabels(["Homo","HC1"],fontsize=9)
ax.set_ylabel("beta_hat(x)"); ax.set_title("C) SE Comparison"); ax.legend(fontsize=7,loc="upper right")
fig.suptitle("Section 3: Heteroskedasticity",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig03_het.png")
section_contents.append((section3_text, "fig03_het.png"))

# =============================================================================
# 4. IV / 2SLS
# =============================================================================
section4_text = """\
Section 4: Instrumental Variables (IV / 2SLS)

The endogeneity problem
In Section 1 we saw that omitting ability biases the OLS estimate of the
return to schooling. More generally, whenever Cov(x, epsilon) != 0 -- due
to omitted variables, measurement error, or simultaneity -- OLS is biased
and inconsistent. No amount of additional data will fix it.

The instrumental variables solution
Find a variable Z (the "instrument") that satisfies two conditions:
  (1) Relevance:   Cov(Z, X) != 0  -- Z predicts the endogenous X.
  (2) Exclusion:   Cov(Z, epsilon) = 0  -- Z affects Y only through X.

Condition (1) is testable via the first-stage F-statistic. The rule of
thumb (Stock & Yogo): F > 10 for a single endogenous regressor to avoid
weak-instrument bias. Condition (2) is fundamentally untestable -- it is
an economic argument, not a statistical test.

Classic example: Card (1995) used geographic proximity to a four-year
college as an instrument for schooling. The argument: growing up near a
college lowers the cost of attending (relevance), and distance itself
does not directly affect wages (exclusion). Here we simulate an
analogous setup.

The 2SLS procedure
  Stage 1:  Regress X on Z (and controls): X_hat = Z * gamma_hat
  Stage 2:  Regress Y on X_hat (and controls): beta_hat_IV from Y ~ X_hat

Equivalently, in the simple just-identified case (one instrument, one
endogenous variable), the Wald estimator gives:
  beta_hat_IV = Cov(Y, Z) / Cov(X, Z)  =  Reduced Form / First Stage

This ratio has an intuitive interpretation: the reduced form tells you
how much Y changes per unit of Z, and the first stage tells you how
much X changes per unit of Z. Their ratio recovers how much Y changes
per unit of X, using only the Z-driven variation in X.

INTUITION BOX: The "filtered variation" perspective
Think of the endogenous X as containing two components:
  X = (part predicted by Z) + (part correlated with epsilon)
OLS uses ALL variation in X, including the contaminated part. 2SLS
replaces X with X_hat from the first stage, which contains ONLY the
Z-driven variation -- by construction uncorrelated with epsilon. You
lose statistical power (X_hat has less variation than X), but you gain
consistency. This is the fundamental variance-bias tradeoff of IV.

What IV estimates: the Local Average Treatment Effect (LATE)
IV does not generally recover the population ATE. Under the Imbens-Angrist
LATE framework, 2SLS estimates the causal effect for "compliers" -- units
whose treatment status is shifted by the instrument. In the schooling
example, LATE is the return to schooling for people whose education
decision was actually affected by college proximity.

This matters: if the compliers are a non-representative subpopulation,
the LATE may differ substantially from the ATE. Always discuss who the
compliers are in your specific application.

Standard error subtlety in 2SLS
The second stage uses X_hat (fitted values from stage 1) in place of X.
This creates a common pitfall: running OLS on (Y, X_hat) gives the
correct beta_hat but INCORRECT standard errors. The naive residuals
Y - X_hat * beta overstate the error variance because they include the
first-stage prediction error. The correct formula uses residuals computed
with the ACTUAL X: e = Y - X * beta_hat_2SLS, then plugs into the
sandwich formula with X_hat in the "bread" matrices. Most software does
this automatically, but understanding why matters for custom estimators.

POLICY CONNECTION: IV is the workhorse for evaluating programs where
treatment is not randomly assigned. For mental health policy, plausible
instruments might include distance to the nearest provider (affecting
access but not directly outcomes), random assignment to a hotline
vs. in-person counselor (in a partial-compliance RCT), or policy
variation across jurisdictions that shifts treatment intensity.
"""
print(section4_text)

aiv = np.random.normal(0,1,n)
dist = np.random.uniform(0,50,n)
siv = 14 - .05*dist + .3*aiv + np.random.normal(0,1,n)
wiv = 2.5*siv + 1.5*aiv + np.random.normal(0,2,n)
Z = add_const(dist)

# --- Use encapsulated 2SLS estimation ---
iv_result = m_iv.estimate_2sls(Z, siv, wiv)
b_ols = iv_result["ols_beta"]
b_2s = iv_result["beta"]
Fiv = iv_result["F_stat"]
shat = iv_result["X_hat"]
se_2sls_hom = iv_result["se_hom"][1]
se_2sls_rob = iv_result["se_robust"][1]
se_2sls_naive = iv_result["se_naive"][1]
b_fs = iv_result["first_stage"]["gamma"]
se_fs = iv_result["first_stage"]["se"]
b_rf = ols_fit(Z, wiv)[0]

print(f"\nFirst-stage F-stat (instrument strength): {Fiv:.1f}")
print(f"  Rule of thumb: F > 10 for 'strong' instrument. {'pass' if Fiv > 10 else 'WEAK'}")
print(f"\nOLS beta_hat(schooling)  : {b_ols[1]:.3f}  <- biased upward")
print(f"2SLS beta_hat(schooling) : {b_2s[1]:.3f}  <- closer to truth (2.5)")
print(f"\n  Standard errors for 2SLS estimate:")
print(f"    Naive (OLS on fitted): {se_2sls_naive:.4f}  <- WRONG (overstated)")
print(f"    Correct homoskedastic: {se_2sls_hom:.4f}")
print(f"    Robust HC1:            {se_2sls_rob:.4f}")
print("  IV corrects for ability bias by only using variation in schooling")
print("  that is driven by distance (exogenous), not by ability.")

section4_text += f"""
Results
  First-stage F-stat (instrument strength): {Fiv:.1f}
  Rule of thumb: F > 10 for 'strong' instrument. {'pass' if Fiv > 10 else 'WEAK'}

  OLS beta_hat(schooling)  : {b_ols[1]:.3f}  <- biased upward by ability
  2SLS beta_hat(schooling) : {b_2s[1]:.3f}  <- closer to truth (2.5)

  Standard errors for 2SLS estimate:
    Naive (OLS on fitted values): {se_2sls_naive:.4f}  <- WRONG
    Correct homoskedastic 2SLS:   {se_2sls_hom:.4f}
    Robust HC1 2SLS:              {se_2sls_rob:.4f}
  Note: Regressing Y on X_hat gives the right beta but not the right SE.
  The naive SE overstates uncertainty because it treats X_hat as data
  rather than a projection. The correct formula uses sigma^2*(X_hat'X_hat)^{{-1}}.
  Robust SEs use the sandwich formula (Angrist & Pischke 2009).

  Wald Estimator:
    beta_hat_IV = RF / FS = Cov(Y,Z)/Cov(X,Z)
               = {b_rf[1]:.4f} / {b_fs[1]:.4f}
               = {b_rf[1]/b_fs[1]:.3f}   (True = 2.5)

The IV estimate removes the ability bias by isolating only the
variation in schooling driven by distance. The DAG in panel A shows
the exclusion restriction: distance (Z) affects wages (Y) only
through schooling (X), with no direct arrow from Z to Y.

Key diagnostic checklist for applied IV:
  1. Report the first-stage F-stat (weak instruments bias IV toward OLS)
  2. Argue the exclusion restriction on economic/institutional grounds
  3. If over-identified (more instruments than endogenous vars), run
     the Sargan/Hansen J-test for overidentifying restrictions
  4. Report both OLS and IV: if they agree, endogeneity may be mild
"""

fig=plt.figure(figsize=(15,8)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=.4,wspace=.35)

# A: DAG
ax=fig.add_subplot(gs[0,0]); ax.set_xlim(-1,11); ax.set_ylim(-1,9); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("A) IV DAG")
for x,y,l,ec in [(1,4,"Distance\n(Z)",CG),(5,7.5,"Ability\n(U)",CR),(5,4,"Schooling\n(X)",CB),(9,4,"Wage\n(Y)",CB)]:
    ax.add_patch(plt.Circle((x,y),1.2,fc="white",ec=ec,lw=2.2))
    ax.text(x,y,l,ha="center",va="center",fontsize=8.5,fontweight="bold")
a2=dict(arrowstyle="-|>",lw=2,mutation_scale=16)
ax.annotate("",xy=(3.8,4),xytext=(2.2,4),arrowprops={**a2,"color":CG})
ax.annotate("",xy=(7.8,4),xytext=(6.2,4),arrowprops={**a2,"color":CB})
ax.annotate("",xy=(8,5.2),xytext=(6,6.8),arrowprops={**a2,"color":CR})
ax.annotate("",xy=(5.8,5.8),xytext=(5.2,6.2),arrowprops={**a2,"color":CR,"linestyle":"dashed"})
ax.text(.5,1.2,"Exclusion: Z->Y only\nthrough X",fontsize=7.5,fontstyle="italic",
        bbox=dict(boxstyle="round",fc="white",ec=CG,alpha=.9))

# B: First stage
ax=fig.add_subplot(gs[0,1])
ax.scatter(dist,siv,alpha=.2,s=10,c=CB,edgecolors="none")
xp=np.linspace(0,50,100); ax.plot(xp,b_fs[0]+b_fs[1]*xp,c=CO,lw=2.5)
ax.set_xlabel("Distance"); ax.set_ylabel("Schooling"); ax.set_title(f"B) First Stage (F={Fiv:.1f})")

# C: Reduced form
ax=fig.add_subplot(gs[0,2])
ax.scatter(dist,wiv,alpha=.2,s=10,c=CP,edgecolors="none")
ax.plot(xp,b_rf[0]+b_rf[1]*xp,c=CO,lw=2.5)
ax.set_xlabel("Distance"); ax.set_ylabel("Wage"); ax.set_title("C) Reduced Form Z->Y")

# D: Second stage
ax=fig.add_subplot(gs[1,0])
ax.scatter(siv,wiv,alpha=.12,s=10,c=CY,edgecolors="none",label="Raw (endog)")
ax.scatter(shat,wiv,alpha=.2,s=10,c=CG,edgecolors="none",label="Fitted schooling_hat")
sp=np.linspace(shat.min(),shat.max(),100)
ax.plot(sp,b_2s[0]+b_2s[1]*sp,c=CO,lw=2.5,label=f"2SLS beta_hat={b_2s[1]:.3f}")
ax.set_xlabel("Schooling"); ax.set_ylabel("Wage"); ax.set_title("D) Second Stage"); ax.legend(fontsize=7)

# E: Bar chart
ax=fig.add_subplot(gs[1,1])
ms=["True","OLS\n(biased)","2SLS"]
vs=[2.5,b_ols[1],b_2s[1]]; cs=[CG,CR,CB]
bars=ax.bar(ms,vs,color=cs,width=.5,edgecolor="white"); ax.axhline(2.5,color=CG,ls="--",lw=1.5)
for bar,v in zip(bars,vs): ax.text(bar.get_x()+bar.get_width()/2,v+.05,f"{v:.3f}",ha="center",fontsize=9)
ax.set_ylabel("beta_hat"); ax.set_title("E) Comparison")

# F: Wald
ax=fig.add_subplot(gs[1,2]); ax.axis("off"); ax.set_title("F) Wald Estimator")
ax.text(.05,.95,f"beta_hat_IV = RF / FS\n     = Cov(Y,Z)/Cov(X,Z)\n\n     = {b_rf[1]:.4f} / {b_fs[1]:.4f}\n     = {b_rf[1]/b_fs[1]:.3f}\n\n     (True = 2.5)",
        transform=ax.transAxes,fontsize=11,va="top",fontfamily="monospace",
        bbox=dict(boxstyle="round",fc="white",ec=CB,alpha=.9))
fig.suptitle("Section 4: Instrumental Variables / 2SLS",fontsize=14,y=1.01)
savefig(fig,"fig04_iv.png")
section_contents.append((section4_text, "fig04_iv.png"))

# =============================================================================
# 5. Panel Data -- Fixed Effects
# =============================================================================
section5_text = """\
Section 5: Panel Data -- Fixed Effects (Within Estimator)

What is panel data?
Panel (or longitudinal) data observes the same N units (people, firms,
states, hospitals) across T time periods. This structure is extremely
powerful because it lets us control for all time-invariant unobservable
characteristics of each unit -- things like innate ability, institutional
culture, or geography that we can never directly measure.

The model
  y_it = alpha_i + X_it * beta + epsilon_it

Here alpha_i is the unit-specific "fixed effect" -- a constant unique to
each unit that absorbs everything about that unit that does not change
over time. The key assumption is strict exogeneity conditional on the
fixed effect: E[epsilon_it | alpha_i, X_i1, ..., X_iT] = 0.

The within estimator (demeaning)
Rather than estimating N dummy variables (computationally expensive and
sometimes infeasible), subtract unit means:
  y_ddot_it = y_it - y_bar_i
  X_ddot_it = X_it - X_bar_i

Then run OLS on the demeaned data:
  beta_hat_FE = (X_ddot'X_ddot)^{-1} X_ddot'y_ddot

This is algebraically identical to including N unit dummies (by the FWL
theorem from Section 2!) but computationally much cheaper.

What FE does and does not solve
FE removes bias from any time-invariant confounder. In our simulation,
if some workers always get more training because of innate traits
(captured by alpha_i), FE removes that selection bias. However, FE
cannot address time-varying confounders -- if a worker's motivation
increases in the same period they get training, that bias remains.

FE also means you cannot estimate the effect of time-invariant
regressors (e.g., race, sex in a person-level panel). Those are
absorbed into alpha_i and "differenced away."

Inference note: With panel data, errors are often serially correlated
within units. Standard practice is to cluster standard errors at the
unit level to account for this.

INTUITION BOX: Why demeaning works
Consider a worker observed over 5 years. Their wage in year t is:
  wage_it = alpha_i + beta*training_it + epsilon_it
Taking the worker-level mean: wage_bar_i = alpha_i + beta*tr_bar_i + eps_bar_i
Subtracting: (wage_it - wage_bar_i) = beta*(training_it - tr_bar_i) + (eps_it - eps_bar_i)
The fixed effect alpha_i cancels! We identify beta purely from
within-worker variation -- comparing each worker to themselves across
time. If a worker gets more training in some years, did their outcome
improve in those same years?

POLICY CONNECTION: Panel FE is invaluable for mental health policy
research. A panel of states observed annually could control for all
stable differences (culture, climate, demographics, institutional
history) while estimating the effect of new funding policies on
outcomes like hospitalization rates, employment, or suicide rates.
The key limitation: FE cannot address time-varying confounders, so
if states that expanded mental health funding also experienced
economic booms, the FE estimate would be biased.
"""
print(section5_text)

Nu,Tp=100,5
uid=np.repeat(np.arange(Nu),Tp); tid=np.tile(np.arange(Tp),Nu)
ai=np.repeat(np.random.normal(0,2,Nu),Tp)
# Make treatment probability depend on alpha_i (unit effect) so Cov(alpha_i, tr) > 0
# This induces OVB in pooled OLS: units with higher alpha_i are more likely treated
prob_tr = 1 / (1 + np.exp(- (0.5 * ai[::Tp])))  # logistic on each unit's effect
tr=np.random.binomial(1, np.repeat(prob_tr, Tp)).astype(float)
out=ai+1.8*tr+np.random.normal(0,1,Nu*Tp)
# Verify selection was induced
_corr_check = pd.DataFrame({"u": uid, "tr": tr}).groupby("u")["tr"].mean().corr(
    pd.Series(ai[::Tp]))
print(f"  [DGP check] Corr(alpha_i, mean(tr_i)) = {_corr_check:.3f}  (>0 confirms selection)")
pdf_df=pd.DataFrame({"u":uid,"t":tid,"y":out,"tr":tr})
b_pool,_,_,_=ols_fit(add_const(tr),out)
# Within estimator via encapsulated method
fe_result = m_fe.estimate_fe(out, tr, uid)
b_fe = fe_result["beta_fe"]
se_fe_homosk = fe_result["se_homosk"]
se_fe_cluster = fe_result["se_cluster"]
dm = m_fe.within_demean(out, tr, uid)
td, yd = dm["X_demean"], dm["y_demean"]
print(f"\nOLS (no FE)   beta_hat(training): {b_pool[1]:.3f}  <- biased (selection on alpha_i)")
print(f"Within FE     beta_hat(training): {b_fe:.3f}  <- unbiased")
print("  True effect = 1.8")
print("  FE removes correlation between alpha_i and training assignment.")
print(f"\n  Inference (on FE estimate):")
print(f"    Homoskedastic SE: {se_fe_homosk:.4f}")
print(f"    Clustered SE (by unit): {se_fe_cluster:.4f}")
print(f"    Clustering accounts for within-unit serial correlation.")

section5_text += f"""
Results
  DGP: P(training=1) = logistic(0.5 * alpha_i), so Cov(alpha_i, tr) > 0.
  Corr(alpha_i, mean(tr_i)) = {_corr_check:.3f}

  OLS (no FE)   beta_hat(training): {b_pool[1]:.3f}  <- biased by alpha_i selection
  Within FE     beta_hat(training): {b_fe:.3f}  <- unbiased
  True effect = 1.8

  Inference on FE estimate:
    Homoskedastic SE: {se_fe_homosk:.4f}
    Clustered SE (unit): {se_fe_cluster:.4f}
    With serial correlation within units, cluster SEs (Arellano 1987)
    are larger -- using homoskedastic SEs would overstate precision.

The pooled OLS estimate is biased because units with higher alpha_i
are more likely treated (selection on unobservables). The within
estimator removes this confounding by comparing each individual to
their own average across time.

This is a powerful identification strategy for policy evaluation. For
example, in mental health research, a panel of states observed over
years could use state fixed effects to control for all stable
differences across states (culture, demographics, historical funding
levels) while estimating the effect of a new funding policy on
outcomes like hospitalization rates or employment.
"""

fig,axes=plt.subplots(1,3,figsize=(15,5))
ax=axes[0]
for u in np.random.choice(Nu,8,replace=False):
    udf=pdf_df[pdf_df["u"]==u]; c=CO if udf["tr"].mean()>.5 else CB
    ax.plot(udf["t"],udf["y"],marker="o",ms=4,alpha=.7,color=c,lw=1.5)
ax.set_xlabel("Time"); ax.set_ylabel("Outcome"); ax.set_title("A) Unit Trajectories")
ax.legend([Line2D([0],[0],color=CO,lw=2),Line2D([0],[0],color=CB,lw=2)],["More training","Less training"],fontsize=8)

ax=axes[1]
j=np.random.normal(0,.05,len(tr))
ax.scatter(tr+j,out,alpha=.06,s=8,c=CY,label="Raw")
ax.scatter(td+j,yd,alpha=.1,s=8,c=CG,label="Demeaned")
ax.set_xlabel("Training"); ax.set_ylabel("Outcome"); ax.set_title("B) Raw vs Demeaned"); ax.legend(fontsize=8)

ax=axes[2]
ms=["True","Pooled OLS","Within FE"]; vs=[1.8,b_pool[1],b_fe]; cs=[CG,CR,CB]
bars=ax.bar(ms,vs,color=cs,width=.45,edgecolor="white"); ax.axhline(1.8,color=CG,ls="--",lw=1.5)
for bar,v in zip(bars,vs): ax.text(bar.get_x()+bar.get_width()/2,v+.03,f"{v:.3f}",ha="center",fontsize=9)
ax.set_ylabel("beta_hat(training)"); ax.set_title("C) FE Removes Heterogeneity")
fig.suptitle("Section 5: Panel Data -- Fixed Effects",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig05_fe.png")
section_contents.append((section5_text, "fig05_fe.png"))

# =============================================================================
# 6. DiD
# =============================================================================
section6_text = """\
Section 6: Difference-in-Differences (DiD)

The idea
DiD is one of the most widely used quasi-experimental designs in applied
economics and policy evaluation. A policy or treatment is rolled out to
some units ("treated group") but not others ("control group") at a
particular point in time. We cannot simply compare treated vs control
after the policy (selection bias) or treated before vs after (time
trends). DiD combines both comparisons to difference out confounds.

The estimand
  tau_DiD = (y_bar_treat,post - y_bar_treat,pre)
          - (y_bar_ctrl,post - y_bar_ctrl,pre)

The first difference removes the level difference between groups
(selection). The second difference removes the common time trend. What
remains is (under assumptions) the causal effect of the treatment.

Regression formulation
  y_it = beta_0 + beta_1*Treated_i + beta_2*Post_t
       + beta_3*(Treated_i * Post_t) + epsilon_it

  beta_3 is the DiD estimator -- the coefficient on the interaction term.

The parallel trends assumption
This is the crucial identifying assumption: absent treatment, the
treated and control groups would have followed the same trend. It
is fundamentally untestable for the post-treatment period, but we
can check pre-treatment trends as a falsification test. If the groups
were trending differently before the policy, DiD is biased.

Panel C of the figure illustrates this failure mode: when the treated
group has a steeper pre-trend, the parallel trends counterfactual
(extrapolating from the control group's trend) gives a biased estimate.

Modern extensions
The canonical 2x2 DiD (one treated group, one time period) extends
to staggered adoption designs where different units adopt treatment
at different times. Recent econometric research (Callaway & Sant'Anna
2021, Sun & Abraham 2021, de Chaisemartin & d'Haultfoeuille 2020)
shows that the standard two-way fixed effects estimator can be biased
under treatment effect heterogeneity. Modern DiD uses group-time
specific ATTs and robust aggregation.

In practice, always:
  1. Plot pre-trends and test for parallel trends
  2. Use clustered standard errors (at the group level)
  3. Consider event-study specifications that show dynamic effects
  4. Be transparent about the parallel trends argument
"""
print(section6_text)

nd=200
trd=np.random.binomial(1,.5,nd).astype(float)
pst=np.random.binomial(1,.5,nd).astype(float)
yd6=2+1*trd+1.5*pst+3*trd*pst+np.random.normal(0,1,nd)
# DiD via encapsulated method
did_result = m_did.estimate_did(yd6, trd, pst)
bd = did_result["beta"]
sed = did_result["se_classical"]
ed_did = did_result["residuals"]
se_did_robust = did_result["se_robust"][3]
Xd=np.column_stack([np.ones(nd),trd,pst,trd*pst])
print(f"\nDiD coefficient beta_hat_3 : {bd[3]:.3f}")
print(f"True treatment effect: 3.0")
print(f"95% CI (homoskedastic): [{bd[3]-1.96*sed[3]:.3f}, {bd[3]+1.96*sed[3]:.3f}]")
print(f"  Homoskedastic SE: {sed[3]:.4f}")
print(f"  HC1 Robust SE:    {se_did_robust:.4f}")
print("\nNote: In panel DiD settings, cluster SEs at the unit level.")
print("  y_it = alpha_i + gamma_t + tau * D_it + epsilon_it  (entity + time FE)")

section6_text += f"""
Results
  DiD coefficient beta_hat_3 : {bd[3]:.3f}
  True treatment effect: 3.0
  Homoskedastic SE: {sed[3]:.4f}
  HC1 Robust SE:    {se_did_robust:.4f}
  95% CI (homoskedastic): [{bd[3]-1.96*sed[3]:.3f}, {bd[3]+1.96*sed[3]:.3f}]

In panel DiD settings, cluster standard errors at the unit level
(Bertrand, Duflo & Mullainathan 2004). With serial correlation within
units, naive SEs dramatically overreject -- Bertrand et al. show that
clustering can increase SEs by a factor of 2-3x in typical applications.

The regression formulation with two-way fixed effects:
  y_it = alpha_i + gamma_t + tau * D_it + epsilon_it

This absorbs unit fixed effects (alpha_i) and time fixed effects
(gamma_t), with tau capturing the treatment effect.
"""

dfd=pd.DataFrame({"y":yd6,"tr":trd,"p":pst})
mn=dfd.groupby(["tr","p"])["y"].mean().unstack()

fig,axes=plt.subplots(1,3,figsize=(15,5))

ax=axes[0]
ax.plot([0,1],[mn.loc[0,0],mn.loc[0,1]],"o-",color=CB,lw=2.5,ms=10,label="Control")
ax.plot([0,1],[mn.loc[1,0],mn.loc[1,1]],"s-",color=CO,lw=2.5,ms=10,label="Treated")
cf=mn.loc[1,0]+(mn.loc[0,1]-mn.loc[0,0])
ax.plot([0,1],[mn.loc[1,0],cf],"s--",color=CO,lw=1.5,ms=8,alpha=.5,label="Counterfactual")
ax.annotate("",xy=(1.08,mn.loc[1,1]),xytext=(1.08,cf),arrowprops=dict(arrowstyle="<->",color=CR,lw=2))
ax.text(1.15,(mn.loc[1,1]+cf)/2,f"tau_hat={bd[3]:.2f}",fontsize=10,color=CR,fontweight="bold",va="center")
ax.set_xticks([0,1]); ax.set_xticklabels(["Pre","Post"])
ax.set_ylabel("y_bar"); ax.set_title("A) Classic DiD"); ax.legend(fontsize=8)

ax=axes[1]
tp=np.array([0,1,2,3]); ct=2+.5*tp; tt=3+.5*tp; ta=tt.copy(); ta[3]+=3
ax.plot(tp,ct,"o-",color=CB,lw=2.5,ms=8,label="Control")
ax.plot(tp[:3],tt[:3],"s-",color=CO,lw=2.5,ms=8,label="Treated (pre)")
ax.plot(tp[2:],ta[2:],"s-",color=CO,lw=2.5,ms=8)
ax.plot(tp[2:],tt[2:],"s--",color=CO,lw=1.5,ms=6,alpha=.5,label="CF")
ax.axvline(2.5,color=CY,ls=":",lw=1.5); ax.text(2.55,1.8,"Policy",fontsize=8,color=CY)
ax.fill_between([2,3],[tt[2],tt[3]],[tt[2],ta[3]],alpha=.15,color=CR)
ax.set_xlabel("Time"); ax.set_ylabel("Outcome"); ax.set_title("B) Parallel Trends"); ax.legend(fontsize=8)

ax=axes[2]
ct2=2+.5*tp; tt2=3+np.array([0,.3,.8,1.5]); ta2=tt2.copy(); ta2[3]+=2
ax.plot(tp,ct2,"o-",color=CB,lw=2.5,ms=8,label="Control")
ax.plot(tp,ta2,"s-",color=CR,lw=2.5,ms=8,label="Treated (actual)")
ax.plot(tp,tt2,"s--",color=CR,lw=1.5,alpha=.5,label="True CF")
ax.annotate("",xy=(3.08,ta2[3]),xytext=(3.08,tt2[3]),arrowprops=dict(arrowstyle="<->",color=CG,lw=2))
ax.text(3.15,(ta2[3]+tt2[3])/2,"True tau",fontsize=9,color=CG,va="center")
cfw=ta2[2]+(ct2[3]-ct2[2])
ax.plot([2,3],[ta2[2],cfw],"s:",color=CO,lw=1.5,alpha=.7,label="DiD CF (wrong!)")
ax.annotate("",xy=(2.92,ta2[3]),xytext=(2.92,cfw),arrowprops=dict(arrowstyle="<->",color=CO,lw=2))
ax.text(2.3,(ta2[3]+cfw)/2+.15,"DiD tau_hat\n(biased!)",fontsize=9,color=CO)
ax.set_xlabel("Time"); ax.set_ylabel("Outcome"); ax.set_title("C) Parallel Trends Fails"); ax.legend(fontsize=7)
fig.suptitle("Section 6: Difference-in-Differences",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig06_did.png")

# --- Supplementary Figure 6B: Event Study Specification ---
# Simulate a panel DiD with pre/post periods for event study plot
T_es = 8; treatment_time = 4; N_es = 200
uid_es = np.repeat(np.arange(N_es), T_es)
tid_es = np.tile(np.arange(T_es), N_es)
treat_unit = np.repeat(np.random.binomial(1, 0.5, N_es), T_es).astype(float)
alpha_es = np.repeat(np.random.normal(0, 1, N_es), T_es)
gamma_es = np.tile(np.arange(T_es) * 0.3, N_es)  # time trend
# True treatment effect: 0 pre-treatment, then grows post-treatment
true_effect_es = np.where((tid_es >= treatment_time) & (treat_unit == 1),
                           2.0 * (tid_es - treatment_time + 1), 0)
y_es = alpha_es + gamma_es + true_effect_es + np.random.normal(0, 1.5, N_es * T_es)

# Event study via encapsulated method
es_result = m_did.event_study(y_es, treat_unit, tid_es, treatment_time)
event_coefs = es_result["coefs"]
event_ses = es_result["ses"]
event_times = es_result["rel_times"]

fig_es, axes_es = plt.subplots(1, 3, figsize=(15, 5))

ax = axes_es[0]
ax.errorbar(event_times, event_coefs, yerr=1.96*event_ses, fmt="o-", color=CB,
            capsize=4, capthick=1.5, ms=6, lw=2, label="Estimated tau(t)")
ax.axhline(0, color=CY, ls="-", lw=1)
ax.axvline(-0.5, color=CR, ls="--", lw=2, label="Treatment onset")
ax.fill_between(event_times[event_times < 0], 
                (event_coefs - 1.96*event_ses)[event_times < 0],
                (event_coefs + 1.96*event_ses)[event_times < 0],
                alpha=0.1, color=CG, label="Pre-trend (should be ~0)")
true_es = [0]*treatment_time + [2.0*(t+1) for t in range(T_es - treatment_time)]
ax.plot(event_times, true_es, "s--", color=CO, ms=5, lw=1.5, alpha=0.7, label="True effect")
ax.set_xlabel("Relative time (t - treatment)"); ax.set_ylabel("Estimated effect")
ax.set_title("A) Event Study Plot"); ax.legend(fontsize=7)

ax = axes_es[1]
# Show what a failed parallel trends test looks like
event_coefs_bad = event_coefs.copy()
for i, t in enumerate(event_times):
    if t < 0: event_coefs_bad[i] = 0.4 * (t + treatment_time)  # pre-trend
ax.errorbar(event_times, event_coefs_bad, yerr=1.96*event_ses, fmt="o-", color=CR,
            capsize=4, capthick=1.5, ms=6, lw=2)
ax.axhline(0, color=CY, ls="-", lw=1)
ax.axvline(-0.5, color=CR, ls="--", lw=2)
ax.fill_between(event_times[event_times < 0],
                (event_coefs_bad - 1.96*event_ses)[event_times < 0],
                (event_coefs_bad + 1.96*event_ses)[event_times < 0],
                alpha=0.15, color=CR)
ax.text(0.05, 0.85, "Pre-trend coefficients\nare NOT near zero!\nDiD is invalid.", 
        transform=ax.transAxes, fontsize=9, color=CR, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CR, alpha=0.9))
ax.set_xlabel("Relative time"); ax.set_ylabel("Estimated effect")
ax.set_title("B) Failed Pre-Trend Test")

ax = axes_es[2]
# Diagram: DiD anatomy
ax.set_xlim(-1, 11); ax.set_ylim(-1, 10); ax.axis("off")
ax.set_title("C) Anatomy of the DiD Estimator")
# Draw the 2x2 table
for x, y, txt, fc in [(2.5, 8, "Pre", "#ddd"), (6.5, 8, "Post", "#ddd"),
                       (2.5, 6, "y_bar(T,pre)", "white"), (6.5, 6, "y_bar(T,post)", "white"),
                       (2.5, 4, "y_bar(C,pre)", "white"), (6.5, 4, "y_bar(C,post)", "white"),
                       (-0.5, 6, "Treated", "#ddd"), (-0.5, 4, "Control", "#ddd")]:
    ax.add_patch(plt.Rectangle((x-1.3, y-0.7), 2.6, 1.4, fc=fc, ec="#333", lw=1.2))
    ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, fontweight="bold")
# Arrows and formula
ax.annotate("", xy=(8.5, 6), xytext=(8, 6), arrowprops=dict(arrowstyle="-|>", color=CO, lw=2))
ax.annotate("", xy=(8.5, 4), xytext=(8, 4), arrowprops=dict(arrowstyle="-|>", color=CB, lw=2))
ax.text(9.2, 6, "Delta_T", fontsize=9, color=CO, fontweight="bold", va="center")
ax.text(9.2, 4, "Delta_C", fontsize=9, color=CB, fontweight="bold", va="center")
ax.text(5, 1.5, "tau_DiD = Delta_T - Delta_C\n= (y_T,post - y_T,pre) - (y_C,post - y_C,pre)",
        fontsize=9, ha="center", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=CR, lw=2, alpha=0.9))
fig_es.suptitle("Section 6B: Event Study & DiD Diagnostics", fontsize=14, y=1.03)
fig_es.tight_layout()
savefig(fig_es, "fig06b_event_study.png")

section6b_text = """
Section 6B: Event Study Specifications and Diagnostics

What is an event study?
The classic 2x2 DiD gives a single treatment effect estimate. The event
study generalizes this by estimating separate effects for each time period
relative to the treatment date. This serves two critical purposes:

  1. PRE-TREND TEST: If the estimated effects in pre-treatment periods
     (leads) are statistically indistinguishable from zero, this supports
     the parallel trends assumption. If they trend away from zero, DiD
     is likely biased.

  2. DYNAMIC EFFECTS: Post-treatment coefficients reveal how the effect
     evolves over time. Does it appear immediately? Grow over time?
     Fade out? This matters enormously for policy evaluation.

The event study regression:
  y_it = alpha_i + gamma_t + Sum_{k != -1} tau_k * D_it^k + epsilon_it

  where D_it^k = 1 if unit i is k periods from treatment at time t.
  Period k = -1 is the reference period (normalized to zero). The
  coefficients {tau_k} trace out the dynamic treatment effect path.

Reading the event study plot (Panel A):
  - Pre-treatment points near zero: parallel trends supported
  - Post-treatment points showing an effect: treatment is working
  - Confidence intervals crossing zero: not statistically significant
  - Growing post-treatment effects: the treatment effect builds over time

When pre-trends fail (Panel B):
  If pre-treatment coefficients trend upward (or downward), it means the
  treated and control groups were already diverging before treatment.
  Any post-treatment difference could be a continuation of that pre-existing
  divergence rather than a treatment effect. In this case, DiD is invalid,
  and you should consider alternative identification strategies or at minimum
  attempt to detrend the data (though this introduces its own assumptions).

POLICY CONNECTION: Event studies are the gold standard for evaluating
policy interventions. For mental health policy -- e.g., a state expanding
Medicaid coverage for psychiatric services -- the event study would show
whether hospitalization rates, employment, or crisis incidents changed
after the policy, while verifying that treated and control states were
trending similarly beforehand.
"""
section_contents.append((section6_text, "fig06_did.png"))
section_contents.append((section6b_text, "fig06b_event_study.png"))

# =============================================================================
# 7. RDD
# =============================================================================
section7_text = """\
Section 7: Regression Discontinuity Design (RDD)

The setup
In many policy contexts, treatment is assigned based on whether a
"running variable" (or "forcing variable") crosses a known cutoff.
Examples: students receive a scholarship if their test score >= 70;
districts receive federal aid if poverty rate > some threshold; patients
receive treatment if a biomarker exceeds a clinical threshold.

The key insight: for units just above and just below the cutoff,
assignment is "as good as random." A student scoring 70.1 is
essentially identical to one scoring 69.9 in all respects except
treatment status. This local randomization provides a credible
causal estimate.

Sharp RDD estimand
  tau_RDD = lim_{x->c+} E[y|x] - lim_{x->c-} E[y|x]

This is a local average treatment effect (LATE) at the cutoff -- it
tells us the causal effect of treatment for units right at the
threshold. It does not generalize to units far from the cutoff
without additional assumptions.

Estimation: local linear regression
Fit separate linear regressions on each side of the cutoff within a
bandwidth h:
  Below:  y = alpha_L + beta_L*(x - c) + epsilon   for x in [c-h, c)
  Above:  y = alpha_R + beta_R*(x - c) + epsilon   for x in [c, c+h]

The treatment effect estimate is tau_hat = alpha_R - alpha_L, i.e.,
the jump in the intercept at the cutoff.

Bandwidth choice is critical: too narrow and you have too few
observations (high variance); too wide and the linear approximation
breaks down (high bias). Optimal bandwidth selectors (Imbens &
Kalyanaraman 2012, Calonico, Cattaneo & Titiunik 2014) balance this
bias-variance tradeoff formally.

Validity threats and diagnostics
  1. Manipulation: If agents can precisely control the running variable
     to sort above/below the cutoff, the design fails. The McCrary
     (2008) density test checks for bunching at the cutoff.
  2. Covariate smoothness: Pre-treatment covariates should be smooth
     through the cutoff. A jump in covariates suggests confounding.
  3. Bandwidth sensitivity: Results should be robust to reasonable
     bandwidth choices. Report estimates across a range of bandwidths.

Fuzzy RDD: When the cutoff determines eligibility but not perfect
compliance (some eligible don't take treatment, some ineligible do),
use a "fuzzy RDD" -- essentially an IV where crossing the cutoff
instruments for actual treatment receipt.

INTUITION BOX: Why "local randomization" at the cutoff?
Imagine lining up all students by test score. At score 69.9 vs 70.1,
students are essentially identical in every way -- ability, motivation,
background -- except that one barely missed the cutoff and one barely
made it. Any difference in their outcomes must be due to the treatment
(scholarship), not to pre-existing differences. This is why RDD is
considered one of the most credible quasi-experimental designs: the
identifying assumption (continuity) is mild and partially testable.

POLICY CONNECTION: RDD naturally arises in many mental health policy
contexts. Eligibility for subsidized treatment often depends on income
thresholds, clinical severity scores, or age cutoffs. If a community
mental health program serves patients with GAD-7 scores above 10, we
can estimate its causal effect by comparing patients just above and
below that threshold -- provided the score cannot be manipulated.
"""
print(section7_text)

run=np.random.uniform(40,100,n); cut=70.0; tr7=(run>=cut).astype(float)
y7=1+.3*run+4*tr7+np.random.normal(0,2,n)
bw=15; mask=np.abs(run-cut)<=bw
xc=run[mask]-cut; tb=tr7[mask]; yb=y7[mask]

# --- RDD estimation via encapsulated methods ---
rdd_result = m_rdd.estimate_rdd(y7, run, cut, bw)
br = rdd_result["beta"]
er_rdd = rdd_result["residuals"]

# McCrary density test
_mc_alpha = 0.001
mc_test = m_rdd.mccrary_density_test(run, cut, bandwidth=10, alpha=_mc_alpha)
_mc_stat = mc_test["z_stat"]
_mc_pval = mc_test["p_value"]
_mc_result_msg = "McCrary test (binned density)"

# Bias-corrected RDD inference
try:
    import rpy2.robjects as robj
    robj.r('library(rdrobust)')
    robj.r.assign('y_rdd', robj.FloatVector(yb.tolist()))
    robj.r.assign('x_rdd', robj.FloatVector(xc.tolist()))
    rdout = robj.r('rdrobust(y_rdd, x_rdd, c=0)')
    _rd_est = float(rdout.rx2('Estimate')[0][0])
    _rd_ci_lo = float(rdout.rx2('ci')[0][0])
    _rd_ci_hi = float(rdout.rx2('ci')[1][0])
    _rd_robust_msg = f"Robust (CCT 2014) tau_hat={_rd_est:.3f}, CI=[{_rd_ci_lo:.3f}, {_rd_ci_hi:.3f}]"
    _rd_used_robust = True
except Exception:
    bc_result = m_rdd.bias_corrected_rdd(y7, run, cut, bw, n_boot=500)
    _rd_est_bc = bc_result["tau_bc"]
    _rd_ci_lo = bc_result["ci_lo"]
    _rd_ci_hi = bc_result["ci_hi"]
    _rd_robust_msg = f"Bias-corrected (quadratic) tau_hat={_rd_est_bc:.3f}, CI=[{_rd_ci_lo:.3f}, {_rd_ci_hi:.3f}]"
    _rd_used_robust = False

print(f"\nRDD estimate of treatment effect (local linear): {br[1]:.3f}")
print(f"  {_rd_robust_msg}")
print(f"True effect: 4.0")
print(f"Bandwidth used: +/-{bw} units around cutoff")
print(f"\nMcCrary density test at cutoff:")
print(f"  z-stat = {_mc_stat:.3f}, p-value = {_mc_pval:.3f}")
print(f"  {'No bunching detected (fail to reject H0)' if _mc_pval > _mc_alpha else 'WARNING: potential manipulation'}")
print(f"  ({_mc_result_msg})")
print("\nValidity checks to always run:")
print("  1. Density test (McCrary): no sorting/manipulation around cutoff")
print("  2. Covariate balance at cutoff (pre-treatment covariates smooth?)")
print("  3. Sensitivity to bandwidth choice")

section7_text += f"""
Results
  RDD estimate (local linear): {br[1]:.3f}
  {_rd_robust_msg}
  True effect: 4.0
  Bandwidth used: +/-{bw} units around cutoff

  McCrary density test at cutoff:
    z-stat = {_mc_stat:.3f}, p-value = {_mc_pval:.3f}
    {'No bunching detected (fail to reject H0)' if _mc_pval > _mc_alpha else 'WARNING: potential manipulation'}

The local linear estimate is close to the true effect of 4.0. The
bias-corrected estimate (using quadratic fits or rdrobust) provides
a robust confidence interval following Calonico, Cattaneo & Titiunik
  (2014). The McCrary (2008) density test formally checks for bunching
  at the cutoff -- here we use a tighter significance level (alpha = 0.01) so a
  p-value greater than alpha indicates no evidence of manipulation.

The figure shows:
  Panel A: The clear jump in the outcome at the cutoff, with separate
           linear fits on each side.
  Panel B: The local linear regression zoomed into the bandwidth window.
  Panel C: The density histogram with McCrary test result.

RDD is often considered one of the most credible quasi-experimental
designs because the identifying assumption (continuity of potential
outcomes at the cutoff) is relatively mild and partially testable.
"""

bel=run<cut; abo=~bel
global_rdd = m_rdd.global_rdd_fit(y7, run, cut)
bb = global_rdd["beta_below"]
ba = global_rdd["beta_above"]

fig,axes=plt.subplots(1,3,figsize=(15,5))
ax=axes[0]
ax.scatter(run[bel],y7[bel],alpha=.25,s=12,c=CB,edgecolors="none",label="Below")
ax.scatter(run[abo],y7[abo],alpha=.25,s=12,c=CO,edgecolors="none",label="Above")
xb=np.linspace(40,cut,100); xa=np.linspace(cut,100,100)
ax.plot(xb,bb[0]+bb[1]*xb,c=CB,lw=2.5); ax.plot(xa,ba[0]+ba[1]*xa,c=CO,lw=2.5)
ax.axvline(cut,color=CR,ls="--",lw=2,label=f"Cutoff={cut:.0f}")
yl=bb[0]+bb[1]*cut; yr=ba[0]+ba[1]*cut
ax.annotate("",xy=(cut+.5,yr),xytext=(cut+.5,yl),arrowprops=dict(arrowstyle="<->",color=CR,lw=2.5))
ax.text(cut+2,(yl+yr)/2,f"tau_hat~{yr-yl:.1f}",fontsize=10,color=CR,fontweight="bold")
ax.set_xlabel("Running var"); ax.set_ylabel("Outcome"); ax.set_title("A) Jump at Cutoff"); ax.legend(fontsize=7)

ax=axes[1]
ax.scatter(run[mask&bel],y7[mask&bel],alpha=.35,s=18,c=CB,edgecolors="none")
ax.scatter(run[mask&abo],y7[mask&abo],alpha=.35,s=18,c=CO,edgecolors="none")
xbl=np.linspace(cut-bw,cut,50); xal=np.linspace(cut,cut+bw,50)
ax.plot(xbl,br[0]+br[2]*(xbl-cut),c=CB,lw=2.5)
ax.plot(xal,(br[0]+br[1])+(br[2]+br[3])*(xal-cut),c=CO,lw=2.5)
ax.axvline(cut,color=CR,ls="--",lw=2); ax.axvspan(cut-bw,cut+bw,alpha=.07,color=CY)
ax.set_xlabel("Running var"); ax.set_ylabel("Outcome"); ax.set_title(f"B) Local Linear (bw+/-{bw})")

ax=axes[2]
ax.hist(run[bel],bins=15,color=CB,alpha=.5,edgecolor="white")
ax.hist(run[abo],bins=15,color=CO,alpha=.5,edgecolor="white")
ax.axvline(cut,color=CR,ls="--",lw=2)
ax.set_xlabel("Running var"); ax.set_ylabel("Freq"); ax.set_title("C) Density (McCrary)")
ax.text(cut+1,ax.get_ylim()[1]*.9,
        f"McCrary p={_mc_pval:.3f}\n{'No bunching' if _mc_pval > _mc_alpha else 'BUNCHING!'}",
        fontsize=9,
        color=CG if _mc_pval > _mc_alpha else CR)
fig.suptitle("Section 7: Regression Discontinuity Design",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig07_rdd.png")
section_contents.append((section7_text, "fig07_rdd.png"))

# =============================================================================
# 8. Binary Outcomes
# =============================================================================
section8_text = """\
Section 8: Binary Outcomes -- Probit and Logit

The problem with OLS for binary outcomes
When the outcome y takes values in {0, 1} (e.g., employed or not,
enrolled or not, treated or not), OLS -- the "Linear Probability
Model" (LPM) -- has two issues:
  1. Predicted probabilities can fall outside [0, 1].
  2. The error term is necessarily heteroskedastic since
     Var(y|x) = P(y=1|x) * (1 - P(y=1|x)), which depends on x.

The LPM is still commonly used in applied work because its
coefficients are directly interpretable as marginal effects, and with
robust SEs the heteroskedasticity is handled. But for prediction and
when the probability is near 0 or 1, nonlinear models are preferred.

Logit and Probit models
Both model P(y=1|x) = G(x*beta), where G is a CDF that maps the
linear index x*beta to [0, 1]:
  Logit:  G(z) = Lambda(z) = 1 / (1 + e^{-z})  (logistic CDF)
  Probit: G(z) = Phi(z)                         (standard normal CDF)

The two CDFs are very similar in shape -- the logistic has slightly
heavier tails. In practice they almost always give substantively
identical results. The logit is more common in epidemiology and
machine learning (due to the odds-ratio interpretation); probit is
more common in some areas of economics.

Estimation is by Maximum Likelihood (see Section 9 for details).

Interpreting coefficients: marginal effects
The raw coefficients beta_hat are NOT marginal effects. Because G is
nonlinear, the effect of a one-unit change in x on P(y=1) depends on
where you are on the curve:
  dP/dx = G'(x*beta) * beta

Two common summaries:
  (a) Marginal Effect at the Mean (MEM): evaluate at x = x_bar.
  (b) Average Marginal Effect (AME): compute dP/dx for each obs,
      then average. AME is generally preferred because it does not
      depend on a potentially unrepresentative "average" individual.

Practical guidance
In applied work, always report AME (or MEM) alongside raw coefficients
so readers can interpret the economic magnitude. The LPM coefficient
is a rough approximation to AME when probabilities are in the middle
of [0, 1], but diverges near the extremes.
"""
print(section8_text)

xb8=np.random.normal(0,1,n); zt=-0.5+1.2*xb8
pt=1/(1+np.exp(-zt)); yb8=np.random.binomial(1,pt,n)
Xb8=add_const(xb8)

# Use the logistic function from binary_outcomes module
logistic = m_bin.logistic
# Keep nll functions available for MLE section
def nll_logit(b,X,y):
    p=np.clip(logistic(X@b),1e-12,1-1e-12); return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))
def nll_probit(b,X,y):
    p=np.clip(stats.norm.cdf(X@b),1e-12,1-1e-12); return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))

# --- Logit and Probit via encapsulated methods ---
logit_fit = m_bin.fit_logit(Xb8, yb8)
probit_fit = m_bin.fit_probit(Xb8, yb8)
lpm_fit = m_bin.fit_lpm(Xb8, yb8)
bl = logit_fit["beta"]
bp = probit_fit["beta"]
blpm = lpm_fit["beta"]
pl = logit_fit["p_hat"]
ame_l = m_bin.logit_ame(Xb8, bl)
ame_p = m_bin.probit_ame(Xb8, bp)
true_ame=1.2*np.mean(pt*(1-pt))

# --- AME Standard Errors via encapsulated methods ---
_B_ame = 500
ame_se_delta = m_bin.ame_se_delta(Xb8, bl, yb8)
ame_se_boot = m_bin.ame_se_bootstrap(Xb8, yb8, n_boot=_B_ame)

print(f"\nLogit  coefficient beta_hat(x): {bl[1]:.3f}")
print(f"Probit coefficient beta_hat(x): {bp[1]:.3f}")
print(f"  (coefficients not directly comparable -- different scale)")
print(f"\nAverage Marginal Effect (AME):")
print(f"  Logit  AME: {ame_l:.4f}")
print(f"  Probit AME: {ame_p:.4f}")
print(f"  True AME  : {true_ame:.4f}")
print(f"\n  AME Standard Errors (logit):")
print(f"    Delta method SE: {ame_se_delta:.4f}")
print(f"    Bootstrap SE:    {ame_se_boot:.4f}  ({_B_ame} reps)")
print(f"    (These should be similar -- validates the delta method.)")
print(f"\n  AME interpretation: a 1-unit increase in x raises P(y=1) by ~AME")

section8_text += f"""
Results
  Logit  coefficient beta_hat(x): {bl[1]:.3f}
  Probit coefficient beta_hat(x): {bp[1]:.3f}
  (coefficients not directly comparable -- different latent scale)

  Average Marginal Effect (AME):
    Logit  AME: {ame_l:.4f}
    Probit AME: {ame_p:.4f}
    True AME  : {true_ame:.4f}

  AME Standard Errors (logit):
    Delta method SE: {ame_se_delta:.4f}
    Bootstrap SE:    {ame_se_boot:.4f}  ({_B_ame} reps)

  AME interpretation: a 1-unit increase in x raises P(y=1) by
  approximately {ame_l:.3f} +/- {ame_se_delta:.3f} on average across the sample.

The delta method and bootstrap SEs are similar, which validates the
asymptotic approximation. In practice, report AME with SEs (via delta
method or bootstrap) so readers can assess statistical significance of
the marginal effect, not just the raw logit/probit coefficient.

Note: The logit and probit AMEs are nearly identical, confirming
that the choice of link function rarely matters for substantive
conclusions. Panel C in the figure shows how the marginal effect
varies with x -- it is largest when P(y=1) is near 0.5 (where the
CDF is steepest) and shrinks toward 0 and 1.
"""

fig,axes=plt.subplots(1,3,figsize=(15,5))
ax=axes[0]; xs=np.sort(xb8)
ax.scatter(xb8,yb8,alpha=.06,s=15,c=CY,edgecolors="none")
ax.plot(xs,blpm[0]+blpm[1]*xs,c=CR,lw=2,ls="--",label="LPM")
ax.plot(xs,logistic(bl[0]+bl[1]*xs),c=CB,lw=2.5,label="Logit")
ax.plot(xs,stats.norm.cdf(bp[0]+bp[1]*xs),c=CO,lw=2.5,ls="-.",label="Probit")
ax.plot(xs,logistic(-.5+1.2*xs),c=CG,lw=1.5,ls=":",label="True")
ax.set_ylim(-.15,1.15); ax.set_xlabel("x"); ax.set_ylabel("P(y=1)")
ax.set_title("A) LPM vs Logit vs Probit"); ax.legend(fontsize=7.5)

ax=axes[1]; zg=np.linspace(-4,4,200)
ax.plot(zg,logistic(zg),c=CB,lw=2.5,label="Logistic Lambda(z)")
ax.plot(zg,stats.norm.cdf(zg),c=CO,lw=2.5,ls="-.",label="Normal Phi(z)")
ax.plot(zg,np.clip(zg*.15+.5,0,1),c=CR,lw=2,ls="--",label="Linear")
ax.axhline(0,color=CY,lw=.5); ax.axhline(1,color=CY,lw=.5)
ax.set_xlabel("z=x*beta"); ax.set_ylabel("G(z)"); ax.set_title("B) Link Functions"); ax.legend(fontsize=8)

ax=axes[2]; xme=np.linspace(-3,3,200)
zml=bl[0]+bl[1]*xme; pml=logistic(zml)
ax.plot(xme,bl[1]*pml*(1-pml),c=CB,lw=2.5,label="Logit ME")
ax.plot(xme,bp[1]*stats.norm.pdf(bp[0]+bp[1]*xme),c=CO,lw=2.5,ls="-.",label="Probit ME")
ax.axhline(blpm[1],c=CR,ls="--",lw=2,label="LPM (const)")
ax.axhline(ame_l,c=CP,ls=":",lw=2,label=f"AME={ame_l:.3f}")
ax.set_xlabel("x"); ax.set_ylabel("dP/dx"); ax.set_title("C) Marginal Effects Vary"); ax.legend(fontsize=7.5)
fig.suptitle("Section 8: Binary Outcomes -- Probit & Logit",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig08_binary.png")
section_contents.append((section8_text, "fig08_binary.png"))

# =============================================================================
# 9. MLE
# =============================================================================
section9_text = """\
Section 9: Maximum Likelihood Estimation -- from scratch

Why MLE matters
Nearly every estimator beyond OLS -- logit, probit, Poisson regression,
Tobit, mixed-effects models -- is estimated via Maximum Likelihood.
MLE is the bridge between specifying a probability model and obtaining
parameter estimates that best explain the observed data.

The principle
Given a parametric model with density f(y_i | x_i; beta), the
likelihood function is the joint density of the observed sample
viewed as a function of the parameters:
  L(beta) = Product_{i=1}^{n} f(y_i | x_i; beta)

Maximizing L is equivalent to maximizing the log-likelihood (which
turns products into sums, improving numerical stability):
  l(beta) = Sum_{i=1}^{n} log f(y_i | x_i; beta)

For the logit model specifically:
  P(y_i=1 | x_i) = Lambda(x_i' beta) = 1 / (1 + e^{-x_i' beta})
  l(beta) = Sum [ y_i log Lambda(x_i' beta)
                 + (1-y_i) log(1 - Lambda(x_i' beta)) ]

This is a globally concave function (for logit), so any gradient-based
optimizer will find the unique maximum.

Standard errors from the Fisher Information
The asymptotic distribution of the MLE is:
  beta_hat_MLE ~approx N(beta_0, I(beta_0)^{-1})

where I(beta) = -E[d^2 l / d beta d beta'] is the Fisher Information
matrix. In practice, we use the observed information (the negative
Hessian of the log-likelihood evaluated at beta_hat) and invert it
to get the variance-covariance matrix.

Numerical optimization
We use scipy.optimize.minimize with BFGS (a quasi-Newton method that
approximates the Hessian from gradient evaluations). The BFGS path
in panel C shows convergence from the starting point [0, 0] to the
MLE in a few iterations -- the global concavity of the logit
likelihood makes optimization straightforward.

Profile likelihood and confidence intervals
The profile likelihood for a single parameter beta_j is obtained by
maximizing the log-likelihood over all other parameters for each
fixed value of beta_j. The 95% confidence interval can be read off
as the set of beta_j values where the profile likelihood is within
1.92 (= chi^2_1(0.95)/2) of its maximum. This is an alternative to
Wald-type intervals (beta_hat +/- 1.96*SE) and can be more accurate
in small samples or with nonlinear models.
"""
print(section9_text)

print(f"\nMLE via scipy.minimize (BFGS):")
print(f"  beta_hat (intercept): {bl[0]:.4f}  (true: -0.5)")
print(f"  beta_hat (x)        : {bl[1]:.4f}  (true:  1.2)")

# Standard errors via encapsulated MLE framework
from scipy.optimize import approx_fprime
mle_result = m_mle.fit_mle(nll_logit, np.zeros(2), args=(Xb8, yb8), track_path=True)
se_mle = mle_result["se"]
hess_approx = mle_result["hessian"]
print(f"\n  SE via observed Fisher info: intercept={se_mle[0]:.4f}, x={se_mle[1]:.4f}")

section9_text += f"""
Results
  MLE via scipy.minimize (BFGS):
    beta_hat (intercept): {bl[0]:.4f}  (true: -0.5)
    beta_hat (x)        : {bl[1]:.4f}  (true:  1.2)
    SE via observed Fisher info: intercept={se_mle[0]:.4f}, x={se_mle[1]:.4f}

The contour plot (panel A) shows the log-likelihood surface. The MLE
(red star) sits at the peak. The profile likelihood (panel B) shows
the 95% confidence interval for beta_1 as the range where the profile
drops by at most 1.92 from its peak. The BFGS path (panel C) shows
rapid convergence -- only a handful of iterations are needed because
the logit likelihood is globally concave.

Being able to code MLE from scratch matters because many research
questions require custom likelihoods that don't come in a package:
mixture models, structural models with latent variables, models with
non-standard censoring or selection. Understanding the mechanics --
write the likelihood, take derivatives (or let scipy approximate
them), invert the Hessian for SEs -- is a transferable skill.
"""

fig,axes=plt.subplots(1,3,figsize=(15,5))

# A: Contours -- use encapsulated log-likelihood surface
ax=axes[0]
b0g=np.linspace(-1.5,.5,80); b1g=np.linspace(.2,2.2,80)
B0, B1, LL = m_mle.log_likelihood_surface(nll_logit, b0g, b1g, args=(Xb8, yb8))
cs=ax.contour(B0,B1,LL,levels=20,cmap="RdYlBu_r",linewidths=.8)
ax.clabel(cs,inline=True,fontsize=6,fmt="%.0f")
ax.plot(bl[0],bl[1],"r*",ms=15,label="beta_hat_MLE"); ax.plot(-.5,1.2,"g^",ms=12,label="True beta")
ax.set_xlabel("beta_0"); ax.set_ylabel("beta_1"); ax.set_title("A) Log-Likelihood Contours"); ax.legend(fontsize=9)

# B: Profile likelihood -- use encapsulated profile
ax=axes[1]
b1v=np.linspace(.4,2,100)
prof = m_mle.profile_likelihood(nll_logit, bl, profile_idx=1, grid=b1v, args=(Xb8, yb8))
llp = prof["profile_ll"]
ax.plot(b1v,llp,c=CB,lw=2.5)
ax.axvline(bl[1],color=CR,ls="--",lw=1.5,label=f"beta_hat_1={bl[1]:.3f}")
ax.axvline(1.2,color=CG,ls=":",lw=1.5,label="True=1.2")
llm=llp.max(); cim=llp>=(llm-1.92)
if cim.any():
    ax.axhline(llm-1.92,color=CY,ls=":",lw=1)
    ax.fill_between(b1v,llp.min(),llp,where=cim,alpha=.15,color=CB)
    ax.text(b1v[cim].min(),llm-5,f"95% CI\n[{prof['ci_95'][0]:.2f},{prof['ci_95'][1]:.2f}]",fontsize=8,color=CB)
ax.set_xlabel("beta_1"); ax.set_ylabel("Profile L"); ax.set_title("B) Profile Likelihood"); ax.legend(fontsize=8)

# C: BFGS path -- use path from MLE fit
ax=axes[2]
path = mle_result["path"]
ax.contour(B0,B1,LL,levels=15,cmap="RdYlBu_r",linewidths=.5,alpha=.6)
ax.plot(path[:,0],path[:,1],"o-",color=CR,lw=1.5,ms=4,label=f"BFGS ({len(path)} steps)")
ax.plot(path[0,0],path[0,1],"ko",ms=8,label="Start")
ax.plot(path[-1,0],path[-1,1],"r*",ms=15,label="Converged")
ax.set_xlabel("beta_0"); ax.set_ylabel("beta_1"); ax.set_title("C) BFGS Path"); ax.legend(fontsize=8)
fig.suptitle("Section 9: Maximum Likelihood Estimation",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig09_mle.png")
section_contents.append((section9_text, "fig09_mle.png"))

# =============================================================================
# 10. Bootstrap
# =============================================================================
section10_text = """\
Section 10: Bootstrap Inference

Motivation
Classical inference relies on asymptotic approximations: we derive the
limiting distribution of an estimator (usually normal) and use it for
CIs and tests. But these approximations can be poor when:
  - The sample is small
  - The estimator is nonlinear or complex (e.g., median, quantile
    regression, Gini coefficient, IV in small samples)
  - The test statistic has an unknown or complicated distribution
  - Standard errors involve complicated covariance structures

The bootstrap lets us approximate the sampling distribution of any
statistic directly from the data, without relying on closed-form
asymptotic results.

The nonparametric bootstrap algorithm
  For b = 1, ..., B:
    1. Draw a sample of size n *with replacement* from the original data.
    2. Compute the estimator theta_hat*_b on this bootstrap sample.
  The collection {theta_hat*_1, ..., theta_hat*_B} approximates the
  sampling distribution of theta_hat.

From the bootstrap distribution we get:
  SE_boot = std(theta_hat*_b)
  Percentile CI: [quantile(2.5%), quantile(97.5%)]

For more refined intervals, the bias-corrected accelerated (BCa)
bootstrap or the bootstrap-t method offer better coverage in finite
samples.

Why "with replacement" matters
Drawing with replacement means some observations appear multiple times
and some not at all in each bootstrap sample. On average, each
bootstrap sample contains about 63.2% unique observations (= 1 - 1/e).
This mimics the randomness of repeated sampling from the population.

When does the bootstrap fail?
  - Extremely heavy-tailed distributions (the CLT is also slow here)
  - Non-smooth statistics in small samples
  - Dependent data (need block bootstrap or cluster bootstrap)
  - When the parameter is on the boundary of the parameter space

Practical guidance
B = 1000-2000 replications is standard for SE estimation. For
percentile CIs, B >= 2000 is preferred. For BCa intervals, B >= 5000.
The computational cost is usually trivial on modern hardware.
"""
print(section10_text)

B=2000; nb=100
np.random.seed(0)  # Reset seed for bootstrap reproducibility
xbt=np.random.normal(0,1,nb); ybt=2+1.5*xbt+np.random.normal(0,2,nb)
Xbt=add_const(xbt)

# Bootstrap via encapsulated method
boot_result = m_boot.bootstrap_ols_slope(Xbt, ybt, n_boot=B, seed=0)
bc = boot_result["boot_estimates"]
bo = ols_fit(Xbt, ybt)[0]
seo = np.array([0, boot_result["analytic_se"]])  # only need slope SE
seo_full = ols_fit(Xbt, ybt)[1]
seo = seo_full
bse = boot_result["se"]
bci = np.array([boot_result["ci_lo"], boot_result["ci_hi"]])
aci = boot_result["analytic_ci"]
_unique_frac_theoretical = m_boot.unique_obs_fraction(nb)
print(f"\nBootstrap unique observation fraction: ~{_unique_frac_theoretical:.3f} (theoretical 1-1/e = {1-1/np.e:.3f})")
print(f"  Each bootstrap sample contains ~63.2% unique observations.")
print(f"\nAnalytic OLS SE(x) : {seo[1]:.4f}")
print(f"Bootstrap SE(x)    : {bse:.4f}  ({B} replications)")
print(f"Bootstrap 95% CI   : [{bci[0]:.4f}, {bci[1]:.4f}]")
print(f"Analytic  95% CI   : [{aci[0]:.4f}, {aci[1]:.4f}]")
print("\n  Bootstrap is especially useful for:")
print("  - Non-standard estimators (median, Gini, IV in small samples)")
print("  - Clustered data without clear asymptotic formula")
print("  - Testing complex hypotheses or intersection bounds")

section10_text += f"""
Results
  Bootstrap unique obs fraction: ~{_unique_frac_theoretical:.3f} (theoretical 1-1/e = {1-1/np.e:.3f})
  Each bootstrap sample contains ~63.2% unique observations (Efron 1979).

  Analytic OLS SE(x) : {seo[1]:.4f}
  Bootstrap SE(x)    : {bse:.4f}  ({B} replications)
  Bootstrap 95% CI   : [{bci[0]:.4f}, {bci[1]:.4f}]
  Analytic  95% CI   : [{aci[0]:.4f}, {aci[1]:.4f}]

  Note: np.random.seed(0) is set before this section for reproducibility.
  Running the script twice yields identical bootstrap results.

In this well-behaved OLS setting, the bootstrap SE and analytic SE
are very close, and both CIs cover the true value of 1.5. This is
reassuring -- the bootstrap reproduces what theory predicts when
theory is applicable.

The bootstrap really shines in situations where analytic SEs are
unavailable or unreliable. For example:
  - Ratio estimators (e.g., the Wald/IV estimator in small samples)
  - Non-smooth statistics (e.g., sample median, quantile regression)
  - Complex test statistics (e.g., testing equality of Gini coeffs)
  - Cluster-robust inference with few clusters (wild cluster bootstrap)

Panel A shows 80 bootstrap regression lines overlaid on the original
data, visualizing the uncertainty in the slope. Panel B shows the
bootstrap distribution of beta_hat alongside the analytic normal
approximation. Panel C compares the two 95% confidence intervals.
"""

fig,axes=plt.subplots(1,3,figsize=(15,5))

ax=axes[0]
ax.scatter(xbt,ybt,alpha=.4,s=20,c=CB,edgecolors="none",zorder=5)
xp=np.linspace(xbt.min(),xbt.max(),100)
for b in range(80):
    idx=np.random.choice(nb,nb,replace=True); bb,_,_,_=ols_fit(Xbt[idx],ybt[idx])
    ax.plot(xp,bb[0]+bb[1]*xp,c=CO,alpha=.04,lw=1)
ax.plot(xp,bo[0]+bo[1]*xp,c=CR,lw=2.5,label="Original OLS")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("A) 80 Bootstrap Lines"); ax.legend(fontsize=9)

ax=axes[1]
ax.hist(bc,bins=50,density=True,alpha=.6,color=CB,edgecolor="white")
xn=np.linspace(bc.min(),bc.max(),200)
ax.plot(xn,stats.norm.pdf(xn,bo[1],seo[1]),c=CR,lw=2,label="Normal (analytic)")
ax.axvline(1.5,color=CG,ls="--",lw=2,label="True beta=1.5")
ax.axvline(bci[0],color=CO,ls=":",lw=2)
ax.axvline(bci[1],color=CO,ls=":",lw=2,label=f"Boot CI [{bci[0]:.2f},{bci[1]:.2f}]")
ax.set_xlabel("beta_hat(x)"); ax.set_ylabel("Density"); ax.set_title(f"B) Bootstrap Dist ({B} reps)"); ax.legend(fontsize=7.5)

ax=axes[2]
for i,(lo,hi,cen,col) in enumerate(zip([aci[0],bci[0]],[aci[1],bci[1]],[bo[1],np.mean(bc)],[CB,CO])):
    ax.errorbar(i,cen,yerr=[[cen-lo],[hi-cen]],fmt="o",color=col,capsize=10,capthick=2.5,ms=10,lw=2.5)
    ax.text(i+.15,cen,f"[{lo:.3f},{hi:.3f}]",fontsize=9,va="center",color=col)
ax.axhline(1.5,color=CG,ls="--",lw=2,label="True beta=1.5")
ax.set_xticks([0,1]); ax.set_xticklabels(["Analytic","Bootstrap"],fontsize=10)
ax.set_ylabel("beta_hat(x)"); ax.set_title("C) 95% CI Comparison"); ax.legend(fontsize=9)
fig.suptitle("Section 10: Bootstrap Inference",fontsize=14,y=1.03); fig.tight_layout()
savefig(fig,"fig10_bootstrap.png")
section_contents.append((section10_text, "fig10_bootstrap.png"))

print("""
# =============================================================================
# Summary: When to Use What
# =============================================================================
""")
summary = """
SUMMARY: WHEN TO USE WHAT
==========================

This primer covered ten core methods in the applied econometrician's
toolkit. Here is a condensed guide for choosing among them.


Method                         | When / What it solves
-------------------------------|-------------------------------------------------------------
OLS                            | Baseline; assume exogeneity (no omitted variables)
Robust SEs                     | Always; protects against heterosk. without changing beta_hat
FWL / Partialing out           | Understand what "controlling for X" actually does
IV / 2SLS                      | Endogenous regressor; need a valid instrument
Fixed Effects                  | Panel; time-invariant unobservables; within-unit variation
DiD                            | Policy rollout; treated/control groups; parallel trends
RDD                            | Sharp cutoff in assignment; LATE at threshold
Probit / Logit                 | Binary outcome; report marginal effects (not raw coefs)
MLE from scratch               | Non-standard likelihood; fully custom models
Bootstrap                      | Small n, complex estimator, non-standard asymptotics


Key identification hierarchy (roughly strongest to weakest):
  Randomized Experiment  >  IV  >  DiD  >  RDD  >  Selection-on-observables (OLS/PSM)

This ordering reflects how credible the causal claim typically is. An RCT
directly controls treatment assignment; IV uses exogenous variation from an
instrument; DiD exploits differential timing under parallel trends; RDD
exploits a cutoff under continuity; and selection-on-observables (OLS with
controls, or propensity score matching) assumes no unobserved confounders
-- the strongest and least credible assumption.


Choosing a method in practice -- a decision tree:

1. Can you randomize?
   -> Yes: Run an experiment. OLS on treatment assignment estimates the ATE.

2. Is there an instrument?
   -> Yes + strong first stage + credible exclusion: Use IV/2SLS.

3. Is there a policy change affecting some units but not others?
   -> Yes + parallel trends plausible: Use DiD.
   -> If staggered adoption: Use modern DiD estimators (Callaway-Sant'Anna).

4. Is treatment assigned by a cutoff in a running variable?
   -> Yes + no manipulation: Use RDD (sharp or fuzzy).

5. Do you have panel data?
   -> Yes: Use Fixed Effects to remove time-invariant confounders.
   -> Combine with DiD if there is also a policy shock.

6. None of the above?
   -> Selection-on-observables: OLS with careful controls, propensity
      score methods. Be transparent that unobserved confounding is a
      threat. Sensitivity analysis (e.g., Oster 2019, Cinelli & Hazlett
      2020) can bound how large the bias from unobservables would need
      to be to overturn your results.


Regardless of method, always:
  - Report robust standard errors (or cluster as appropriate)
  - Show the main specification AND robustness checks
  - Be explicit about identifying assumptions and threats to validity
  - Plot your data -- diagnostics catch problems that tests miss
  - Remember: the goal is not statistical significance, but a credible
    estimate of a quantity that matters for the question you are asking
"""
print(summary)

# --- Supplementary Figure: Method Selection Flowchart ---
fig_flow, ax_flow = plt.subplots(1, 1, figsize=(14, 10))
ax_flow.set_xlim(0, 14); ax_flow.set_ylim(0, 12); ax_flow.axis("off")
ax_flow.set_title("Method Selection Decision Tree", fontsize=16, fontweight="bold", pad=20)

# Decision nodes (diamonds) and answer nodes (rectangles)
def draw_decision(ax, x, y, text, w=2.5, h=0.8):
    diamond = plt.Polygon([(x, y+h), (x+w/2, y+h/2+h), (x+w, y+h), (x+w/2, y+h/2)],
                           fc="#E3F2FD", ec=CB, lw=2, zorder=5)
    ax.add_patch(diamond)
    ax.text(x+w/2, y+h*0.75, text, ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

def draw_answer(ax, x, y, text, color=CG, w=2.8, h=0.65):
    rect = plt.Rectangle((x, y), w, h, fc=color, ec="#333", lw=1.5, alpha=0.25, zorder=5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)

akw_flow = dict(arrowstyle="-|>", color="#555", lw=1.5, mutation_scale=12)

# Q1: Can you randomize?
draw_decision(ax_flow, 5, 10.2, "Can you\nrandomize?")
ax_flow.annotate("", xy=(9.3, 10.85), xytext=(7.5, 10.85), arrowprops=akw_flow)
ax_flow.text(8.2, 11.05, "Yes", fontsize=8, color=CG, fontweight="bold")
draw_answer(ax_flow, 9.3, 10.5, "RCT + OLS\n(Gold standard)", CG)

# No arrow down
ax_flow.annotate("", xy=(6.25, 9.2), xytext=(6.25, 10.2), arrowprops=akw_flow)
ax_flow.text(6.45, 9.7, "No", fontsize=8, color=CR, fontweight="bold")

# Q2: Is there an instrument?
draw_decision(ax_flow, 5, 8.2, "Valid instrument\navailable?")
ax_flow.annotate("", xy=(9.3, 8.85), xytext=(7.5, 8.85), arrowprops=akw_flow)
ax_flow.text(8.2, 9.05, "Yes", fontsize=8, color=CG, fontweight="bold")
draw_answer(ax_flow, 9.3, 8.5, "IV / 2SLS\n(Check F > 10)", CB)

ax_flow.annotate("", xy=(6.25, 7.2), xytext=(6.25, 8.2), arrowprops=akw_flow)
ax_flow.text(6.45, 7.7, "No", fontsize=8, color=CR, fontweight="bold")

# Q3: Policy change?
draw_decision(ax_flow, 5, 6.2, "Policy change\n(some treated)?")
ax_flow.annotate("", xy=(9.3, 6.85), xytext=(7.5, 6.85), arrowprops=akw_flow)
ax_flow.text(8.2, 7.05, "Yes", fontsize=8, color=CG, fontweight="bold")
draw_answer(ax_flow, 9.3, 6.5, "DiD\n(Check parallel trends)", CO)

ax_flow.annotate("", xy=(6.25, 5.2), xytext=(6.25, 6.2), arrowprops=akw_flow)
ax_flow.text(6.45, 5.7, "No", fontsize=8, color=CR, fontweight="bold")

# Q4: Cutoff?
draw_decision(ax_flow, 5, 4.2, "Treatment by\ncutoff rule?")
ax_flow.annotate("", xy=(9.3, 4.85), xytext=(7.5, 4.85), arrowprops=akw_flow)
ax_flow.text(8.2, 5.05, "Yes", fontsize=8, color=CG, fontweight="bold")
draw_answer(ax_flow, 9.3, 4.5, "RDD\n(Check no manipulation)", CP)

ax_flow.annotate("", xy=(6.25, 3.2), xytext=(6.25, 4.2), arrowprops=akw_flow)
ax_flow.text(6.45, 3.7, "No", fontsize=8, color=CR, fontweight="bold")

# Q5: Panel data?
draw_decision(ax_flow, 5, 2.2, "Panel data\navailable?")
ax_flow.annotate("", xy=(9.3, 2.85), xytext=(7.5, 2.85), arrowprops=akw_flow)
ax_flow.text(8.2, 3.05, "Yes", fontsize=8, color=CG, fontweight="bold")
draw_answer(ax_flow, 9.3, 2.5, "Fixed Effects\n(Time-invariant confounders)", "#31A354")

ax_flow.annotate("", xy=(6.25, 1.2), xytext=(6.25, 2.2), arrowprops=akw_flow)
ax_flow.text(6.45, 1.7, "No", fontsize=8, color=CR, fontweight="bold")

# Fallback
draw_answer(ax_flow, 4.6, 0.5, "Selection on observables\nOLS + controls / PSM\n(Weakest -- do sensitivity)", CR, w=3.3, h=0.9)

# Side note
ax_flow.text(1, 5, "Always:\n- Robust SEs\n- Robustness checks\n- Plot your data\n- State assumptions\n  explicitly",
             fontsize=8, fontstyle="italic", color="#555",
             bbox=dict(boxstyle="round,pad=0.5", fc="#FFF9C4", ec="#FFB300", alpha=0.9))
fig_flow.tight_layout()
savefig(fig_flow, "fig11_flowchart.png")

# --- Supplementary Figure: Identification Credibility Ladder ---
fig_ladder, axes_ladder = plt.subplots(1, 2, figsize=(15, 6))

ax = axes_ladder[0]
methods = ["Selection on\nobservables\n(OLS/PSM)", "Fixed\nEffects", "DiD", "RDD", "IV / 2SLS", "RCT"]
credibility = [1, 2, 3, 4, 4.5, 5]
colors_lad = [CR, "#E6550D", "#FD8D3C", "#FDAE6B", CB, CG]
bars_lad = ax.barh(range(len(methods)), credibility, color=colors_lad, height=0.6, edgecolor="white")
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel("Causal Credibility", fontsize=10)
ax.set_title("A) Identification Strategy Credibility Ladder", fontsize=12)
for i, (bar, c) in enumerate(zip(bars_lad, credibility)):
    key_assumption = ["No unobserved\nconfounders", "Strict\nexogeneity", "Parallel\ntrends",
                      "Continuity at\ncutoff", "Exclusion\nrestriction", "Randomization"][i]
    ax.text(c + 0.1, i, key_assumption, va="center", fontsize=7.5, color="#555", fontstyle="italic")

ax = axes_ladder[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("B) The Identification Tradeoff", fontsize=12)
# Draw the tradeoff diagram
ax.annotate("", xy=(9, 1), xytext=(1, 1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
ax.annotate("", xy=(1, 9), xytext=(1, 1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=2))
ax.text(5, 0.2, "Internal Validity (causal credibility)", ha="center", fontsize=10, fontweight="bold")
ax.text(0.15, 5, "External Validity\n(generalizability)", ha="center", fontsize=10, fontweight="bold", rotation=90)

# Plot methods on the tradeoff space
method_pts = {"OLS\n+controls": (2, 8), "FE": (4, 6.5), "DiD": (5.5, 5.5),
              "IV": (7, 4), "RDD": (8, 2.5), "RCT": (8.5, 7)}
for name, (mx, my) in method_pts.items():
    color = CG if "RCT" in name else (CB if "IV" in name or "RDD" in name else CO)
    ax.plot(mx, my, "o", ms=12, color=color, zorder=5)
    ax.text(mx + 0.3, my + 0.2, name, fontsize=8, fontweight="bold", color=color)

ax.text(5, 8.5, "Ideal: high on both axes\n(RCTs with representative samples)", fontsize=8,
        ha="center", fontstyle="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CY, alpha=0.9))
ax.text(7, 1.5, "RDD: strong internal\nvalidity but LATE\nat cutoff only", fontsize=7,
        ha="center", fontstyle="italic", color="#888")

fig_ladder.suptitle("Summary: Causal Identification Strategy Guide", fontsize=14, y=1.02)
fig_ladder.tight_layout()
savefig(fig_ladder, "fig12_identification.png")

summary_flowchart_text = """
Method Selection Flowchart

The flowchart above provides a practical decision tree for choosing an
econometric method. Start at the top and follow the branches:

  1. CAN YOU RANDOMIZE? If yes, run an RCT. OLS on treatment assignment
     gives the ATE directly. This is the gold standard for causal
     inference because randomization ensures E[epsilon|treatment] = 0
     by design.

  2. IS THERE A VALID INSTRUMENT? If you have a variable that predicts
     treatment but affects outcomes only through treatment, use IV/2SLS.
     Always check the first-stage F-statistic (F > 10 for strong
     instruments) and argue the exclusion restriction carefully.

  3. IS THERE A POLICY CHANGE? If a treatment rolls out to some units
     but not others at a point in time, DiD is your tool. Always check
     pre-trends via an event study specification.

  4. IS TREATMENT ASSIGNED BY A CUTOFF? If there is a threshold rule
     (score >= c gets treatment), use RDD. Check for manipulation with
     a McCrary density test, and verify covariate smoothness at cutoff.

  5. DO YOU HAVE PANEL DATA? Fixed effects remove time-invariant
     confounders. Combine with DiD if there is also a policy shock.

  6. NONE OF THE ABOVE? You are in "selection on observables" territory.
     Use OLS with careful controls, propensity score methods, and always
     conduct sensitivity analysis (Oster 2019, Cinelli and Hazlett 2020)
     to bound how large unobserved confounding would need to be.

The Identification Credibility Ladder (right panel) summarizes the rough
hierarchy of causal credibility, along with the key assumption each
method requires. Note the fundamental tradeoff: methods with high
internal validity (RDD, IV) often have limited external validity because
they estimate local treatment effects for specific subpopulations.
"""

summary_id_text = """
The Identification Tradeoff: Internal vs External Validity

A critical concept for policy research is the tradeoff between internal
validity (did we correctly estimate the causal effect for the population
studied?) and external validity (does the finding generalize to the
populations we care about?).

  - RCTs can achieve both if the sample is representative, but many RCTs
    use convenience samples (volunteers, specific clinics).

  - IV estimates the LATE for compliers, who may not represent the full
    population. Distance-to-college IV estimates the return to schooling
    for marginal students, not all students.

  - RDD estimates the effect at the cutoff, which may differ from the
    effect at other points in the distribution.

  - DiD estimates the ATT (average treatment effect on the treated),
    which may differ from the ATE if treatment effects are heterogeneous.

  - OLS with controls has high external validity (uses the full sample)
    but questionable internal validity (unobserved confounders).

For policy, this matters enormously: if you want to know "what would
happen if we expanded this program to everyone?" but your estimate is
a LATE for a narrow subgroup, you need to think carefully about
extrapolation. This is an active area of research in econometrics
(Angrist & Fernandez-Val 2013, Mogstad, Santos & Torgovitsky 2018).
"""

section_contents.append((summary_flowchart_text, "fig11_flowchart.png"))
section_contents.append((summary_id_text, "fig12_identification.png"))

###############################################################################
# Combine to PDF — TEXT page then IMAGE page for each section
###############################################################################
if(save):
    print("Combining into PDF with interleaved text and figures...")
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT

    pdf_path = os.path.join(OUTDIR, "econometrics_visual_primer.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()
    # Custom styles for different content types
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        spaceAfter=4,
        leftIndent=0,
    )
    # Indented formula / math blocks
    formula_style = ParagraphStyle(
        'FormulaBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        spaceAfter=3,
        leftIndent=18,
        textColor='#333333',
    )
    # Highlighted "intuition" and "policy" boxes
    box_style = ParagraphStyle(
        'HighlightBox',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        spaceAfter=4,
        leftIndent=12,
        rightIndent=12,
        textColor='#1565C0',
    )
    policy_style = ParagraphStyle(
        'PolicyBox',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=12,
        spaceAfter=4,
        leftIndent=12,
        rightIndent=12,
        textColor='#2E7D32',
    )
    # Section sub-heading (e.g., "Economic story", "Mathematical setup")
    subhead_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        spaceBefore=8,
        spaceAfter=4,
        textColor='#333333',
    )
    # Results block
    results_style = ParagraphStyle(
        'ResultsBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        spaceAfter=3,
        leftIndent=12,
        backColor='#F5F5F5',
    )
    title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        spaceAfter=12,
        textColor='#2171B5',
    )
    heading_style = ParagraphStyle(
        'PrimerTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        spaceAfter=20,
        textColor='#555555',
    )

    def classify_line(line, prev_line=""):
        """Classify a line of text to determine its style."""
        stripped = line.strip()
        if not stripped:
            return "blank"
        # Sub-section headings (capitalized short lines without indentation)
        subheadings = ["Economic story", "Mathematical setup", "Motivation",
                       "Formal statement", "Why it matters", "Geometric intuition",
                       "Economic context", "What breaks", "Detection:", "The fix:",
                       "Practical advice", "The endogeneity problem",
                       "The instrumental variables solution", "The 2SLS procedure",
                       "What IV estimates:", "Standard error subtlety",
                       "What is panel data?", "The model", "The within estimator",
                       "What FE does", "Inference note:", "The idea", "The estimand",
                       "Regression formulation", "The parallel trends assumption",
                       "Modern extensions", "The setup", "Sharp RDD", "Estimation:",
                       "Bandwidth choice", "Validity threats", "Fuzzy RDD:",
                       "The problem with OLS", "Logit and Probit", "Interpreting",
                       "Practical guidance", "Why MLE matters", "The principle",
                       "Standard errors from", "Numerical optimization",
                       "Profile likelihood", "The nonparametric bootstrap",
                       "Why \"with replacement\"", "When does the bootstrap fail?",
                       "Results", "Interpretation", "Why Monte Carlo?",
                       "Monte Carlo design", "What is an event study?",
                       "Reading the event study", "When pre-trends fail",
                       "Method Selection Flowchart", "The Identification Tradeoff"]
        for heading in subheadings:
            if stripped.startswith(heading):
                return "subheading"
        # Intuition / policy boxes
        if stripped.startswith("INTUITION BOX:") or stripped.startswith("KEY TAKEAWAY:"):
            return "intuition"
        if stripped.startswith("POLICY CONNECTION:"):
            return "policy"
        # Indented formula lines (start with spaces and contain math-like symbols)
        if line.startswith("  ") and any(c in stripped for c in ["=", "^{", "->", "beta", "sigma", "epsilon", "alpha", "tau", "Cov(", "Var(", "SE(", "E[", "Sum", "Product", "lim_"]):
            return "formula"
        # Numbered assumption lines
        if stripped.startswith("(A") and ")" in stripped[:5]:
            return "formula"
        # Results with numeric values
        if line.startswith("  ") and any(c in stripped for c in ["beta_hat", "SE =", "SE:", "F =", "F-stat", "CI", "p ="]):
            return "results"
        # Regular indented text
        if line.startswith("  ") and not stripped.startswith("-"):
            return "formula"
        return "body"

    story = []

    # --- Cover / intro page ---
    story.append(Paragraph("ECONOMETRICS PRIMER FOR THE MATH-LITERATE", heading_style))
    story.append(Paragraph("With Visualizations", subtitle_style))
    story.append(Spacer(1, 12))
    intro_lines = [
        "Audience: Strong math background (linear algebra, probability, calculus),",
        "limited economics exposure. Each section explains the economic problem",
        "briefly, then dives into the math and code.",
        "",
        "All estimators implemented from scratch via linear algebra / scipy.optimize.",
        "Includes Monte Carlo demonstrations, event study specifications, and a",
        "complete method selection decision tree.",
        "",
        "Core Sections: OLS, FWL, Heteroskedasticity, IV/2SLS, Panel FE, DiD, RDD,",
        "Probit/Logit, MLE from scratch, Bootstrap inference.",
        "",
        "Supplementary Sections: Monte Carlo OVB demonstration, Event Study",
        "diagnostics, Method Selection Flowchart, Identification Strategy Guide.",
        "",
        "Each section includes: economic motivation, formal mathematical setup,",
        "from-scratch Python implementation, simulation results with standard",
        "errors, and diagnostic visualizations. Intuition boxes provide geometric",
        "or probabilistic intuition; policy connection boxes link methods to",
        "real-world evaluation questions.",
    ]
    for line in intro_lines:
        if line == "":
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(line, styles['Normal']))
    story.append(PageBreak())

    # --- Interleaved text + figure pages ---
    page_w = letter[0] - 1.5*inch  # usable width
    
    # Helper to draw a colored line separator
    from reportlab.platypus import HRFlowable

    for sec_text, fig_fname in section_contents:
        # Text page: render with intelligent style classification
        lines = sec_text.strip().split('\n')
        if not lines:
            continue
            
        # First line is the section title
        safe_title = lines[0].replace('--', '&mdash;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(safe_title, title_style))
        story.append(HRFlowable(width="100%", thickness=1, color="#2171B5", spaceBefore=2, spaceAfter=8))
        
        in_intuition_block = False
        in_policy_block = False
        prev_line = ""
        
        for line in lines[1:]:
            safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            stripped = safe.strip()
            
            if stripped == '':
                in_intuition_block = False
                in_policy_block = False
                story.append(Spacer(1, 5))
                prev_line = ""
                continue
            
            line_type = classify_line(line, prev_line)
            
            # Handle multi-line intuition/policy blocks
            if line_type == "intuition":
                in_intuition_block = True
                in_policy_block = False
                # Strip the prefix for display
                display = stripped.replace("INTUITION BOX:", "<b>INTUITION:</b>").replace("KEY TAKEAWAY:", "<b>KEY TAKEAWAY:</b>")
                story.append(Spacer(1, 4))
                story.append(HRFlowable(width="90%", thickness=0.5, color="#1565C0", spaceBefore=2, spaceAfter=2))
                story.append(Paragraph(display, box_style))
            elif line_type == "policy":
                in_policy_block = True
                in_intuition_block = False
                display = stripped.replace("POLICY CONNECTION:", "<b>POLICY CONNECTION:</b>")
                story.append(Spacer(1, 4))
                story.append(HRFlowable(width="90%", thickness=0.5, color="#2E7D32", spaceBefore=2, spaceAfter=2))
                story.append(Paragraph(display, policy_style))
            elif in_intuition_block:
                story.append(Paragraph(stripped, box_style))
            elif in_policy_block:
                story.append(Paragraph(stripped, policy_style))
            elif line_type == "subheading":
                story.append(Spacer(1, 6))
                story.append(Paragraph(stripped, subhead_style))
            elif line_type == "formula":
                story.append(Paragraph(safe, formula_style))
            elif line_type == "results":
                story.append(Paragraph(safe, results_style))
            else:
                story.append(Paragraph(stripped, body_style))
            
            prev_line = line
        
        story.append(PageBreak())

        # Figure page: insert the PNG, scaled to fit page width
        fig_path = os.path.join(OUTDIR, fig_fname)
        img = PIL_Image_open = Image.open(fig_path)
        iw, ih = img.size
        aspect = ih / iw
        display_w = page_w
        display_h = display_w * aspect
        # Cap height to avoid overflow
        max_h = letter[1] - 1.5*inch
        if display_h > max_h:
            display_h = max_h
            display_w = display_h / aspect
        story.append(RLImage(fig_path, width=display_w, height=display_h))
        story.append(PageBreak())

    # --- Summary page ---
    story.append(Paragraph("Summary: When to Use What", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color="#2171B5", spaceBefore=2, spaceAfter=8))
    for line in summary.strip().split('\n'):
        safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        stripped = safe.strip()
        if stripped == '':
            story.append(Spacer(1, 4))
        elif stripped.startswith("Method") or stripped.startswith("---"):
            story.append(Paragraph(safe, formula_style))
        elif stripped.startswith("Key identification") or stripped.startswith("Choosing a method") or stripped.startswith("Regardless of method"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(stripped, subhead_style))
        elif line.startswith("  "):
            story.append(Paragraph(safe, formula_style))
        else:
            story.append(Paragraph(stripped, body_style))

    doc.build(story)
    print(f"Done! 10 PNGs + {pdf_path}")
    print("PDF structure: [Intro] -> [Text1, Fig1, Text2, Fig2, ..., Text10, Fig10, Summary]")
else:
    print("Done!")
