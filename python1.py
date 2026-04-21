import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Lectura de datos ───────────────────────────────────────────────────────
df = pd.read_csv('datos_flujo_tuberia.csv')

# ── 2. Variables de entrada y objetivo ───────────────────────────────────────
X = df[['velocidad_ms']].values   # variable de entrada: velocidad (m/s)
y = df['deltaP_Pa'].values        # variable objetivo: caída de presión (Pa)

# ── 3. División en entrenamiento y prueba ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Entrenamiento del modelo de regresión lineal ───────────────────────────
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# ── 5. Parámetros del ajuste y predicciones sobre el conjunto de prueba ───────
pendiente  = modelo.coef_[0]
intercepto = modelo.intercept_
y_pred     = modelo.predict(X_test)

print("─── Parámetros del ajuste lineal ───────────────────")
print(f"  Pendiente  (m) : {pendiente:.4f}  Pa·s/m")
print(f"  Intercepto (b) : {intercepto:.4f}  Pa")
print(f"  Modelo: ΔP = {pendiente:.4f}·v + ({intercepto:.4f})")

# ── 6. Evaluación del desempeño ───────────────────────────────────────────────
R2  = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

print("\n─── Métricas de desempeño ──────────────────────────")
print(f"  R²  = {R2:.4f}")
print(f"  MSE = {MSE:.4f}  Pa²")

# ── 7. Visualización: datos experimentales + recta de regresión ───────────────
v_rng  = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
dP_rng = modelo.predict(v_rng)

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(X, y, color='steelblue', alpha=0.65, s=40,
           label='Datos experimentales', zorder=3)
ax.plot(v_rng, dP_rng, color='crimson', lw=2,
        label=f'Regresión lineal  ($R^2 = {R2:.3f}$)')

ax.set_xlabel('Velocidad media $v$  (m/s)')
ax.set_ylabel('Caída de presión $\\Delta P$  (Pa)')
ax.set_title('Regresión lineal: velocidad vs caída de presión')
ax.legend()
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

# ── 8. Análisis: ¿es el modelo lineal suficiente? ─────────────────────────────
# En régimen laminar la relación ΔP ∝ v es exactamente lineal (Hagen-Poiseuille).
# Sin embargo, en régimen turbulento la caída de presión escala como v^1.75
# (correlación de Blasius), por lo que un único modelo lineal ajustado sobre
# todo el rango subestimará los valores a altas velocidades y sobreestimará en
# la zona de transición. El modelo lineal NO es suficiente para representar el
# sistema en todo el rango de velocidades; se necesitaría al menos un ajuste
# potencial o dos modelos separados por régimen.


