"""
Ejercicio 1 – Dinámica de Fluidos en Tuberías
IF4702 – Física Computacional I

Modelos de regresión y clasificación para predecir la caída de presión
y el régimen de flujo en función de la velocidad media del fluido.

Regresión  (10 modelos): Lineal, Polinomial, Árbol, KNN, SVR,
           Random Forest, Gradient Boosting, MLP-ReLU, MLP-tanh, MLP-sigmoid
Clasificación (10 modelos): Logística, Árbol, KNN, SVM, Random Forest,
           Gradient Boosting, MLP-ReLU, MLP-tanh, MLP-sigmoid, Naive Bayes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, log_loss

plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('datos_flujo_tuberia.csv')
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")
print(df.head())


# ═════════════════════════════════════════════════════════════════════════════
#
#  PARTE 1 – REGRESIÓN
#
# ═════════════════════════════════════════════════════════════════════════════

X = df[['velocidad_ms']].values   # entrada: velocidad (m/s)
y = df['deltaP_Pa'].values        # objetivo: caída de presión (Pa)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[Regresión] Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

v_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def metricas_reg(y_test, y_pred, nombre):
    R2  = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    print(f"\n─── {nombre} ───")
    print(f"  R²  = {R2:.4f}")
    print(f"  MSE = {MSE:.4f}  Pa²")
    return R2, MSE

def plot_reg(v_rng, y_rng, label, color, titulo):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, color='steelblue', alpha=0.6, s=40,
               label='Datos experimentales')
    ax.plot(v_rng, y_rng, color=color, lw=2, label=label)
    ax.set_xlabel('Velocidad media $v$  (m/s)')
    ax.set_ylabel('Caída de presión $\\Delta P$  (Pa)')
    ax.set_title(titulo)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ─── REG 01: Regresión Lineal ─────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 01 – Regresión Lineal")
print("═"*60)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

pendiente  = modelo.coef_[0]
intercepto = modelo.intercept_
y_pred     = modelo.predict(X_test)

print(f"  Modelo: ΔP = {pendiente:.4f}·v + ({intercepto:.4f})")
R2, MSE = metricas_reg(y_test, y_pred, "Métricas")

plot_reg(v_rng, modelo.predict(v_rng),
         f'Lineal  ($R^2={R2:.3f}$)', 'crimson',
         'Regresión Lineal: $v$ vs $\\Delta P$')

# Análisis: exacta en régimen laminar (ΔP ∝ v), insuficiente en turbulento
# donde la relación sigue ΔP ∝ v^1.75 (Blasius). No apta para todo el rango.


# ─── REG 02: Regresión Polinomial ────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 02 – Regresión Polinomial")
print("═"*60)

mejorR2, mejor_modelo, mejor_grado = -np.inf, None, None
for g in [2, 3, 4]:
    pipe = Pipeline([('poly', PolynomialFeatures(degree=g, include_bias=False)),
                     ('lr',   LinearRegression())])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"  Grado {g}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_grado = r2, pipe, g

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor grado={mejor_grado}")

plot_reg(v_rng, mejor_modelo.predict(v_rng),
         f'Polinomio grado {mejor_grado}  ($R^2={R2:.3f}$)', 'darkorange',
         f'Regresión Polinomial (grado {mejor_grado}): $v$ vs $\\Delta P$')

# Análisis: captura la curvatura turbulenta mejor que el modelo lineal.
# Grado 2-3 aproxima bien la ley de Blasius; grados altos pueden sobreajustar.


# ─── REG 03: Árbol de Decisión ───────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 03 – Árbol de Decisión")
print("═"*60)

mejorR2, mejor_modelo, mejor_prof = -np.inf, None, None
for d in [2, 4, 6, 8]:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    mse = mean_squared_error(y_test, m.predict(X_test))
    print(f"  depth={d}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_prof = r2, m, d

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor depth={mejor_prof}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
axes[0].plot(v_rng, mejor_modelo.predict(v_rng), color='forestgreen', lw=2,
             label=f'Árbol depth={mejor_prof}  ($R^2={R2:.3f}$)')
axes[0].set_xlabel('Velocidad media $v$  (m/s)')
axes[0].set_ylabel('Caída de presión $\\Delta P$  (Pa)')
axes[0].set_title('Árbol de Decisión: $v$ vs $\\Delta P$')
axes[0].legend()
plot_tree(mejor_modelo, max_depth=3, feature_names=['v (m/s)'],
          filled=True, rounded=True, ax=axes[1], fontsize=8)
axes[1].set_title(f'Estructura del árbol (depth={mejor_prof})')
plt.tight_layout()
plt.show()

# Análisis: aproxima ΔP con particiones constantes a trozos. Captura bien la
# discontinuidad entre regímenes, pero profundidades altas memorizan el ruido.


# ─── REG 04: KNN ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 04 – KNN Regressor")
print("═"*60)

mejorR2, mejor_modelo, mejor_k = -np.inf, None, None
for k in [1, 3, 5, 7, 10, 15]:
    m = KNeighborsRegressor(n_neighbors=k)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    mse = mean_squared_error(y_test, m.predict(X_test))
    print(f"  k={k:2d}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_k = r2, m, k

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor k={mejor_k}")

plot_reg(v_rng, mejor_modelo.predict(v_rng),
         f'KNN k={mejor_k}  ($R^2={R2:.3f}$)', 'darkorchid',
         f'Regresión KNN (k={mejor_k}): $v$ vs $\\Delta P$')

# Análisis: promedia los k vecinos más cercanos. k=1 memoriza; k grande
# sobresuaviza. Sensible a la escala de v (sin normalización puede fallar).


# ─── REG 05: SVR ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 05 – SVR")
print("═"*60)

colores_kernel = {'linear': 'crimson', 'poly': 'darkorange', 'rbf': 'forestgreen'}
mejorR2, mejor_modelo, mejor_kernel = -np.inf, None, None
for k in ['linear', 'poly', 'rbf']:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svr',    SVR(kernel=k, C=1000, epsilon=0.1))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"  kernel={k:6s}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_kernel = r2, pipe, k

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor kernel={mejor_kernel}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
for k in ['linear', 'poly', 'rbf']:
    pipe_k = Pipeline([('scaler', StandardScaler()),
                       ('svr',    SVR(kernel=k, C=1000, epsilon=0.1))])
    pipe_k.fit(X_train, y_train)
    ax.plot(v_rng, pipe_k.predict(v_rng), lw=1.8, color=colores_kernel[k],
            ls='-' if k == mejor_kernel else '--',
            alpha=1.0 if k == mejor_kernel else 0.6,
            label=f'SVR {k}' + ('  ← mejor' if k == mejor_kernel else ''))
ax.set_xlabel('Velocidad media $v$  (m/s)')
ax.set_ylabel('Caída de presión $\\Delta P$  (Pa)')
ax.set_title('SVR – Comparación de kernels')
ax.legend()
plt.tight_layout()
plt.show()

# Análisis: minimiza el error solo fuera del margen ε; robusta ante ruido.
# El kernel RBF es el más flexible. Requiere escalado previo obligatorio.


# ─── REG 06: Random Forest ───────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 06 – Random Forest Regressor")
print("═"*60)

mejorR2, mejor_modelo, mejor_n = -np.inf, None, None
for n in [10, 50, 100, 200]:
    m = RandomForestRegressor(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    mse = mean_squared_error(y_test, m.predict(X_test))
    print(f"  n={n:3d}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_n = r2, m, n

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor n={mejor_n}")

y_rng = mejor_modelo.predict(v_rng)
preds_arboles = np.array([t.predict(v_rng) for t in mejor_modelo.estimators_])
std_rng = preds_arboles.std(axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
ax.plot(v_rng, y_rng, color='saddlebrown', lw=2,
        label=f'Random Forest n={mejor_n}  ($R^2={R2:.3f}$)')
ax.fill_between(v_rng.ravel(), y_rng - std_rng, y_rng + std_rng,
                alpha=0.2, color='saddlebrown', label='±1σ entre árboles')
ax.set_xlabel('Velocidad media $v$  (m/s)')
ax.set_ylabel('Caída de presión $\\Delta P$  (Pa)')
ax.set_title(f'Random Forest (n={mejor_n}): $v$ vs $\\Delta P$')
ax.legend()
plt.tight_layout()
plt.show()

# Análisis: promedia árboles entrenados con subconjuntos aleatorios. La banda
# ±1σ es una medida informal de incertidumbre. No asume forma funcional.


# ─── REG 07: Gradient Boosting ───────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 07 – Gradient Boosting Regressor")
print("═"*60)

mejorR2, mejor_modelo, mejor_lr = -np.inf, None, None
for lr in [0.01, 0.05, 0.1, 0.2]:
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=lr,
                                   max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    mse = mean_squared_error(y_test, m.predict(X_test))
    print(f"  lr={lr:.2f}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_lr = r2, m, lr

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor lr={mejor_lr}")

train_err = [mean_squared_error(y_train, yp) for yp in mejor_modelo.staged_predict(X_train)]
test_err  = [mean_squared_error(y_test,  yp) for yp in mejor_modelo.staged_predict(X_test)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
axes[0].plot(v_rng, mejor_modelo.predict(v_rng), color='darkgreen', lw=2,
             label=f'GBR lr={mejor_lr}  ($R^2={R2:.3f}$)')
axes[0].set_xlabel('Velocidad media $v$  (m/s)')
axes[0].set_ylabel('Caída de presión $\\Delta P$  (Pa)')
axes[0].set_title('Gradient Boosting: $v$ vs $\\Delta P$')
axes[0].legend()
axes[1].plot(train_err, label='MSE entrenamiento', color='royalblue')
axes[1].plot(test_err,  label='MSE prueba',        color='crimson')
axes[1].set_xlabel('Iteración')
axes[1].set_ylabel('MSE  (Pa²)')
axes[1].set_title('Evolución del error por iteración')
axes[1].legend()
plt.tight_layout()
plt.show()

# Análisis: construye árboles secuencialmente corrigiendo el residuo anterior.
# La curva de error permite detectar sobreajuste (test sube mientras train baja).


# ─── REG 08: MLP ReLU ────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 08 – MLP Regressor (ReLU)")
print("═"*60)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPRegressor(hidden_layer_sizes=arq,
                                             activation='relu',
                                             max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor arq={mejor_arq}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
axes[0].plot(v_rng, mejor_modelo.predict(v_rng), color='tomato', lw=2,
             label=f'MLP ReLU {mejor_arq}  ($R^2={R2:.3f}$)')
axes[0].set_xlabel('Velocidad media $v$  (m/s)')
axes[0].set_ylabel('Caída de presión $\\Delta P$  (Pa)')
axes[0].set_title('MLP ReLU: $v$ vs $\\Delta P$')
axes[0].legend()
axes[1].plot(mejor_modelo.named_steps['mlp'].loss_curve_, color='tomato', lw=1.8)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Curva de pérdida – entrenamiento')
plt.tight_layout()
plt.show()

# Análisis: ReLU evita el gradiente desvanecido y converge rápido. Requiere
# escalado previo. Puede aproximar cualquier función continua con suficientes
# neuronas (teorema de aproximación universal).


# ─── REG 09: MLP tanh ────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 09 – MLP Regressor (tanh)")
print("═"*60)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPRegressor(hidden_layer_sizes=arq,
                                             activation='tanh',
                                             max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor arq={mejor_arq}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
axes[0].plot(v_rng, mejor_modelo.predict(v_rng), color='mediumseagreen', lw=2,
             label=f'MLP tanh {mejor_arq}  ($R^2={R2:.3f}$)')
axes[0].set_xlabel('Velocidad media $v$  (m/s)')
axes[0].set_ylabel('Caída de presión $\\Delta P$  (Pa)')
axes[0].set_title('MLP tanh: $v$ vs $\\Delta P$')
axes[0].legend()
axes[1].plot(mejor_modelo.named_steps['mlp'].loss_curve_, color='mediumseagreen', lw=1.8)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Curva de pérdida – entrenamiento')
plt.tight_layout()
plt.show()

# Análisis: tanh produce salidas simétricas en [-1,1]; facilita la convergencia
# cuando los datos están centrados. Puede saturarse en valores extremos.


# ─── REG 10: MLP sigmoid ─────────────────────────────────────────────────────
print("\n" + "═"*60)
print("REG 10 – MLP Regressor (sigmoid)")
print("═"*60)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPRegressor(hidden_layer_sizes=arq,
                                             activation='logistic',
                                             max_iter=3000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    mse = mean_squared_error(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R²={r2:.4f}  |  MSE={mse:.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
R2, MSE = metricas_reg(y_test, y_pred, f"Mejor arq={mejor_arq}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color='steelblue', alpha=0.6, s=40, label='Datos experimentales')
axes[0].plot(v_rng, mejor_modelo.predict(v_rng), color='goldenrod', lw=2,
             label=f'MLP sigmoid {mejor_arq}  ($R^2={R2:.3f}$)')
axes[0].set_xlabel('Velocidad media $v$  (m/s)')
axes[0].set_ylabel('Caída de presión $\\Delta P$  (Pa)')
axes[0].set_title('MLP sigmoid: $v$ vs $\\Delta P$')
axes[0].legend()
axes[1].plot(mejor_modelo.named_steps['mlp'].loss_curve_, color='goldenrod', lw=1.8)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Curva de pérdida – entrenamiento')
plt.tight_layout()
plt.show()

# Análisis: σ(x) = 1/(1+e^{-x}) mapea a (0,1); más propensa al gradiente
# desvanecido que ReLU o tanh. Históricamente la primera activación usada;
# hoy se prefiere ReLU para regresión.


# ═════════════════════════════════════════════════════════════════════════════
#
#  PARTE 2 – CLASIFICACIÓN
#
# ═════════════════════════════════════════════════════════════════════════════

X = df[['velocidad_ms']].values   # entrada: velocidad (m/s)
y = df['turbulento'].values       # objetivo: 1=turbulento, 0=laminar

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[Clasificación] Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

v_rng = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
colores   = {0: 'steelblue', 1: 'orangered'}
etiquetas = {0: 'Laminar (0)', 1: 'Turbulento (1)'}


# ─── Helpers ─────────────────────────────────────────────────────────────────
def metricas_clf(y_test, y_pred, y_proba, nombre):
    acc  = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    print(f"\n─── {nombre} ───")
    print(f"  Accuracy      = {acc:.4f}")
    print(f"  Cross-entropy = {loss:.4f}")
    return acc, loss

def frontera_clf(modelo, v_rng):
    cambios = np.where(np.diff(modelo.predict(v_rng)))[0]
    return v_rng[cambios[0], 0] if len(cambios) > 0 else None

def plot_clf(modelo, v_rng, proba, v_frontera, color, titulo):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for clase in [0, 1]:
        m = df['turbulento'] == clase
        ax1.scatter(df.loc[m, 'velocidad_ms'], df.loc[m, 'deltaP_Pa'],
                    color=colores[clase], alpha=0.7, s=45, label=etiquetas[clase])
    ax1.set_xlabel('Velocidad media $v$  (m/s)')
    ax1.set_ylabel('Caída de presión $\\Delta P$  (Pa)')
    ax1.set_title('Figura 1 – Datos por clase')
    ax1.legend()

    ax2.plot(v_rng, proba, color=color, lw=2.5,
             label='$P(\\mathrm{turbulento}|v)$')
    if v_frontera is not None:
        ax2.axvline(v_frontera, color='black', ls='--', lw=1.5,
                    label=f'Frontera  $v≈{v_frontera:.3f}$ m/s')
    ax2.axhline(0.5, color='gray', ls=':', lw=1)
    ax2.fill_between(v_rng.ravel(), proba, 0.5,
                     where=(proba >= 0.5), alpha=0.12, color='orangered')
    ax2.fill_between(v_rng.ravel(), proba, 0.5,
                     where=(proba < 0.5),  alpha=0.12, color='steelblue')
    ax2.set_xlabel('Velocidad media $v$  (m/s)')
    ax2.set_ylabel('Probabilidad predicha')
    ax2.set_title('Figura 2 – Probabilidad de flujo turbulento')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=9)

    plt.suptitle(titulo, fontsize=13)
    plt.tight_layout()
    plt.show()


# ─── CLF 01: Regresión Logística ─────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 01 – Regresión Logística")
print("═"*60)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, "Métricas")

v_frontera = -modelo.intercept_[0] / modelo.coef_[0][0]
print(f"  Frontera analítica: v = {v_frontera:.4f} m/s")

proba = modelo.predict_proba(v_rng)[:, 1]
plot_clf(modelo, v_rng, proba, v_frontera, 'purple', 'Regresión Logística')

# Interpretación: Re ∝ v → existe v crítica que separa regímenes. La velocidad
# sola es suficiente; el alto accuracy confirma la separabilidad lineal.


# ─── CLF 02: Árbol de Decisión ───────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 02 – Árbol de Decisión")
print("═"*60)

mejorAcc, mejor_modelo, mejor_prof = -np.inf, None, None
for d in [2, 4, 6, 8]:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    loss = log_loss(y_test, m.predict_proba(X_test))
    print(f"  depth={d}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_prof = acc, m, d

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor depth={mejor_prof}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'forestgreen',
         f'Árbol de Decisión (depth={mejor_prof})')

# Interpretación: con una sola variable la frontera es siempre un punto de
# corte, coherente con el criterio de Re crítico bien definido.


# ─── CLF 03: KNN ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 03 – KNN Classifier")
print("═"*60)

mejorAcc, mejor_modelo, mejor_k = -np.inf, None, None
for k in [1, 3, 5, 7, 10, 15]:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    loss = log_loss(y_test, m.predict_proba(X_test))
    print(f"  k={k:2d}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_k = acc, m, k

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor k={mejor_k}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'darkorchid',
         f'KNN Classifier (k={mejor_k})')

# Interpretación: probabilidad escalonada en vez de continua. k=1 memoriza;
# k grande sobresuaviza la frontera entre regímenes.


# ─── CLF 04: SVM ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 04 – SVM Classifier")
print("═"*60)

mejorAcc, mejor_modelo, mejor_kernel = -np.inf, None, None
for k in ['linear', 'poly', 'rbf']:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svc',    SVC(kernel=k, C=10,
                                    probability=True, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    loss = log_loss(y_test, pipe.predict_proba(X_test))
    print(f"  kernel={k:6s}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_kernel = acc, pipe, k

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor kernel={mejor_kernel}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera,
         colores_kernel[mejor_kernel], f'SVM (kernel={mejor_kernel})')

# Interpretación: SVM maximiza el margen entre clases. Con una variable la
# frontera es un punto; equivale al criterio de Re crítico.


# ─── CLF 05: Random Forest ───────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 05 – Random Forest Classifier")
print("═"*60)

mejorAcc, mejor_modelo, mejor_n = -np.inf, None, None
for n in [10, 50, 100, 200]:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    loss = log_loss(y_test, m.predict_proba(X_test))
    print(f"  n={n:3d}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_n = acc, m, n

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor n={mejor_n}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'saddlebrown',
         f'Random Forest Classifier (n={mejor_n})')

# Interpretación: probabilidad más suave que un árbol individual. Refleja
# mejor la incertidumbre en la zona de transición entre regímenes.


# ─── CLF 06: Gradient Boosting ───────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 06 – Gradient Boosting Classifier")
print("═"*60)

mejorAcc, mejor_modelo, mejor_lr = -np.inf, None, None
for lr in [0.01, 0.05, 0.1, 0.2]:
    m = GradientBoostingClassifier(n_estimators=200, learning_rate=lr,
                                    max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    loss = log_loss(y_test, m.predict_proba(X_test))
    print(f"  lr={lr:.2f}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_lr = acc, m, lr

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor lr={mejor_lr}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'darkgreen',
         f'Gradient Boosting Classifier (lr={mejor_lr})')

# Interpretación: curva sigmoide más pronunciada que otros métodos; el modelo
# reduce el error de clasificación residual de forma agresiva en la transición.


# ─── CLF 07: MLP ReLU ────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 07 – MLP Classifier (ReLU)")
print("═"*60)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPClassifier(hidden_layer_sizes=arq,
                                              activation='relu',
                                              max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    loss = log_loss(y_test, pipe.predict_proba(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor arq={mejor_arq}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'tomato',
         f'MLP Classifier ReLU {mejor_arq}')

# Interpretación: la red aprende la frontera sin asumir forma funcional. ReLU
# introduce no linealidades que capturan la discontinuidad entre regímenes.


# ─── CLF 08: MLP tanh ────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 08 – MLP Classifier (tanh)")
print("═"*60)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPClassifier(hidden_layer_sizes=arq,
                                              activation='tanh',
                                              max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    loss = log_loss(y_test, pipe.predict_proba(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor arq={mejor_arq}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'mediumseagreen',
         f'MLP Classifier tanh {mejor_arq}')

# Interpretación: curva de probabilidad más suave que ReLU; puede reflejar
# mejor la transición gradual en la zona de Re entre 2300 y 4000.


# ─── CLF 09: MLP sigmoid ─────────────────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 09 – MLP Classifier (sigmoid)")
print("═"*60)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('mlp',    MLPClassifier(hidden_layer_sizes=arq,
                                              activation='logistic',
                                              max_iter=3000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    loss = log_loss(y_test, pipe.predict_proba(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={loss:.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, f"Mejor arq={mejor_arq}")

proba      = mejor_modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(mejor_modelo, v_rng)
plot_clf(mejor_modelo, v_rng, proba, v_frontera, 'goldenrod',
         f'MLP Classifier sigmoid {mejor_arq}')

# Interpretación: σ(x) propensa al gradiente desvanecido en redes profundas.
# Para este problema unidimensional el impacto es menor.


# ─── CLF 10: Naive Bayes Gaussiano ───────────────────────────────────────────
print("\n" + "═"*60)
print("CLF 10 – Naive Bayes Gaussiano")
print("═"*60)

modelo = GaussianNB()
modelo.fit(X_train, y_train)

y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)
acc, loss = metricas_clf(y_test, y_pred, y_proba, "Métricas")

print("\n  Parámetros aprendidos por clase:")
for i, clase in enumerate([0, 1]):
    print(f"  Clase {clase}:  μ={modelo.theta_[i,0]:.4f} m/s  |"
          f"  σ²={modelo.var_[i,0]:.6f} m²/s²")

proba      = modelo.predict_proba(v_rng)[:, 1]
v_frontera = frontera_clf(modelo, v_rng)
plot_clf(modelo, v_rng, proba, v_frontera, 'teal', 'Naive Bayes Gaussiano')

# Interpretación: asume P(v|clase) ~ Normal. Coherente físicamente: flujos
# laminares se concentran en v bajas y turbulentos en altas. La media y
# varianza aprendidas corresponden a cada régimen de flujo.
