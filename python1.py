# ==============================================================================
#  TEMPLATE — Machine Learning con scikit-learn
#  IF4702 · Física Computacional I
#  Cambia SOLO el bloque de configuración y ejecuta.
# ==============================================================================

# ── CONFIGURACIÓN ──────────────────────────────────────────────────────────────
ARCHIVO_CSV        = "datos.csv"       # nombre del archivo

COL_ENTRADA        = "x"              # columna variable independiente (X)
COL_REGRESION      = "y"              # columna objetivo → REGRESIÓN
COL_CLASIFICACION  = "clase"          # columna objetivo → CLASIFICACIÓN (0/1)

ETIQUETA_X         = "Variable X"     # eje X en gráficas
ETIQUETA_Y_REG     = "Variable Y"     # eje Y en regresión
ETIQUETA_Y_CLF     = "Clase"          # eje Y en clasificación

VECINOS_KNN        = [3, 8, 20]       # valores de n_neighbors a comparar
K_ELEGIDO          = 8                # k seleccionado para el modelo final

TEST_SIZE          = 0.20             # fracción del conjunto de prueba
RANDOM_STATE       = 42               # semilla

# ── BIBLIOTECAS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split
from sklearn.neighbors        import KNeighborsRegressor
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (r2_score, mean_squared_error,
                                      accuracy_score, log_loss,
                                      confusion_matrix, ConfusionMatrixDisplay)

# ── ESTILOS ANSI para print en terminal ────────────────────────────────────────
_C  = "\033[96m"; _Y = "\033[93m"; _G = "\033[92m"
_B  = "\033[1m";  _R = "\033[0m"

# ==============================================================================
#  CARGA Y EXPLORACIÓN
# ==============================================================================
df = pd.read_csv(ARCHIVO_CSV)

print(f"{_B}{'─'*60}")
print(f"  DATASET: {ARCHIVO_CSV}")
print(f"{'─'*60}{_R}")
print(f"  Filas × columnas : {df.shape}")
print(f"  Columnas         : {list(df.columns)}")
print()
print(df.head())
print()
print(df.describe())

# ==============================================================================
#  PARTE 1 — REGRESIÓN KNN
# ==============================================================================
print(f"\n{_B}{'═'*60}")
print("  PARTE 1 — REGRESIÓN KNN")
print(f"{'═'*60}{_R}")

# ── 1–2. Variables ─────────────────────────────────────────────────────────────
X_reg = df[[COL_ENTRADA]].values          # matriz columna requerida por sklearn
y_reg = df[COL_REGRESION].values

# ── 3. División entrenamiento / prueba ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n  Entrenamiento: {len(X_train)} muestras  |  Prueba: {len(X_test)} muestras")

# Malla de puntos ordenada para graficar curvas suaves
X_plot = np.linspace(X_reg.min(), X_reg.max(), 300).reshape(-1, 1)

# ── 4–5. Comparación de n_neighbors ───────────────────────────────────────────
print(f"\n  {'k':>4} │ {'R²':>8} │ {'MSE':>12}")
print(f"  {'─'*4}─┼─{'─'*8}─┼─{'─'*12}")

modelos_knn = {}
for k in VECINOS_KNN:
    m = KNeighborsRegressor(n_neighbors=k)
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    r2  = r2_score(y_test, yp)
    mse = mean_squared_error(y_test, yp)
    print(f"  {k:>4} │ {r2:>8.4f} │ {mse:>12.4f}")
    modelos_knn[k] = dict(modelo=m, r2=r2, mse=mse)

# ── 6. Comparación visual de suavidad ─────────────────────────────────────────
fig, axes = plt.subplots(1, len(VECINOS_KNN),
                         figsize=(5 * len(VECINOS_KNN), 4), sharey=True)
for ax, k in zip(axes, VECINOS_KNN):
    m      = modelos_knn[k]["modelo"]
    y_pred = m.predict(X_test)
    y_curv = m.predict(X_plot)

    ax.scatter(X_reg, y_reg, s=12, alpha=0.35,
               color="steelblue", label="Datos")
    ax.scatter(X_test, y_pred, s=20, color="orange",
               zorder=3, label="Pred. prueba")
    ax.plot(X_plot, y_curv, color="crimson",
            linewidth=2, label=f"KNN k={k}")

    ax.set_title(f"k = {k}\nR²={modelos_knn[k]['r2']:.3f} | "
                 f"MSE={modelos_knn[k]['mse']:.3f}")
    ax.set_xlabel(ETIQUETA_X)
    if ax is axes[0]:
        ax.set_ylabel(ETIQUETA_Y_REG)
    ax.legend(fontsize=8)

plt.suptitle("Comparación de ajuste KNN — distintos n_neighbors",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# ── 7. Modelo elegido — métricas finales ───────────────────────────────────────
modelo_reg = KNeighborsRegressor(n_neighbors=K_ELEGIDO)
modelo_reg.fit(X_train, y_train)
y_pred_reg   = modelo_reg.predict(X_test)
y_curva_reg  = modelo_reg.predict(X_plot)

r2_final  = r2_score(y_test, y_pred_reg)
mse_final = mean_squared_error(y_test, y_pred_reg)

print(f"\n  Modelo seleccionado: KNN  k = {K_ELEGIDO}")
print(f"  R²  = {_G}{r2_final:.4f}{_R}")
print(f"  MSE = {_Y}{mse_final:.4f}{_R}")

# ── 8. Gráfica final ──────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(X_reg, y_reg, s=15, alpha=0.4,
            color="steelblue", label="Datos experimentales")
plt.plot(X_plot, y_curva_reg, color="crimson",
         linewidth=2.5, label=f"KNN  k = {K_ELEGIDO}")
plt.xlabel(ETIQUETA_X)
plt.ylabel(ETIQUETA_Y_REG)
plt.title(f"Regresión KNN  (k = {K_ELEGIDO})  |  R² = {r2_final:.4f}")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
#  PARTE 2 — CLASIFICACIÓN LOGÍSTICA
# ==============================================================================
print(f"\n{_B}{'═'*60}")
print("  PARTE 2 — CLASIFICACIÓN LOGÍSTICA")
print(f"{'═'*60}{_R}")

# ── 1–3. Variables y división ─────────────────────────────────────────────────
X_clf = df[[COL_ENTRADA]].values
y_clf = df[COL_CLASIFICACION].values.astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\n  Entrenamiento: {len(X_train_c)} muestras  |  Prueba: {len(X_test_c)} muestras")
print(f"  Distribución de clases (train): {np.bincount(y_train_c)}")

# ── 4. Entrenamiento ──────────────────────────────────────────────────────────
modelo_clf = LogisticRegression(C=1e4, solver="lbfgs")
modelo_clf.fit(X_train_c, y_train_c)

y_pred_c = modelo_clf.predict(X_test_c)
y_prob_c = modelo_clf.predict_proba(X_test_c)[:, 1]

# ── 5. Métricas ───────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test_c, y_pred_c)
loss = log_loss(y_test_c, y_prob_c)

print(f"\n  {'Métrica':<18} {'Valor':>10}")
print(f"  {'─'*30}")
print(f"  {'Accuracy':<18} {_G}{acc:>10.4f}{_R}")
print(f"  {'Log Loss':<18} {_Y}{loss:>10.4f}{_R}")

# ── Matriz de confusión ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test_c, y_pred_c)
vn, fp, fn, vp = cm.ravel()
print(f"\n  Verdaderos negativos : {vn}")
print(f"  Falsos positivos     : {fp}")
print(f"  Falsos negativos     : {fn}")
print(f"  Verdaderos positivos : {vp}")

# ── 6. Visualización 1×3 (datos / probabilidad / matriz confusión) ─────────────
X_plot_c  = np.linspace(X_clf.min(), X_clf.max(), 300).reshape(-1, 1)
prob_plot  = modelo_clf.predict_proba(X_plot_c)[:, 1]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# — Figura 1: datos por clase ——————————————————————————————————————————————————
colores   = {0: "steelblue", 1: "crimson"}
etiquetas = {0: "Clase 0", 1: "Clase 1"}
for clase in [0, 1]:
    mask = y_clf == clase
    ax1.scatter(df[COL_ENTRADA].values[mask],
                y_clf[mask],
                s=15, alpha=0.5,
                color=colores[clase],
                label=etiquetas[clase])
ax1.set_xlabel(ETIQUETA_X)
ax1.set_ylabel(ETIQUETA_Y_CLF)
ax1.set_title("Datos por clase")
ax1.legend()

# — Figura 2: probabilidad de clase 1 ─────────────────────────────────────────
ax2.plot(X_plot_c, prob_plot,
         color="darkorange", linewidth=2.5, label="P(clase 1)")
ax2.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Umbral 0.5")
ax2.set_xlabel(ETIQUETA_X)
ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Probabilidad — Regresión Logística")
ax2.set_ylim(-0.05, 1.05)
ax2.legend()

# — Figura 3: matriz de confusión ─────────────────────────────────────────────
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax3, colorbar=False)
ax3.set_title("Matriz de confusión")

plt.suptitle("Clasificación Logística", fontsize=13)
plt.tight_layout()
plt.show()

print(f"\n{_B}{'─'*60}")
print("  ANÁLISIS COMPLETADO")
print(f"{'─'*60}{_R}")



