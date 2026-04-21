"""
Ejercicio 1 – Modelos de Regresión y Clasificación
IF4702 – Física Computacional I

En cada sección, completar:
    pd.read_csv("...")         <- nombre del archivo
    X = df[["..."]].values    <- columna(s) de entrada
    y = df["..."].values      <- columna objetivo
"""


# ═════════════════════════════════════════════════════════════════════════════
# REG 01 – Regresión Lineal
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print(f"  Pendiente  : {modelo.coef_[0]:.4f}")
print(f"  Intercepto : {modelo.intercept_:.4f}")
print(f"  R2  = {r2_score(y_test, y_pred):.4f}")
print(f"  MSE = {mean_squared_error(y_test, y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
ax.plot(x_rng, modelo.predict(x_rng), color="crimson", lw=2,
        label=f"Lineal  (R2={r2_score(y_test, y_pred):.3f})")
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("Regresión Lineal")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 02 – Regresión Polinomial
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_grado = -np.inf, None, None
for g in [2, 3, 4]:
    pipe = Pipeline([("poly", PolynomialFeatures(degree=g, include_bias=False)),
                     ("lr",   LinearRegression())])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"  Grado {g}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, pipe.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_grado = r2, pipe, g

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor grado: {mejor_grado} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
ax.plot(x_rng, mejor_modelo.predict(x_rng), color="darkorange", lw=2,
        label=f"Polinomio grado {mejor_grado}  (R2={mejorR2:.3f})")
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title(f"Regresión Polinomial (grado {mejor_grado})")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 03 – Árbol de Decisión
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_prof = -np.inf, None, None
for d in [2, 4, 6, 8]:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    print(f"  depth={d}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, m.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_prof = r2, m, d

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor depth: {mejor_prof} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
axes[0].plot(x_rng, mejor_modelo.predict(x_rng), color="forestgreen", lw=2,
             label=f"Árbol depth={mejor_prof}  (R2={mejorR2:.3f})")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y"); axes[0].set_title("Árbol de Decisión")
axes[0].legend(); axes[0].grid(True, alpha=0.35)
plot_tree(mejor_modelo, max_depth=3, filled=True, rounded=True, ax=axes[1], fontsize=8)
axes[1].set_title(f"Estructura del árbol (depth={mejor_prof})")
plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 04 – KNN Regressor
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_k = -np.inf, None, None
for k in [1, 3, 5, 7, 10, 15]:
    m = KNeighborsRegressor(n_neighbors=k)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    print(f"  k={k:2d}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, m.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_k = r2, m, k

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor k: {mejor_k} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
ax.plot(x_rng, mejor_modelo.predict(x_rng), color="darkorchid", lw=2,
        label=f"KNN k={mejor_k}  (R2={mejorR2:.3f})")
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title(f"KNN Regressor (k={mejor_k})")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 05 – SVR
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

colores_k = {"linear": "crimson", "poly": "darkorange", "rbf": "forestgreen"}
mejorR2, mejor_modelo, mejor_kernel = -np.inf, None, None
for k in ["linear", "poly", "rbf"]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svr",    SVR(kernel=k, C=1000, epsilon=0.1))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"  kernel={k:6s}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, pipe.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_kernel = r2, pipe, k

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor kernel: {mejor_kernel} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
for k in ["linear", "poly", "rbf"]:
    pipe_k = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel=k, C=1000, epsilon=0.1))])
    pipe_k.fit(X_train, y_train)
    ax.plot(x_rng, pipe_k.predict(x_rng), lw=1.8, color=colores_k[k],
            ls="-" if k == mejor_kernel else "--",
            alpha=1.0 if k == mejor_kernel else 0.5,
            label=f"SVR {k}" + ("  <- mejor" if k == mejor_kernel else ""))
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title("SVR – Comparación de kernels")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 06 – Random Forest Regressor
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_n = -np.inf, None, None
for n in [10, 50, 100, 200]:
    m = RandomForestRegressor(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    print(f"  n={n:3d}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, m.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_n = r2, m, n

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor n: {mejor_n} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng   = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_rng   = mejor_modelo.predict(x_rng)
std_rng = np.array([t.predict(x_rng) for t in mejor_modelo.estimators_]).std(axis=0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
ax.plot(x_rng, y_rng, color="saddlebrown", lw=2, label=f"RF n={mejor_n}  (R2={mejorR2:.3f})")
ax.fill_between(x_rng.ravel(), y_rng - std_rng, y_rng + std_rng,
                alpha=0.2, color="saddlebrown", label="±1σ entre árboles")
ax.set_xlabel("X"); ax.set_ylabel("y"); ax.set_title(f"Random Forest (n={mejor_n})")
ax.legend(); ax.grid(True, alpha=0.35); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 07 – Gradient Boosting Regressor
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_lr = -np.inf, None, None
for lr in [0.01, 0.05, 0.1, 0.2]:
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=lr, max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    r2 = r2_score(y_test, m.predict(X_test))
    print(f"  lr={lr:.2f}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, m.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_lr = r2, m, lr

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor lr: {mejor_lr} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng     = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
train_err = [mean_squared_error(y_train, yp) for yp in mejor_modelo.staged_predict(X_train)]
test_err  = [mean_squared_error(y_test,  yp) for yp in mejor_modelo.staged_predict(X_test)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
axes[0].plot(x_rng, mejor_modelo.predict(x_rng), color="darkgreen", lw=2,
             label=f"GBR lr={mejor_lr}  (R2={mejorR2:.3f})")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y"); axes[0].set_title("Gradient Boosting")
axes[0].legend(); axes[0].grid(True, alpha=0.35)
axes[1].plot(train_err, label="MSE entrenamiento", color="royalblue")
axes[1].plot(test_err,  label="MSE prueba",        color="crimson")
axes[1].set_xlabel("Iteración"); axes[1].set_ylabel("MSE")
axes[1].set_title("Error por iteración"); axes[1].legend(); axes[1].grid(True, alpha=0.35)
plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 08 – MLP Regressor (ReLU)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPRegressor(hidden_layer_sizes=arq, activation="relu",
                                             max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, pipe.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor arq: {mejor_arq} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
axes[0].plot(x_rng, mejor_modelo.predict(x_rng), color="tomato", lw=2,
             label=f"MLP ReLU {mejor_arq}  (R2={mejorR2:.3f})")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y"); axes[0].set_title("MLP ReLU")
axes[0].legend(); axes[0].grid(True, alpha=0.35)
axes[1].plot(mejor_modelo.named_steps["mlp"].loss_curve_, color="tomato", lw=1.8)
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss"); axes[1].set_title("Curva de pérdida")
axes[1].grid(True, alpha=0.35)
plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 09 – MLP Regressor (tanh)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPRegressor(hidden_layer_sizes=arq, activation="tanh",
                                             max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, pipe.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor arq: {mejor_arq} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
axes[0].plot(x_rng, mejor_modelo.predict(x_rng), color="mediumseagreen", lw=2,
             label=f"MLP tanh {mejor_arq}  (R2={mejorR2:.3f})")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y"); axes[0].set_title("MLP tanh")
axes[0].legend(); axes[0].grid(True, alpha=0.35)
axes[1].plot(mejor_modelo.named_steps["mlp"].loss_curve_, color="mediumseagreen", lw=1.8)
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss"); axes[1].set_title("Curva de pérdida")
axes[1].grid(True, alpha=0.35)
plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# REG 10 – MLP Regressor (sigmoid)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorR2, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPRegressor(hidden_layer_sizes=arq, activation="logistic",
                                             max_iter=3000, random_state=42))])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  R2={r2:.4f}  |  MSE={mean_squared_error(y_test, pipe.predict(X_test)):.4f}")
    if r2 > mejorR2:
        mejorR2, mejor_modelo, mejor_arq = r2, pipe, arq

y_pred = mejor_modelo.predict(X_test)
print(f"  → Mejor arq: {mejor_arq} | R2={r2_score(y_test,y_pred):.4f} | MSE={mean_squared_error(y_test,y_pred):.4f}")

x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X, y, color="steelblue", alpha=0.6, s=40, label="Datos experimentales")
axes[0].plot(x_rng, mejor_modelo.predict(x_rng), color="goldenrod", lw=2,
             label=f"MLP sigmoid {mejor_arq}  (R2={mejorR2:.3f})")
axes[0].set_xlabel("X"); axes[0].set_ylabel("y"); axes[0].set_title("MLP sigmoid")
axes[0].legend(); axes[0].grid(True, alpha=0.35)
axes[1].plot(mejor_modelo.named_steps["mlp"].loss_curve_, color="goldenrod", lw=1.8)
axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss"); axes[1].set_title("Curva de pérdida")
axes[1].grid(True, alpha=0.35)
plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLF 01 – Regresión Logística
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)
mejor_modelo = modelo

print(f"  Accuracy      = {accuracy_score(y_test, y_pred):.4f}")
print(f"  Cross-entropy = {log_loss(y_test, y_proba):.4f}")
frontera = -modelo.intercept_[0] / modelo.coef_[0][0]
print(f"  Frontera analítica: X = {frontera:.4f}")

x_rng = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba = modelo.predict_proba(x_rng)[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="purple", lw=2.5, label="P(clase=1|X)")
ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera = {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha"); ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.suptitle("Regresión Logística", fontsize=13); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# CLF 02 – Árbol de Decisión
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_prof = -np.inf, None, None
for d in [2, 4, 6, 8]:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  depth={d}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, m.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_prof = acc, m, d

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor depth: {mejor_prof} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="forestgreen", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"Árbol de Decisión (depth={mejor_prof})", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 03 – KNN Classifier
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_k = -np.inf, None, None
for k in [1, 3, 5, 7, 10, 15]:
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  k={k:2d}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, m.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_k = acc, m, k

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor k: {mejor_k} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="darkorchid", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"KNN Classifier (k={mejor_k})", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 04 – SVM Classifier
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

colores_k = {"linear": "crimson", "poly": "darkorange", "rbf": "forestgreen"}
mejorAcc, mejor_modelo, mejor_kernel = -np.inf, None, None
for k in ["linear", "poly", "rbf"]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svc",    SVC(kernel=k, C=10, probability=True, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  kernel={k:6s}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, pipe.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_kernel = acc, pipe, k

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor kernel: {mejor_kernel} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="steelblue", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"SVM (kernel={mejor_kernel})", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 05 – Random Forest Classifier
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_n = -np.inf, None, None
for n in [10, 50, 100, 200]:
    m = RandomForestClassifier(n_estimators=n, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  n={n:3d}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, m.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_n = acc, m, n

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor n: {mejor_n} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="saddlebrown", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"Random Forest Classifier (n={mejor_n})", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 06 – Gradient Boosting Classifier
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_lr = -np.inf, None, None
for lr in [0.01, 0.05, 0.1, 0.2]:
    m = GradientBoostingClassifier(n_estimators=200, learning_rate=lr, max_depth=3, random_state=42)
    m.fit(X_train, y_train)
    acc = accuracy_score(y_test, m.predict(X_test))
    print(f"  lr={lr:.2f}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, m.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_lr = acc, m, lr

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor lr: {mejor_lr} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="darkgreen", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"Gradient Boosting Classifier (lr={mejor_lr})", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 07 – MLP Classifier (ReLU)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPClassifier(hidden_layer_sizes=arq, activation="relu",
                                              max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, pipe.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor arq: {mejor_arq} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="tomato", lw=2.5, label="P(clase=1|X)  ReLU")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"MLP Classifier ReLU {mejor_arq}", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 08 – MLP Classifier (tanh)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPClassifier(hidden_layer_sizes=arq, activation="tanh",
                                              max_iter=2000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, pipe.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor arq: {mejor_arq} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="mediumseagreen", lw=2.5, label="P(clase=1|X)  tanh")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"MLP Classifier tanh {mejor_arq}", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 09 – MLP Classifier (sigmoid)
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mejorAcc, mejor_modelo, mejor_arq = -np.inf, None, None
for arq in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("mlp",    MLPClassifier(hidden_layer_sizes=arq, activation="logistic",
                                              max_iter=3000, random_state=42))])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"  {str(arq):15s}:  Accuracy={acc:.4f}  |  Cross-entropy={log_loss(y_test, pipe.predict_proba(X_test)):.4f}")
    if acc > mejorAcc:
        mejorAcc, mejor_modelo, mejor_arq = acc, pipe, arq

y_pred  = mejor_modelo.predict(X_test)
y_proba = mejor_modelo.predict_proba(X_test)
print(f"  → Mejor arq: {mejor_arq} | Accuracy={accuracy_score(y_test,y_pred):.4f} | Cross-entropy={log_loss(y_test,y_proba):.4f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="goldenrod", lw=2.5, label="P(clase=1|X)  sigmoid")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle(f"MLP Classifier sigmoid {mejor_arq}", fontsize=13)


# ═════════════════════════════════════════════════════════════════════════════
# CLF 10 – Naive Bayes Gaussiano
# ═════════════════════════════════════════════════════════════════════════════
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("")
X = df[[""]].values
y = df[""].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)
mejor_modelo = modelo

print(f"  Accuracy      = {accuracy_score(y_test, y_pred):.4f}")
print(f"  Cross-entropy = {log_loss(y_test, y_proba):.4f}")
for i, clase in enumerate(modelo.classes_):
    print(f"  Clase {int(clase)}:  mu={modelo.theta_[i, 0]:.4f}  |  sigma2={modelo.var_[i, 0]:.6f}")
x_rng   = np.linspace(X.min() - 0.01, X.max() + 0.01, 400).reshape(-1, 1)
proba   = mejor_modelo.predict_proba(x_rng)[:, 1]
cambios = np.where(np.diff(mejor_modelo.predict(x_rng)))[0]
frontera = x_rng[cambios[0], 0] if len(cambios) > 0 else None

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for clase in np.unique(y):
    m = y.ravel() == clase
    ax1.scatter(X[m], y[m], alpha=0.7, s=45, label=f"Clase {int(clase)}")
ax1.set_xlabel("X"); ax1.set_ylabel("y"); ax1.set_title("Figura 1 – Datos por clase")
ax1.legend(); ax1.grid(True, alpha=0.35)
ax2.plot(x_rng, proba, color="teal", lw=2.5, label="P(clase=1|X)")
if frontera is not None:
    ax2.axvline(frontera, color="black", ls="--", lw=1.5, label=f"Frontera ≈ {frontera:.3f}")
ax2.axhline(0.5, color="gray", ls=":", lw=1)
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba >= 0.5), alpha=0.12, color="orangered")
ax2.fill_between(x_rng.ravel(), proba, 0.5, where=(proba <  0.5), alpha=0.12, color="steelblue")
ax2.set_xlabel("X"); ax2.set_ylabel("Probabilidad predicha")
ax2.set_title("Figura 2 – Probabilidad y frontera")
ax2.set_ylim(-0.05, 1.05); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
plt.tight_layout(); plt.show()
plt.suptitle("Naive Bayes Gaussiano", fontsize=13)
