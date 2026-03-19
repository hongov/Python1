# -------------------

# Librerías esenciales

# -------------------

import pandas as pd  # para manejo de datos
import numpy as np   # para operaciones numéricas
import matplotlib.pyplot as plt  # para gráficos

# -------------------

# Lectura de archivos

# -------------------

df = pd.read_csv("archivo.csv")  # Leer CSV
df_excel = pd.read_excel("archivo.xlsx")  # Leer Excel
df_txt = pd.read_table("archivo.txt", sep="\t")  # Leer TXT

# -------------------

# Exploración de datos

# -------------------

df.head()           # Primeras 5 filas
df.info()           # Información general del DataFrame
df.describe()       # Estadísticas básicas
df.columns          # Nombres de columnas
df.dtypes           # Tipos de columnas
df.isnull().sum()   # Conteo de valores nulos

# -------------------

# Limpieza de datos

# -------------------

df = df.dropna()  # Eliminar filas con valores nulos
df["columna"] = df["columna"].fillna(df["columna"].mean())  # Reemplazar nulos por la media
df["columna"] = df["columna"].astype(float)  # Convertir a tipo float
df["columna"] = pd.to_numeric(df["columna"], errors="coerce")  # Convertir a numérico seguro
df["fecha"] = pd.to_datetime(df["fecha"], format="%y/%m/%d")  # Convertir a datetime
df = df.drop_duplicates()  # Eliminar filas duplicadas
df = df.dropna(subset=["col1", "col2", "col3"])  # Eliminar filas con NaN en columnas específicas

# -------------------

# Series y DataFrame

# -------------------

serie = pd.Series(df["columna"])  # Crear Series
serie.mean()       # Media
serie.median()     # Mediana
serie.std()        # Desviación estándar

# Crear DataFrame de ejemplo

df_col = pd.DataFrame({"col1": [1,2,3], "col2": [4,5,6]})
df_col["col2"].mean()  # Media de columna

# -------------------

# Ordenar y agrupar

# -------------------

df_sorted = df.sort_values("columna")  # Orden ascendente
df_sorted_desc = df.sort_values("columna", ascending=False)  # Orden descendente
df_grouped_mean = df.groupby("categoria")["columna"].mean()  # Promedio por grupo
df_grouped_sum  = df.groupby("categoria")["columna"].sum()   # Suma por grupo
df_grouped_std  = df.groupby("categoria")["columna"].std()   # Desviación estándar por grupo

# -------------------

# Ejemplos rápidos

# -------------------

df_example = pd.DataFrame({"energia_eV": [5.2, 3.1, 7.8, 4.0], "intensidad": [10, 25, 5, 18]})
df_example.sort_values("energia_eV", ascending=False)  # Ordenar por energía

# Agrupar por categoría

df2 = pd.DataFrame({"estacion": ["Norte", "Norte", "Sur", "Sur"], "concentracion": [12, 15, 20, 18]})
df2.groupby("estacion")["concentracion"].mean()  # Promedio por estación
df2.groupby("estacion")["concentracion"].sum()   # Suma por estación

# -------------------

# NumPy: operaciones básicas

# -------------------

x = np.array([1,2,3])  # Crear arreglo
y = x**2              # Operación vectorizada
np.mean(x)             # Media
np.median(x)           # Mediana
np.std(x, ddof=1)      # Desviación estándar muestral
np.min(x)              # Valor mínimo
np.max(x)              # Valor máximo
np.where(x>1)          # Índices donde se cumple condición
np.argmax(x)           # Índice del valor máximo
np.argmin(x)           # Índice del valor mínimo

# -------------------

# Matplotlib: gráficos básicos

# -------------------

plt.figure(figsize=(10,4))  # Crear figura
plt.plot(x, y, label="datos")  # Graficar datos
plt.title("Título")        # Título del gráfico
plt.xlabel("Eje X")       # Etiqueta eje X
plt.ylabel("Eje Y")       # Etiqueta eje Y
plt.legend()                # Mostrar leyenda
plt.grid(True)              # Activar cuadrícula
plt.savefig("grafico.png", dpi=300)  # Guardar figura
plt.show()                  # Mostrar figura

# -------------------

# Matplotlib: interfaz orientada a objetos

# -------------------

fig, axes = plt.subplots(1, 2, figsize=(12,4))  # Crear figura con 2 subgráficos
axes[0].hist(df2["concentracion"], bins=15, label="Cs-137")  # Histograma
axes[0].set_title("Histograma Cs-137")
axes[0].set_xlabel("Concentración")
axes[0].set_ylabel("Frecuencia")
axes[0].grid(True)
axes[0].legend()

axes[1].scatter(df2["estacion"], df2["concentracion"])  # Dispersión
axes[1].set_title("Dispersión")
axes[1].set_xlabel("Estación")
axes[1].set_ylabel("Concentración")
axes[1].grid(True)
plt.tight_layout()
plt.show()

# =========================================
# Librerías esenciales
# =========================================
import pandas as pd          # manejo de datos tipo DataFrame
import numpy as np           # operaciones numéricas y vectorizadas
import matplotlib.pyplot as plt  # gráficos
from scipy.constants import R  # constante universal de gases

# =========================================
# FUNCIONES EN PYTHON
# =========================================

# Función para calcular posición y velocidad en movimiento rectilíneo uniformemente acelerado
def estado_movimiento(x0, v0, a, t):
    """
    Calcula la posición y velocidad en MRUA.
    
    Parámetros:
    x0: posición inicial
    v0: velocidad inicial
    a: aceleración
    t: tiempo
    
    Retorna:
    x: posición
    v: velocidad
    """
    x = x0 + v0 * t + 0.5 * a * t**2
    v = v0 + a * t
    return x, v

# Función de energía cinética
def energia_cinetica(m, v):
    """
    Calcula la energía cinética.
    
    Parámetros:
    m: masa
    v: velocidad
    
    Retorna:
    Energía cinética en J
    """
    return 0.5 * m * v**2

# Función lambda equivalente
energia_cinetica_lambda = lambda m, v: 0.5 * m * v**2

# Función de presión de gas ideal
def presion_gas(n, T, V, R=8.314):
    """
    Calcula presión de un gas ideal.
    
    n: moles
    T: temperatura en K
    V: volumen en m^3
    R: constante de gas (por defecto 8.314 J/mol·K)
    """
    P = n * R * T / V
    return P

# =========================================
# EJEMPLO 1: Masa de aire en recipiente
# =========================================
def solicitar_datos():
    P = float(input("Presión en Pa: "))
    V = float(input("Volumen en m³: "))
    T = float(input("Temperatura en K: "))
    return P, V, T

def calcular_moles(P, V, T):
    n_moles = P*V/(R*T)
    return n_moles

def calcular_masa(n_moles):
    MM = 28.9647  # g/mol
    return MM * n_moles

def mostrar_resultados(P, V, T, n_moles, masa):
    print(f"Presión: {P} Pa | Volumen: {V} m³ | Temperatura: {T} K")
    print(f"Moles: {n_moles:.4f} mol | Masa del aire: {masa:.4f} g")

# main
def main_masa():
    P, V, T = solicitar_datos()
    n = calcular_moles(P, V, T)
    m = calcular_masa(n)
    mostrar_resultados(P, V, T, n, m)

# =========================================
# EJEMPLO 2: Alquiler diario de equipo
# =========================================
def solicitud_precio():
    return float(input("Precio del equipo (₡): "))

def calcular_alquiler(precio):
    diario = precio/45
    diario_ganancia = diario*1.3
    return diario, diario_ganancia

def mostrar_alquiler(precio, diario, diario_ganancia):
    print(f"Precio: ₡{precio}")
    print(f"Alquiler para recuperar inversión: ₡{diario:.2f}")
    print(f"Alquiler con 30% ganancia: ₡{diario_ganancia:.2f}")

def main_alquiler():
    precio = solicitud_precio()
    diario, diario_ganancia = calcular_alquiler(precio)
    mostrar_alquiler(precio, diario, diario_ganancia)

# =========================================
# EJEMPLO 3: Iteraciones con listas y NumPy
# =========================================
def movimiento_particula(x0, v0, a, dt, N):
    # listas
    x_list = [x0]
    v_list = [v0]
    for i in range(N):
        xn = x_list[-1]
        vn = v_list[-1]
        x_list.append(xn + vn*dt)
        v_list.append(vn + a*dt)
    # NumPy preasignado
    x_np = np.zeros(N+1)
    v_np = np.zeros(N+1)
    x_np[0] = x0
    v_np[0] = v0
    for n in range(N):
        x_np[n+1] = x_np[n] + v_np[n]*dt
        v_np[n+1] = v_np[n] + a*dt
    return x_list, v_list, x_np, v_np

# =========================================
# EJEMPLO 4: Iteración de punto fijo
# =========================================
def punto_fijo(g, x0, maxit=60, tol=1e-10):
    x_vals = [x0]
    e_vals = [0]
    r_vals = [0]
    x = x0
    for i in range(maxit):
        x_new = g(x)
        e = abs(x_new - x)
        er = e/abs(x_new + 1e-15)  # evita división por cero
        x_vals.append(x_new)
        e_vals.append(e)
        r_vals.append(er)
        if er < tol:
            break
        x = x_new
    return x_vals, e_vals, r_vals, i+1

# =========================================
# EJEMPLO 5: Condiciones booleanas
# =========================================
def par_o_impar():
    n = int(input("Número de oscilaciones: "))
    if n % 2 == 0:
        print("La cantidad de oscilaciones es par")
    else:
        print("La cantidad de oscilaciones es impar")

# =========================================
# EJEMPLO 6: Manejo de archivos con NumPy
# =========================================
def guardar_array(nombre, arr):
    np.savetxt(nombre, arr, delimiter=",", fmt="%.6f")

def cargar_array(nombre):
    return np.loadtxt(nombre, delimiter=",")

# =========================================
# EJEMPLO 7: Visualización con Matplotlib
# =========================================
def graficar_posicion(t, x_list, x_np):
    plt.figure(figsize=(8,5))
    plt.plot(t, x_list, label="Lista")
    plt.plot(t, x_np, "--", label="NumPy")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Posición (m)")
    plt.title("Posición vs Tiempo")
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================================
# EJEMPLOS de Pandas: ordenar y agrupar
# =========================================
def ejemplo_orden_grupo():
    df = pd.DataFrame({
        "energia_eV":[5.2,3.1,7.8,4.0],
        "intensidad":[10,25,5,18]
    })
    df.sort_values("energia_eV", ascending=False)
    df2 = pd.DataFrame({
        "estacion":["Norte","Norte","Sur","Sur"],
        "concentracion":[12,15,20,18]
    })
    print(df2.groupby("estacion")["concentracion"].mean())
    print(df2.groupby("estacion")["concentracion"].sum())

def E():
    Et = float(input("Ingrese la energía total del sistema en Joules"))
    # Tipos de energía
    tipos = np.array(["Cinética ", "Potencial", "Térmica  "])

    # Porcentajes de cada tipo de energía
    porcentajes = np.array([0.40, 0.30, 0.30])

    # Cálculo de energías usando NumPy
    energias = Et * porcentajes

    print("\nDistribución de Energía del Sistema")
    print("-" * 45)
    print(f"{'Tipo de energía     '}{'Porcentaje      '}{'Energía (J)      '}")
    print("-" * 45)

    for i in range(len(tipos)):
        print(f"{tipos[i]}             {porcentajes[i]*100}%         {energias[i]}")

    print("-" * 45)



E()



def bytes():
    pot = range(31)

    for i in (pot):
        val = 2**i
        if i//10 == 0:
            print(f"2^{i} = {val} bytes = {val:.3g} bytes")
        elif i//10 == 1:
            print(f"2^{i} = {val} bytes = {(val/1000):.3g} kilobytes")
        elif i//10 == 2:
            print(f"2^{i} = {val} bytes = {(val/1000000):.3g} megabytes")
        elif i//10 == 3:
            print(f"2^{i} = {val} bytes = {(val/1000000000):.3g} gigabytes")
        else:
            print(f"2^{i} = {val} bytes = {(val/1000000000000):.1g} terabytes")
        



bytes()



M = 5.972*(10**24) #[kg]
r = 6.771 *(10**6) #[m]


v0 = 7500.0
tol = 1e-12
maxit = 200

#def g(v):
 #   return np.sqrt((G*M/r) * (1/np.sqrt(1 + (v/c)**2)))


v = v0
hist = [v]

for n in range(maxit):

    gv = np.sqrt((G*M/r) * (1/np.sqrt(1 + (v/c)**2)))
    
    v_new = gv
    hist.append(v_new)

    err = abs(v_new - v)/(abs(v) + 1e-15)

    if err < tol:
        print(f"Convergió en {n+1} iteraciones")
        break

    v = v_new

print("Velocidad orbital aproximada:", v_new, "m/s")

hist = np.array(hist)

print(hist)

plt.plot(hist, marker='o')
plt.xlabel("Iteración n")
plt.ylabel("v_n (m/s)")
plt.title("Convergencia del método de punto fijo")
plt.grid()
plt.show()




alpha = 0.6     # coeficiente de absorción
S = 2.0         # fuente constante
I0 = 50.0       # intensidad inicial en z=0

z0 = 0.0
z_end = 10.0
h = 0.5

# malla
z = np.arange(z0, z_end + h, h)


I = np.zeros(len(z))

# Condición inicial
I[0] = I0

# Método de Euler explícito
for i in range(len(z) - 1):
    I[i+1] = I[i] + h * (-alpha * I[i] + S)

print(f" z  --   I")
print("="*15)
# Imprimir resultados
for i in range(len(z)):
    print(f"{z[i]} -- {I[i]:.5}")






for i in range(1, 101):
    print(f"{i:3}", end=" ")
    
    # Salto de línea cada 10 números
    if i % 10 == 0:
        print()

print("\nFin del experimento")







def pit():
    num = range(100)
    #print(num)
    a = float(input("Escriba el valor de a:"))
    b = float(input("Escriba el valor de b:"))
    c = float(input("Escriba el valor de c:"))

    if a**2 + (b**2) == (c**2):
        print("Sí cumple")
    elif a**2 + (c**2) == (b**2):
        print("Sí cumple")
    elif c**2 + (b**2) == (a**2):
        print("Sí cumple")
    else:
        print("No cumple")

    contador = 0
    for a in range(1, 100):
        for b in range(a, 100):
            for c in range(b, 100):
                if a**2 + (b**2) == c**2:
                    print(a, b, c)
                    contador += 1
    print(f"son {contador}")

    
pit()



def buscar_palabra():
    # Solicitar datos
    palabra = input("Ingrese la palabra clave: ").lower()
    archivo = input("Ingrese el nombre del archivo: ")

    try:
        # Abrir y leer archivo
        with open(archivo, "r", encoding="utf-8") as f:
            texto = f.read()

        # Convertir texto a minúsculas
        texto = texto.lower()

        # Convertir a arreglo de palabras con NumPy
        palabras = np.array(texto.split())

        # Buscar la palabra
        if np.any(palabras == palabra):
            print("La palabra SÍ se encuentra en el archivo.")
        else:
            print("La palabra NO se encuentra en el archivo.")

    except FileNotFoundError:
        print("Error: el archivo no existe.")

buscar_palabra()




