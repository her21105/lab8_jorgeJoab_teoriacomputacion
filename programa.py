
import argparse
import math
import time
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Implementaciones -----------------

def problem1(n: int) -> int:
    # Tres bucles: i ~ n/2..n, j ~ 1..(n - n/2), k duplica hasta n → Θ(n^2 log n)
    counter = 0
    i = n // 2
    while i <= n:
        j = 1
        limit_j = n - (n // 2)
        while j <= limit_j:
            k = 1
            while k <= n:
                counter += 1
                k *= 2
            j += 1
        i += 1
    return counter

def problem2(n: int) -> int:
    # Doble bucle con break en el interno → Θ(n)
    if n <= 1:
        return 0
    counter = 0
    i = 1
    while i <= n:
        j = 1
        while j <= n:
            counter += 1  # printf once
            break
        i += 1
    return counter

def problem3(n: int) -> int:
    # Bucle externo ~ n/3 y bucle interno con paso 4 ~ n/4 → Θ(n^2)
    counter = 0
    i = 1
    while i <= n // 3:
        j = 1
        while j <= n:
            counter += 1
            j += 4
        i += 1
    return counter

# ----------------- Utilidades de medición -----------------

def time_function(func, n: int) -> float:
    start = time.perf_counter()
    func(n)
    end = time.perf_counter()
    return end - start

def measure_series(func, ns: List[int], max_runtime_s: float = 30.0) -> pd.DataFrame:
    times, measured_ns = [], []
    spent = 0.0
    for n in ns:
        if spent > max_runtime_s:
            break
        t = time_function(func, n)
        times.append(t)
        measured_ns.append(n)
        spent += t
    return pd.DataFrame({"n": measured_ns, "time_s": times})

def fit_and_predict(df_meas: pd.DataFrame, ns_all: List[int], model: str) -> pd.DataFrame:
    """
    Ajusta un modelo lineal en base:
      - 'n'         → a*n + b
      - 'n2'        → a*n^2 + b
      - 'n2logn'    → a*n^2*log2(n) + b
    y predice para ns_all.
    """
    if df_meas.empty:
        raise ValueError("No hay datos medidos para ajustar el modelo")

    n = df_meas["n"].values.astype(float)
    t = df_meas["time_s"].values.astype(float)

    if model == "n":
        X = np.vstack([n, np.ones_like(n)]).T
    elif model == "n2":
        X = np.vstack([n**2, np.ones_like(n)]).T
    elif model == "n2logn":
        X = np.vstack([n**2 * np.log2(np.maximum(n, 2)), np.ones_like(n)]).T
    else:
        raise ValueError("Modelo desconocido")

    coef, _, _, _ = np.linalg.lstsq(X, t, rcond=None)
    a, b = coef[0], coef[1]

    def predict(nv: float) -> float:
        if model == "n":
            return a * nv + b
        if model == "n2":
            return a * (nv ** 2) + b
        return a * (nv ** 2 * math.log2(max(nv, 2))) + b

    preds = [predict(float(nv)) for nv in ns_all]
    return pd.DataFrame({"n": ns_all, "pred_time_s": preds})

def build_result_table(problem_name: str, df_meas: pd.DataFrame, est_df: pd.DataFrame, target_ns: List[int]) -> pd.DataFrame:
    df = pd.DataFrame({"n": target_ns})
    df = df.merge(est_df.rename(columns={"pred_time_s": "time_s"}), on="n", how="left")
    df["source"] = "estimated"
    if not df_meas.empty:
        m = df_meas.copy()
        m["source"] = "measured"
        df = df.merge(m, on="n", how="left", suffixes=("_est", "_meas"))
        df["time_s"] = df["time_s_meas"].fillna(df["time_s_est"])
        df["source"] = np.where(~df["time_s_meas"].isna(), "measured", "estimated")
        df = df[["n", "time_s", "source"]]
    df.insert(0, "problem", problem_name)
    return df

# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Profiling Problemas 1–3 (Laboratorio 8)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 10, 100, 1000, 10000, 100000, 1000000],
                        help="Tamaños n a reportar (por defecto: 1 10 100 1000 10000 100000 1000000)")
    parser.add_argument("--out", type=str, default="lab8_profiling_results.csv", help="Ruta de salida CSV")
    parser.add_argument("--no-plots", action="store_true", help="No generar PNGs de las gráficas")
    parser.add_argument("--budget", type=float, default=12.0, help="Presupuesto de tiempo por problema (s) para medición real")
    args = parser.parse_args()

    target_ns = args.sizes

    # Problema 1
    p1_measure_ns = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
    p1_df_meas = measure_series(problem1, p1_measure_ns, max_runtime_s=args.budget)
    p1_est_df = fit_and_predict(p1_df_meas, target_ns, model="n2logn")
    p1_table = build_result_table("Problem 1", p1_df_meas, p1_est_df, target_ns)

    # Problema 2
    p2_measure_ns = target_ns  # factible medirlos todos
    p2_df_meas = measure_series(problem2, p2_measure_ns, max_runtime_s=args.budget)
    p2_est_df = fit_and_predict(p2_df_meas, target_ns, model="n")
    p2_table = build_result_table("Problem 2", p2_df_meas, p2_est_df, target_ns)

    # Problema 3
    p3_measure_ns = [1, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
    p3_df_meas = measure_series(problem3, p3_measure_ns, max_runtime_s=args.budget)
    p3_est_df = fit_and_predict(p3_df_meas, target_ns, model="n2")
    p3_table = build_result_table("Problem 3", p3_df_meas, p3_est_df, target_ns)

    all_tables = pd.concat([p1_table, p2_table, p3_table], ignore_index=True)
    all_tables.to_csv(args.out, index=False)
    print(f"[OK] Resultados guardados en: {args.out}")
    print(all_tables)

    if not args.no_plots:
        # Una figura por problema, sin estilos ni colores personalizados
        for name in ["Problem 1", "Problem 2", "Problem 3"]:
            subset = all_tables[all_tables["problem"] == name]
            plt.figure()
            plt.plot(subset["n"], subset["time_s"], marker="o")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("n")
            plt.ylabel("tiempo (s)")
            plt.title(f"{name}: tamaño vs tiempo (log-log)")
            png = f"{name.lower().replace(' ', '_')}.png"
            plt.savefig(png, dpi=140, bbox_inches="tight")
            plt.close()
            print(f"[OK] Gráfica guardada: {png}")

if __name__ == "__main__":
    main()
