import os
import random
import re
import zlib
import base64
import time
import requests
import json
from sklearn.model_selection import train_test_split

# ========
# DATOS BASE
# ========
sistemas = [
    "Sistema de Gestión de Reservas",
    "Plataforma E-learning",
    "Sistema Bancario",
    "Gestión Hospitalaria",
    "Sistema de Inventario",
    "Gestión de Recursos Humanos",
    "Sistema de Ventas Online"
]

actores_base = [
    "Cliente", "Administrador", "Supervisor", "Analista", "Usuario", "Empleado", "Gerente"
]

casos_uso_base = [
    "Gestionar perfil", "Registrar usuario", "Autenticar usuario",
    "Realizar operación", "Consultar datos", "Generar reporte",
    "Configurar sistema", "Auditar actividad", "Actualizar perfil",
    "Procesar pago", "Emitir factura", "Reservar cita"
]

# ========================
# LIMPIEZA DE TEXTO
# ========================
def limpiar(s):
    return re.sub(r"[^\w\s\-\.]", "", s)

# ========================
# GENERAR DIAGRAMA
# ========================
def generar_diagrama(sistema, actores, casos):
    sistema = limpiar(sistema)
    d = f"""@startuml
title {sistema}
left to right direction
skinparam packageStyle rectangle
skinparam usecase {{
  BackgroundColor #EEF7FF
  BorderColor #1A4A7A
}}
skinparam actor {{
  BackgroundColor White
  BorderColor Black
}}
"""

    # Actores
    actor_alias = {}
    for i, a in enumerate(actores):
        alias = f"A{i}"
        actor_alias[a] = alias
        d += f'actor "{limpiar(a)}" as {alias}\n'
    d += "\n"

    # Casos de uso agrupados
    caso_alias = {}
    random.shuffle(casos)
    g1 = casos[:len(casos)//2]
    g2 = casos[len(casos)//2:]

    d += f'package "Módulo Principal" {{\n'
    for i, c in enumerate(g1):
        alias = f"U{i}"
        caso_alias[c] = alias
        d += f' usecase "{limpiar(c)}" as {alias}\n'
    d += "}\n\n"

    d += f'package "Módulo Secundario" {{\n'
    for j, c in enumerate(g2, start=len(g1)):
        alias = f"U{j}"
        caso_alias[c] = alias
        d += f' usecase "{limpiar(c)}" as {alias}\n'
    d += "}\n\n"

    # Relaciones principales
    relaciones = set()
    for a in actores:
        cu = random.choice(casos)
        aliasA = actor_alias[a]
        aliasU = caso_alias[cu]
        d += f"{aliasA} --> {aliasU}\n"
        relaciones.add((aliasA, aliasU))
    d += "\n"

    # Relaciones include / extend
    if len(casos) >= 4:
        base = casos[0]
        incl = casos[1]
        d += f"{caso_alias[base]} ..> {caso_alias[incl]} : <<include>>\n"
        ext = casos[2]
        d += f"{caso_alias[base]} ..> {caso_alias[ext]} : <<extend>>\n"

    d += "\n@enduml"
    return d, actor_alias, caso_alias, relaciones

# ========================
# DESCRIPCIÓN DEL FLUJO
# ========================
def describir_flujo(sistema, actores, casos, actor_alias, caso_alias, relaciones):
    flujo = {
        "sistema": sistema,
        "actores": [],
        "casos_uso": [],
        "relaciones": []
    }

    for a in actores:
        flujo["actores"].append({"nombre": a, "alias": actor_alias[a]})

    for c in casos:
        flujo["casos_uso"].append({"nombre": c, "alias": caso_alias[c]})

    for (a, u) in relaciones:
        flujo["relaciones"].append({
            "actor": a,
            "caso_uso": u,
            "tipo": "principal"
        })

    if len(casos) >= 4:
        flujo["relaciones"].append({
            "origen": caso_alias[casos[0]],
            "destino": caso_alias[casos[1]],
            "tipo": "include"
        })
        flujo["relaciones"].append({
            "origen": caso_alias[casos[0]],
            "destino": caso_alias[casos[2]],
            "tipo": "extend"
        })

    return flujo

# ========================
# CODIFICADOR PLANTUML
# ========================
def plantuml_encode(text):
    compressed = zlib.compress(text.encode("utf-8"))
    encoded = base64.b64encode(compressed).decode("ascii")
    pu_alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    return encoded.translate(str.maketrans(b64, pu_alphabet))

# ========================
# GENERAR DATASET
# ========================
def generar_dataset(n=100):
    os.makedirs("dataset", exist_ok=True)
    diagramas = []

    for i in range(n):
        sistema = random.choice(sistemas)
        actores = random.sample(actores_base, 3)
        casos = random.sample(casos_uso_base, 6)

        codigo, actor_alias, caso_alias, relaciones = generar_diagrama(sistema, actores, casos)
        flujo = describir_flujo(sistema, actores, casos, actor_alias, caso_alias, relaciones)

        # Guardar JSON
        json_path = f"dataset/diagrama_{i+1:04d}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(flujo, f, ensure_ascii=False, indent=2)

        # Guardar imagen
        encoded = plantuml_encode(codigo)
        url = f"https://www.plantuml.com/plantuml/png/~1{encoded}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.content.startswith(b"\x89PNG"):
            with open(f"dataset/diagrama_{i+1:04d}.png", "wb") as f:
                f.write(r.content)
        diagramas.append(f"diagrama_{i+1:04d}")

        time.sleep(10)  # evitar saturar el servidor

    return diagramas

# ========================
# DIVIDIR EN TRAIN/VAL/TEST
# ========================
def dividir_dataset(diagramas):
    train, test = train_test_split(diagramas, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    for split in ["train", "val", "test"]:
        os.makedirs(f"dataset/{split}", exist_ok=True)

    for split, lista in [("train", train), ("val", val), ("test", test)]:
        for name in lista:
            for ext in [".png", ".json"]:
                os.rename(f"dataset/{name}{ext}", f"dataset/{split}/{name}{ext}")

# ========================
# EJECUCIÓN
# ========================
if __name__ == "__main__":
    diagramas = generar_dataset(n=100)  # genera 500 pares imagen+json
    dividir_dataset(diagramas)
    print("✅ Dataset generado y dividido en train/val/test")
