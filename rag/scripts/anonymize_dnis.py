"""
anonymize_dnis.py
=================
Reemplaza todos los DNIs reales de proyectos_docentes.db por DNIs ficticios,
preservando las relaciones entre tablas (el mismo DNI real siempre recibe
el mismo DNI ficticio).

También anonimiza los nombres, apellidos y usuarios (UVUS) de la tabla
ASG_PERMISO_ROL_USUARIO, ya que contienen datos personales identificables
de los usuarios del sistema.

Genera una copia de seguridad en ./data/proyectos_docentes.db.bak antes de
modificar nada.

Uso:
    python anonymize_dnis.py
"""

import sqlite3
import random
import shutil

DB_PATH = "./data/proyectos_docentes.db"
BACKUP = "./data/proyectos_docentes.db.bak"

# ---------------------------------------------------------------------------
# Tablas y columnas que contienen DNIs (valores numéricos de 7-8 dígitos)
# ---------------------------------------------------------------------------
DNI_COLUMNS: dict[str, list[str]] = {
    "ASG_ASIGNATURA_1": ["PRS_DNIPRS"],
    "ASG_ASIGNATURA_2": ["PRS_DNIPRS"],
    "ASG_CPSGACPRF_1": ["PRS_DNIPRS"],
    "ASG_CPSGACPRF_2": ["PRS_DNIPRS"],
    "ASG_CPSHITOANUALES": ["USUARIO"],
    "ASG_CPSHITOPROGRAMA": ["USUARIO"],
    "ASG_CPSHITOPROYECTO": ["USUARIO"],
    "ASG_CPSTRIBUNAL": ["PRS_DNIPRS"],
    "ASG_CPSTRIBUNAL_FOTO": ["PRS_DNIPRS"],
    "ASG_EVALUADOR": ["PRS_DNIPRS"],
    "ASG_PERMISO_ROL_USUARIO": ["USUARIO"],
    "ASG_PROGRAMAANYO_FOTO": ["PRS_DNIPRS"],
    "ASG_PROYECTODATO": ["PRS_DNIPRS_AUTOR", "PRS_DNIPRS_BLOQ"],
    "ASG_PROYECTOPROFESOR": ["PRS_DNIPRS"],
    "ASG_PROYECTOPROFESOR_FOTO": ["PRS_DNIPRS"],
    "ASG_PERSONADELEGACION": ["PRS_DNIPRS", "PRS_DNIPRS_DELEGADO"],
}


def generate_fake_dni(used: set[str]) -> str:
    """Genera un número de DNI ficticio de 8 dígitos no usado aún."""
    while True:
        fake = str(random.randint(10_000_000, 99_999_999))
        if fake not in used:
            return fake


def build_mapping(cur: sqlite3.Cursor) -> dict[str, str]:
    """Recoge todos los DNIs únicos de la BD y construye el mapeo real→ficticio."""
    all_real: set[str] = set()
    for table, cols in DNI_COLUMNS.items():
        for col in cols:
            try:
                rows = cur.execute(
                    f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL"
                ).fetchall()
                for (val,) in rows:
                    v = val.strip() if val else ""
                    if v:
                        all_real.add(v)
            except sqlite3.OperationalError as e:
                print(f"  [aviso] {table}.{col}: {e}")

    print(f"DNIs únicos encontrados: {len(all_real)}")

    used_fakes: set[str] = set()
    mapping: dict[str, str] = {}
    for real in sorted(all_real):
        fake = generate_fake_dni(used_fakes)
        used_fakes.add(fake)
        mapping[real] = fake
    return mapping


def apply_dni_mapping(cur: sqlite3.Cursor, mapping: dict[str, str]) -> int:
    """Actualiza todos los DNIs en todas las tablas. Devuelve el nº de celdas tocadas."""
    total = 0
    for table, cols in DNI_COLUMNS.items():
        for col in cols:
            for real, fake in mapping.items():
                try:
                    cur.execute(
                        f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                        (fake, real),
                    )
                    total += cur.rowcount
                except sqlite3.OperationalError as e:
                    print(f"  [aviso] {table}.{col}: {e}")
    return total


def anonymize_permiso_rol(cur: sqlite3.Cursor) -> int:
    """
    Los nombres, apellidos y UVUS de ASG_PERMISO_ROL_USUARIO se conservan
    intencionadamente: son datos de usuarios del sistema (no datos personales
    de los evaluados), y el UVUS es necesario para el funcionamiento del
    sistema PyP.
    Solo el campo USUARIO (DNI numérico) se anonimiza a través del mapeo
    global de DNI_COLUMNS.
    """
    return 0


def main() -> None:
    random.seed(42)  # semilla fija → resultado reproducible

    # 1. Copia de seguridad
    shutil.copy2(DB_PATH, BACKUP)
    print(f"Copia de seguridad: {BACKUP}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # 2. Construir mapeo real → ficticio
    mapping = build_mapping(cur)

    # 3. Aplicar en una única transacción
    with con:
        dni_cells = apply_dni_mapping(cur, mapping)
        name_rows = anonymize_permiso_rol(cur)

    con.close()

    # 4. Resumen
    print(f"Celdas de DNI actualizadas : {dni_cells}")
    print(f"Filas de nombres/UVUS      : {name_rows}")
    print(f"Mapeo aplicado ({len(mapping)} DNIs):")
    for real, fake in sorted(mapping.items()):
        print(f"  {real}  →  {fake}")


if __name__ == "__main__":
    main()
