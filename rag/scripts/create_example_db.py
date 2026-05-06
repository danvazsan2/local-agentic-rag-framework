#!/usr/bin/env python3
"""
create_example_db.py
====================
Genera una base de datos SQLite de ejemplo en ./data/example.db compatible
con el sistema RAG híbrido (SQL + documentos) del proyecto custom-rag.

La base de datos simula un sistema de e-commerce / gestión empresarial con
tablas de: categorías, productos, proveedores, usuarios, empleados,
departamentos, pedidos, líneas de pedido, reseñas, inventario y transacciones.

Volumetría aproximada:
  - 20 categorías
  - 50 proveedores
  - 500 productos
  - 2 000 usuarios
  - 10 departamentos, 150 empleados
  - 8 000 pedidos  →  ~24 000 líneas de pedido
  - 5 000 reseñas
  - 500 registros de inventario
  - 8 000 transacciones

Uso:
    python create_example_db.py          # crea ./data/example.db
    python create_example_db.py --path ./data/mi_bd.db   # ruta personalizada

Visualización rápida:
    - CLI:        sqlite3 ./data/example.db ".tables"
    - GUI:        Extensión VS Code "SQLite Viewer" (alexcvzz.vscode-sqlite)
    - Web local:  datasette ./data/example.db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sqlite3
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# ────────────────────────────────────────────────────────────────────
# Configuración
# ────────────────────────────────────────────────────────────────────
DB_DEFAULT_PATH = os.path.join(".", "data", "example.db")

SEED = 42  # reproducibilidad
random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Datos semilla
# ────────────────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Carlos",
    "María",
    "Lucía",
    "Pedro",
    "Ana",
    "Javier",
    "Elena",
    "David",
    "Laura",
    "Miguel",
    "Sara",
    "Pablo",
    "Carmen",
    "Daniel",
    "Marta",
    "Adrián",
    "Paula",
    "Sergio",
    "Isabel",
    "Jorge",
    "Raquel",
    "Álvaro",
    "Beatriz",
    "Diego",
    "Clara",
    "Hugo",
    "Nuria",
    "Marcos",
    "Andrea",
    "Iván",
    "Sofía",
    "Alejandro",
    "Cristina",
    "Rubén",
    "Patricia",
    "Óscar",
    "Rosa",
    "Fernando",
    "Teresa",
    "Alberto",
    "Irene",
    "Guillermo",
    "Rocío",
    "Víctor",
    "Alicia",
    "Manuel",
    "Sandra",
    "Gonzalo",
    "Inés",
    "Rafael",
]

LAST_NAMES = [
    "García",
    "Rodríguez",
    "Martínez",
    "López",
    "González",
    "Hernández",
    "Pérez",
    "Sánchez",
    "Ramírez",
    "Torres",
    "Flores",
    "Rivera",
    "Gómez",
    "Díaz",
    "Reyes",
    "Morales",
    "Cruz",
    "Ortiz",
    "Gutiérrez",
    "Chávez",
    "Ramos",
    "Vargas",
    "Castillo",
    "Jiménez",
    "Moreno",
    "Romero",
    "Álvarez",
    "Ruiz",
    "Molina",
    "Iglesias",
    "Navarro",
    "Domínguez",
    "Vázquez",
    "Serrano",
    "Blanco",
    "Suárez",
    "Castro",
    "Delgado",
    "Prieto",
    "Cabrera",
]

CITIES = [
    ("Madrid", "Madrid", "28001"),
    ("Barcelona", "Barcelona", "08001"),
    ("Valencia", "Valencia", "46001"),
    ("Sevilla", "Sevilla", "41001"),
    ("Zaragoza", "Zaragoza", "50001"),
    ("Málaga", "Málaga", "29001"),
    ("Bilbao", "Vizcaya", "48001"),
    ("Murcia", "Murcia", "30001"),
    ("Palma", "Baleares", "07001"),
    ("Las Palmas", "Gran Canaria", "35001"),
    ("Valladolid", "Valladolid", "47001"),
    ("Vigo", "Pontevedra", "36201"),
    ("Gijón", "Asturias", "33201"),
    ("Granada", "Granada", "18001"),
    ("Alicante", "Alicante", "03001"),
    ("Córdoba", "Córdoba", "14001"),
    ("Salamanca", "Salamanca", "37001"),
    ("Santander", "Cantabria", "39001"),
    ("Toledo", "Toledo", "45001"),
    ("Badajoz", "Badajoz", "06001"),
]

STREETS = [
    "Calle Mayor",
    "Avenida de la Constitución",
    "Calle del Sol",
    "Paseo de la Castellana",
    "Gran Vía",
    "Ronda de Valencia",
    "Calle de Alcalá",
    "Avenida de Andalucía",
    "Calle Real",
    "Calle San Fernando",
    "Calle Nueva",
    "Avenida de la Libertad",
    "Calle del Comercio",
    "Paseo Marítimo",
    "Calle Santa María",
]

EMAIL_DOMAINS = [
    "gmail.com",
    "outlook.es",
    "yahoo.es",
    "hotmail.com",
    "empresa.com",
    "mail.com",
    "proton.me",
]

CATEGORY_DATA: List[Tuple[str, str]] = [
    ("Electrónica", "Dispositivos electrónicos y gadgets"),
    ("Informática", "Ordenadores, componentes y periféricos"),
    ("Smartphones", "Teléfonos móviles y accesorios"),
    ("Hogar y Cocina", "Electrodomésticos y utensilios del hogar"),
    ("Deportes", "Equipamiento y ropa deportiva"),
    ("Moda Hombre", "Ropa y accesorios para hombre"),
    ("Moda Mujer", "Ropa y accesorios para mujer"),
    ("Libros", "Libros físicos y digitales"),
    ("Juguetes", "Juguetes y juegos para todas las edades"),
    ("Salud y Belleza", "Productos de cuidado personal y salud"),
    ("Alimentación", "Productos de alimentación gourmet"),
    ("Jardín", "Herramientas y decoración de jardín"),
    ("Automoción", "Accesorios y repuestos para vehículos"),
    ("Mascotas", "Productos para mascotas"),
    ("Oficina", "Material de oficina y papelería"),
    ("Bebés", "Productos para bebés y niños pequeños"),
    ("Bricolaje", "Herramientas y materiales de bricolaje"),
    ("Música", "Instrumentos musicales y equipos de sonido"),
    ("Videojuegos", "Consolas, videojuegos y accesorios gaming"),
    ("Fotografía", "Cámaras, objetivos y accesorios fotográficos"),
]

SUPPLIER_NAMES = [
    "TechDistribución S.L.",
    "ElectroMundo S.A.",
    "Import Global",
    "Distribuciones del Sur",
    "NorteLogística",
    "IberiaSupply",
    "MediterráneoTech",
    "CastillaComercial",
    "AtlánticoImport",
    "PirenaicaDistrib",
    "SolTech Mayoristas",
    "BéticaElectro",
    "LevanteTrade",
    "GaliciaParts",
    "CanariasImport",
    "RíojaSupplies",
    "NavarraLogistics",
    "ExtremaduraTech",
    "AsturCommerce",
    "BalearSupply",
    "CantabriaGoods",
    "AragónDistrib",
    "ManchaTrade",
    "MurciaWholesale",
    "BasqueIndustry",
    "CatalàSupply",
    "ValenciaGoods",
    "AndalucíaTrade",
    "MadridCentral",
    "CentroPenínsula",
    "EuroSupply España",
    "Global Import BCN",
    "QuickShip Madrid",
    "Importaciones León",
    "FastTrade Ibérica",
    "MegaStock S.A.",
    "ProDistribución",
    "TopWholesale ES",
    "PrimeLine Supply",
    "NexoComercial",
    "SmartImport",
    "OneStopTrade",
    "AllGoods España",
    "SuperStock Plus",
    "DirectImport Co.",
    "PeninsulaGoods",
    "TotalSupply Iberia",
    "DriveCommerce",
    "EcoImport Verde",
    "QualityFirst S.L.",
]

PRODUCT_ADJECTIVES = [
    "Premium",
    "Pro",
    "Ultra",
    "Lite",
    "Max",
    "Plus",
    "Essential",
    "Advanced",
    "Smart",
    "Classic",
    "Elite",
    "Eco",
    "Turbo",
    "Mini",
    "XL",
    "Comfort",
    "Digital",
    "Wireless",
    "Portable",
    "Compact",
]

PRODUCT_BASES = {
    "Electrónica": [
        "Altavoz Bluetooth",
        "Auriculares",
        "Cargador",
        "Cable HDMI",
        "Power Bank",
        "Hub USB",
        "Reloj inteligente",
        "Pulsera de actividad",
    ],
    "Informática": [
        "Teclado mecánico",
        "Ratón ergonómico",
        'Monitor 27"',
        "Disco SSD",
        "Memoria RAM 16GB",
        "Webcam HD",
        "Hub USB-C",
        "Soporte portátil",
    ],
    "Smartphones": [
        "Funda silicona",
        "Protector pantalla",
        "Soporte coche",
        "Cargador inalámbrico",
        "Cable USB-C",
        "Soporte escritorio",
        "Batería externa",
    ],
    "Hogar y Cocina": [
        "Robot de cocina",
        "Freidora de aire",
        "Cafetera",
        "Set cuchillos",
        "Batidora",
        "Tostadora",
        "Olla programable",
        "Sartén antiadherente",
    ],
    "Deportes": [
        "Esterilla yoga",
        "Mancuernas ajustables",
        "Cinta de correr",
        "Balón fútbol",
        "Raqueta pádel",
        "Botella térmica",
        "Guantes gimnasio",
    ],
    "Moda Hombre": [
        "Camisa Oxford",
        "Pantalón chino",
        "Zapatillas casual",
        "Cinturón piel",
        "Chaqueta vaquera",
        "Polo algodón",
        "Bufanda lana",
    ],
    "Moda Mujer": [
        "Vestido verano",
        "Bolso bandolera",
        "Zapatillas urbanas",
        "Pañuelo seda",
        "Cárdigan punto",
        "Falda midi",
        "Blusa satén",
    ],
    "Libros": [
        "Novela bestseller",
        "Manual programación",
        "Guía viaje",
        "Libro recetas",
        "Ensayo histórico",
        "Cómic manga",
        "Atlas ilustrado",
    ],
    "Juguetes": [
        "Set LEGO",
        "Muñeca articulada",
        "Puzle 1000 piezas",
        "Juego de mesa",
        "Coche teledirigido",
        "Kit ciencia",
        "Peluche grande",
    ],
    "Salud y Belleza": [
        "Crema hidratante",
        "Sérum facial",
        "Cepillo eléctrico",
        "Kit manicura",
        "Báscula digital",
        "Difusor aromas",
        "Masajeador",
    ],
    "Alimentación": [
        "Aceite oliva virgen",
        "Jamón ibérico",
        "Pack café gourmet",
        "Chocolate artesano",
        "Miel ecológica",
        "Vino reserva",
        "Queso manchego",
    ],
    "Jardín": [
        "Set herramientas",
        "Manguera extensible",
        "Maceta autorriego",
        "Semillas huerto",
        "Tijeras de podar",
        "Lámpara solar",
        "Compostador",
    ],
    "Automoción": [
        "Funda asiento",
        "Cargador coche",
        "Kit emergencia",
        "Ambientador coche",
        "Soporte móvil",
        "Aspirador portátil",
        "Cámara trasera",
    ],
    "Mascotas": [
        "Comedero automático",
        "Rascador gatos",
        "Collar LED",
        "Cama perro",
        "Transportín",
        "Juguete interactivo",
        "Arnés paseo",
    ],
    "Oficina": [
        "Organizador escritorio",
        "Lámpara LED",
        "Silla ergonómica",
        "Archivador",
        "Pizarra blanca",
        "Set rotuladores",
        "Agenda 2026",
    ],
    "Bebés": [
        "Cochecito paseo",
        "Trona plegable",
        "Set biberones",
        "Monitor bebé",
        "Mochila portabebés",
        "Juego estimulación",
        "Cuna viaje",
    ],
    "Bricolaje": [
        "Taladro percutor",
        "Set destornilladores",
        "Nivel láser",
        "Sierra circular",
        "Caja herramientas",
        "Lijadora orbital",
        "Metro láser",
    ],
    "Música": [
        "Guitarra acústica",
        "Teclado MIDI",
        "Micrófono condensador",
        "Auriculares estudio",
        "Soporte micro",
        "Interfaz audio",
        "Cables jack",
    ],
    "Videojuegos": [
        "Mando inalámbrico",
        "Alfombrilla gaming",
        "Auriculares gamer",
        "Silla gaming",
        "Capturadora vídeo",
        "Volante carreras",
        "Soporte mandos",
    ],
    "Fotografía": [
        "Trípode viaje",
        "Mochila cámara",
        "Filtro ND",
        "Flash externo",
        "Tarjeta SD 128GB",
        "Limpiador lentes",
        "Disparador remoto",
    ],
}

DEPARTMENT_DATA = [
    ("Dirección General", "Dirección estratégica de la empresa"),
    ("Recursos Humanos", "Gestión del talento y personal"),
    ("Finanzas", "Contabilidad, presupuestos y finanzas"),
    ("Marketing", "Campañas, publicidad y comunicación"),
    ("Ventas", "Gestión comercial y atención al cliente"),
    ("Tecnología", "Desarrollo, infraestructura IT y soporte"),
    ("Logística", "Almacén, envíos y cadena de suministro"),
    ("Compras", "Aprovisionamiento y relación con proveedores"),
    ("Atención al Cliente", "Soporte post-venta y reclamaciones"),
    ("Legal", "Asesoría jurídica y cumplimiento normativo"),
]

JOB_TITLES = {
    "Dirección General": ["CEO", "Director General Adjunto", "Asistente de Dirección"],
    "Recursos Humanos": [
        "Director RRHH",
        "Técnico RRHH",
        "Responsable Selección",
        "Administrativo RRHH",
    ],
    "Finanzas": [
        "Director Financiero",
        "Controller",
        "Contable Senior",
        "Analista Financiero",
        "Contable Junior",
    ],
    "Marketing": [
        "Director Marketing",
        "Community Manager",
        "Diseñador Gráfico",
        "Analista SEO",
        "Content Manager",
    ],
    "Ventas": [
        "Director Comercial",
        "Key Account Manager",
        "Ejecutivo de Ventas",
        "Técnico Comercial",
    ],
    "Tecnología": [
        "CTO",
        "Desarrollador Backend",
        "Desarrollador Frontend",
        "DevOps Engineer",
        "QA Engineer",
        "Data Analyst",
    ],
    "Logística": [
        "Director Logística",
        "Jefe de Almacén",
        "Coordinador Envíos",
        "Operario Almacén",
    ],
    "Compras": [
        "Director Compras",
        "Comprador Senior",
        "Comprador Junior",
        "Analista de Proveedores",
    ],
    "Atención al Cliente": [
        "Responsable SAC",
        "Agente Soporte Nivel 1",
        "Agente Soporte Nivel 2",
        "Coordinador Reclamaciones",
    ],
    "Legal": [
        "Director Legal",
        "Abogado Senior",
        "Abogado Junior",
        "Responsable Cumplimiento",
    ],
}

ORDER_STATUSES = [
    "pending",
    "confirmed",
    "processing",
    "shipped",
    "delivered",
    "cancelled",
    "returned",
]
PAYMENT_METHODS = [
    "credit_card",
    "debit_card",
    "paypal",
    "bank_transfer",
    "bizum",
    "cash_on_delivery",
]
REVIEW_COMMENTS_POSITIVE = [
    "Excelente producto, muy recomendable.",
    "Muy buena calidad-precio, repetiré.",
    "Llegó rápido y bien embalado. Perfecto.",
    "Cumple con lo prometido, estoy satisfecho.",
    "Gran compra, lo recomiendo sin duda.",
    "Supera mis expectativas, fantástico.",
    "Muy contento con la compra, funciona genial.",
    "Producto de calidad, envío rapidísimo.",
    "Todo perfecto, excelente experiencia.",
    "Justo lo que buscaba, muy útil.",
]
REVIEW_COMMENTS_NEUTRAL = [
    "Producto correcto, nada especial.",
    "Cumple su función, sin más.",
    "Aceptable por el precio que tiene.",
    "Normal, esperaba un poco más.",
    "Está bien para lo que cuesta.",
]
REVIEW_COMMENTS_NEGATIVE = [
    "No cumple con la descripción del producto.",
    "Llegó con retraso y algo dañado.",
    "Calidad inferior a lo esperado.",
    "Decepcionante, no lo recomiendo.",
    "Mala experiencia, pedí devolución.",
]

TRANSACTION_TYPES = ["payment", "refund", "adjustment", "shipping_fee", "discount"]


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _rand_date(start: datetime, end: datetime) -> str:
    """Fecha aleatoria entre start y end en formato ISO."""
    delta = end - start
    offset = random.randint(0, int(delta.total_seconds()))
    dt = start + timedelta(seconds=offset)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _rand_phone() -> str:
    return f"+34 6{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)}"


def _rand_email(first: str, last: str, idx: int) -> str:
    """Email único basado en nombre + índice."""
    clean = (
        first.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    clean_last = (
        last.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )
    domain = random.choice(EMAIL_DOMAINS)
    return f"{clean}.{clean_last}{idx}@{domain}"


def _rand_address() -> Tuple[str, str, str, str]:
    """Devuelve (dirección, ciudad, provincia, código postal)."""
    city, province, cp = random.choice(CITIES)
    street = random.choice(STREETS)
    num = random.randint(1, 120)
    cp_var = str(int(cp) + random.randint(0, 50)).zfill(5)
    return f"{street}, {num}", city, province, cp_var


def _rand_sku(cat_id: int, prod_id: int) -> str:
    return f"SKU-{cat_id:02d}-{prod_id:05d}"


def _rand_money(low: float, high: float) -> float:
    return round(random.uniform(low, high), 2)


# ────────────────────────────────────────────────────────────────────
# Schema DDL
# ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- ============================================================
-- Base de datos de ejemplo para Custom-RAG (e-commerce)
-- ============================================================

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Categorías de producto
CREATE TABLE IF NOT EXISTS categories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Proveedores
CREATE TABLE IF NOT EXISTS suppliers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    contact_name TEXT,
    email        TEXT,
    phone        TEXT,
    address      TEXT,
    city         TEXT,
    province     TEXT,
    postal_code  TEXT,
    country      TEXT    NOT NULL DEFAULT 'España',
    rating       REAL    CHECK (rating BETWEEN 0 AND 5),
    active       INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Productos
CREATE TABLE IF NOT EXISTS products (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sku           TEXT    NOT NULL UNIQUE,
    name          TEXT    NOT NULL,
    description   TEXT,
    category_id   INTEGER NOT NULL REFERENCES categories(id),
    supplier_id   INTEGER REFERENCES suppliers(id),
    price         REAL    NOT NULL CHECK (price >= 0),
    cost          REAL    CHECK (cost >= 0),
    stock         INTEGER NOT NULL DEFAULT 0,
    weight_kg     REAL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_supplier ON products(supplier_id);
CREATE INDEX IF NOT EXISTS idx_products_price    ON products(price);

-- Usuarios / clientes
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name     TEXT    NOT NULL,
    last_name      TEXT    NOT NULL,
    email          TEXT    NOT NULL UNIQUE,
    phone          TEXT,
    address        TEXT,
    city           TEXT,
    province       TEXT,
    postal_code    TEXT,
    country        TEXT    NOT NULL DEFAULT 'España',
    date_of_birth  TEXT,
    status         TEXT    NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active','inactive','suspended')),
    role           TEXT    NOT NULL DEFAULT 'user'
                          CHECK (role IN ('admin','user','guest')),
    registered_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login     TEXT
);

CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_users_city   ON users(city);

-- Departamentos
CREATE TABLE IF NOT EXISTS departments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Empleados
CREATE TABLE IF NOT EXISTS employees (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name     TEXT    NOT NULL,
    last_name      TEXT    NOT NULL,
    email          TEXT    NOT NULL UNIQUE,
    phone          TEXT,
    department_id  INTEGER NOT NULL REFERENCES departments(id),
    job_title      TEXT    NOT NULL,
    salary         REAL    NOT NULL CHECK (salary >= 0),
    hire_date      TEXT    NOT NULL,
    is_active      INTEGER NOT NULL DEFAULT 1,
    manager_id     INTEGER REFERENCES employees(id)
);

CREATE INDEX IF NOT EXISTS idx_employees_dept ON employees(department_id);

-- Pedidos
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    status          TEXT    NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending','confirmed','processing',
                                             'shipped','delivered','cancelled','returned')),
    payment_method  TEXT,
    shipping_address TEXT,
    shipping_city    TEXT,
    shipping_province TEXT,
    shipping_postal_code TEXT,
    subtotal        REAL    NOT NULL DEFAULT 0,
    tax             REAL    NOT NULL DEFAULT 0,
    shipping_cost   REAL    NOT NULL DEFAULT 0,
    total           REAL    NOT NULL DEFAULT 0,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_user      ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status     ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);

-- Líneas de pedido
CREATE TABLE IF NOT EXISTS order_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    unit_price  REAL    NOT NULL CHECK (unit_price >= 0),
    discount    REAL    NOT NULL DEFAULT 0,
    line_total  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_order_items_order   ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);

-- Reseñas de producto
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER NOT NULL REFERENCES products(id),
    user_id     INTEGER NOT NULL REFERENCES users(id),
    rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title       TEXT,
    comment     TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user    ON reviews(user_id);

-- Inventario (movimientos de stock)
CREATE TABLE IF NOT EXISTS inventory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id  INTEGER NOT NULL REFERENCES products(id),
    change      INTEGER NOT NULL,          -- positivo = entrada, negativo = salida
    reason      TEXT    NOT NULL
                       CHECK (reason IN ('purchase','sale','return','adjustment','damaged')),
    reference   TEXT,                      -- p.ej. order_id o PO number
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);

-- Transacciones financieras
CREATE TABLE IF NOT EXISTS transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        INTEGER REFERENCES orders(id),
    user_id         INTEGER REFERENCES users(id),
    type            TEXT    NOT NULL
                           CHECK (type IN ('payment','refund','adjustment','shipping_fee','discount')),
    amount          REAL    NOT NULL,
    currency        TEXT    NOT NULL DEFAULT 'EUR',
    payment_method  TEXT,
    status          TEXT    NOT NULL DEFAULT 'completed'
                           CHECK (status IN ('pending','completed','failed','reversed')),
    description     TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transactions_order ON transactions(order_id);
CREATE INDEX IF NOT EXISTS idx_transactions_user  ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_type  ON transactions(type);
"""


# ────────────────────────────────────────────────────────────────────
# Data generation
# ────────────────────────────────────────────────────────────────────


def populate(conn: sqlite3.Connection) -> None:
    """Puebla todas las tablas con datos realistas."""
    cur = conn.cursor()

    # ── Categorías ──────────────────────────────────────────────────
    log.info("Insertando categorías …")
    for name, desc in CATEGORY_DATA:
        cur.execute(
            "INSERT INTO categories (name, description) VALUES (?, ?)",
            (name, desc),
        )

    # ── Proveedores ─────────────────────────────────────────────────
    log.info("Insertando proveedores …")
    for sname in SUPPLIER_NAMES:
        addr, city, prov, cp = _rand_address()
        contact = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        email = (
            sname.lower().replace(" ", "").replace(".", "").replace(",", "")[:15]
            + "@proveedor.es"
        )
        cur.execute(
            """INSERT INTO suppliers
               (name, contact_name, email, phone, address, city, province, postal_code, rating, active)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                sname,
                contact,
                email,
                _rand_phone(),
                addr,
                city,
                prov,
                cp,
                round(random.uniform(2.5, 5.0), 1),
                1 if random.random() > 0.05 else 0,
            ),
        )

    # ── Productos ───────────────────────────────────────────────────
    log.info("Insertando productos …")
    cat_ids = [r[0] for r in cur.execute("SELECT id FROM categories").fetchall()]
    cat_names = {
        r[0]: r[1] for r in cur.execute("SELECT id, name FROM categories").fetchall()
    }
    sup_ids = [r[0] for r in cur.execute("SELECT id FROM suppliers").fetchall()]

    product_count = 0
    for cat_id in cat_ids:
        cat_name = cat_names[cat_id]
        bases = PRODUCT_BASES.get(cat_name, ["Producto genérico"])
        # Generar ~25 productos por categoría
        for _ in range(25):
            base = random.choice(bases)
            adj = random.choice(PRODUCT_ADJECTIVES)
            name = f"{base} {adj}"
            product_count += 1
            sku = _rand_sku(cat_id, product_count)
            cost = _rand_money(5.0, 300.0)
            margin = random.uniform(1.15, 2.5)
            price = round(cost * margin, 2)
            stock = random.randint(0, 500)
            weight = round(random.uniform(0.1, 15.0), 2)
            created = _rand_date(datetime(2023, 1, 1), datetime(2025, 12, 31))
            cur.execute(
                """INSERT INTO products
                   (sku, name, description, category_id, supplier_id, price, cost, stock, weight_kg, is_active, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sku,
                    name,
                    f"{name} de alta calidad. Categoría: {cat_name}.",
                    cat_id,
                    random.choice(sup_ids),
                    price,
                    cost,
                    stock,
                    weight,
                    1 if random.random() > 0.08 else 0,
                    created,
                    created,
                ),
            )

    # ── Usuarios ────────────────────────────────────────────────────
    log.info("Insertando usuarios …")
    NUM_USERS = 2000
    for i in range(1, NUM_USERS + 1):
        fn = random.choice(FIRST_NAMES)
        ln = f"{random.choice(LAST_NAMES)} {random.choice(LAST_NAMES)}"
        addr, city, prov, cp = _rand_address()
        dob = _rand_date(datetime(1960, 1, 1), datetime(2005, 12, 31))[:10]
        status = random.choices(
            ["active", "inactive", "suspended"], weights=[85, 10, 5]
        )[0]
        role = random.choices(["user", "admin", "guest"], weights=[80, 5, 15])[0]
        reg = _rand_date(datetime(2020, 1, 1), datetime(2025, 12, 31))
        last_login = (
            _rand_date(datetime(2025, 6, 1), datetime(2026, 2, 10))
            if status == "active"
            else None
        )
        cur.execute(
            """INSERT INTO users
               (first_name, last_name, email, phone, address, city, province,
                postal_code, date_of_birth, status, role, registered_at, last_login)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fn,
                ln,
                _rand_email(fn, ln, i),
                _rand_phone(),
                addr,
                city,
                prov,
                cp,
                dob,
                status,
                role,
                reg,
                last_login,
            ),
        )

    # ── Departamentos y Empleados ───────────────────────────────────
    log.info("Insertando departamentos y empleados …")
    for dname, ddesc in DEPARTMENT_DATA:
        cur.execute(
            "INSERT INTO departments (name, description) VALUES (?, ?)",
            (dname, ddesc),
        )

    dept_ids = {
        r[1]: r[0] for r in cur.execute("SELECT id, name FROM departments").fetchall()
    }
    employee_id = 0

    for dept_name, titles in JOB_TITLES.items():
        dept_id = dept_ids[dept_name]
        manager_eid = None
        # generar entre 8 y 20 empleados por depto
        n_emps = random.randint(8, 20)
        for j in range(n_emps):
            employee_id += 1
            fn = random.choice(FIRST_NAMES)
            ln = f"{random.choice(LAST_NAMES)} {random.choice(LAST_NAMES)}"
            title = titles[j % len(titles)]
            base_salary = (
                random.uniform(22000, 85000)
                if "Director" not in title and "CTO" not in title and "CEO" not in title
                else random.uniform(65000, 130000)
            )
            hire = _rand_date(datetime(2018, 1, 1), datetime(2025, 12, 31))[:10]
            email = f"emp.{fn.lower().replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')}.{employee_id}@empresa.com"
            cur.execute(
                """INSERT INTO employees
                   (first_name, last_name, email, phone, department_id, job_title,
                    salary, hire_date, is_active, manager_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    fn,
                    ln,
                    email,
                    _rand_phone(),
                    dept_id,
                    title,
                    round(base_salary, 2),
                    hire,
                    1 if random.random() > 0.05 else 0,
                    manager_eid,
                ),
            )
            # El primer empleado de cada depto es el manager
            if j == 0:
                manager_eid = employee_id

    # ── Pedidos y líneas de pedido ──────────────────────────────────
    log.info("Insertando pedidos …")
    user_ids = [
        r[0]
        for r in cur.execute("SELECT id FROM users WHERE status='active'").fetchall()
    ]
    prod_rows = cur.execute(
        "SELECT id, price FROM products WHERE is_active=1"
    ).fetchall()
    prod_price = {r[0]: r[1] for r in prod_rows}
    prod_ids = list(prod_price.keys())

    NUM_ORDERS = 8000
    TAX_RATE = 0.21

    order_data = []  # (order_id, user_id, items_list)

    for oid in range(1, NUM_ORDERS + 1):
        uid = random.choice(user_ids)
        status = random.choices(
            ORDER_STATUSES,
            weights=[5, 8, 10, 15, 50, 8, 4],
        )[0]
        payment = random.choice(PAYMENT_METHODS)
        addr, city, prov, cp = _rand_address()
        created = _rand_date(datetime(2024, 1, 1), datetime(2026, 2, 9))

        # Generar líneas
        n_items = random.choices([1, 2, 3, 4, 5], weights=[30, 35, 20, 10, 5])[0]
        chosen_prods = random.sample(prod_ids, min(n_items, len(prod_ids)))
        items = []
        subtotal = 0.0
        for pid in chosen_prods:
            qty = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
            unit_price = prod_price[pid]
            discount_pct = random.choices(
                [0, 5, 10, 15, 20], weights=[60, 15, 12, 8, 5]
            )[0]
            discount = round(unit_price * qty * discount_pct / 100, 2)
            line_total = round(unit_price * qty - discount, 2)
            subtotal += line_total
            items.append((pid, qty, unit_price, discount, line_total))

        subtotal = round(subtotal, 2)
        tax = round(subtotal * TAX_RATE, 2)
        shipping = round(random.choice([0, 0, 0, 3.99, 4.99, 6.99, 9.99]), 2)
        total = round(subtotal + tax + shipping, 2)

        cur.execute(
            """INSERT INTO orders
               (user_id, status, payment_method, shipping_address, shipping_city,
                shipping_province, shipping_postal_code, subtotal, tax,
                shipping_cost, total, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                uid,
                status,
                payment,
                addr,
                city,
                prov,
                cp,
                subtotal,
                tax,
                shipping,
                total,
                created,
                created,
            ),
        )
        order_data.append((oid, uid, total, status, payment, created, items))

    log.info("Insertando líneas de pedido …")
    for oid, uid, total, status, payment, created, items in order_data:
        for pid, qty, up, disc, lt in items:
            cur.execute(
                """INSERT INTO order_items
                   (order_id, product_id, quantity, unit_price, discount, line_total)
                   VALUES (?,?,?,?,?,?)""",
                (oid, pid, qty, up, disc, lt),
            )

    # ── Reseñas ─────────────────────────────────────────────────────
    log.info("Insertando reseñas …")
    NUM_REVIEWS = 5000
    for _ in range(NUM_REVIEWS):
        pid = random.choice(prod_ids)
        uid = random.choice(user_ids)
        rating = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 15, 35, 37])[0]
        if rating >= 4:
            comment = random.choice(REVIEW_COMMENTS_POSITIVE)
        elif rating == 3:
            comment = random.choice(REVIEW_COMMENTS_NEUTRAL)
        else:
            comment = random.choice(REVIEW_COMMENTS_NEGATIVE)
        title_words = comment.split()[:4]
        title = " ".join(title_words)
        created = _rand_date(datetime(2024, 1, 1), datetime(2026, 2, 10))
        cur.execute(
            """INSERT INTO reviews (product_id, user_id, rating, title, comment, created_at)
               VALUES (?,?,?,?,?,?)""",
            (pid, uid, rating, title, comment, created),
        )

    # ── Inventario ──────────────────────────────────────────────────
    log.info("Insertando movimientos de inventario …")
    reasons = ["purchase", "sale", "return", "adjustment", "damaged"]
    for pid in prod_ids:
        n_moves = random.randint(0, 5)
        for _ in range(n_moves):
            reason = random.choice(reasons)
            if reason in ("purchase", "return"):
                change = random.randint(10, 200)
            elif reason == "sale":
                change = -random.randint(1, 50)
            elif reason == "damaged":
                change = -random.randint(1, 10)
            else:
                change = random.randint(-20, 20)
            created = _rand_date(datetime(2024, 6, 1), datetime(2026, 2, 10))
            cur.execute(
                """INSERT INTO inventory (product_id, change, reason, created_at)
                   VALUES (?,?,?,?)""",
                (pid, change, reason, created),
            )

    # ── Transacciones ───────────────────────────────────────────────
    log.info("Insertando transacciones …")
    for oid, uid, total, status, payment, created, _ in order_data:
        if status == "cancelled":
            continue
        # Pago principal
        cur.execute(
            """INSERT INTO transactions
               (order_id, user_id, type, amount, payment_method, status, description, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                oid,
                uid,
                "payment",
                total,
                payment,
                "completed",
                f"Pago pedido #{oid}",
                created,
            ),
        )
        # Reembolso si devuelto
        if status == "returned":
            refund_date = _rand_date(
                datetime.strptime(created, "%Y-%m-%d %H:%M:%S"),
                datetime(2026, 2, 10),
            )
            cur.execute(
                """INSERT INTO transactions
                   (order_id, user_id, type, amount, payment_method, status, description, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    oid,
                    uid,
                    "refund",
                    -total,
                    payment,
                    "completed",
                    f"Reembolso pedido #{oid}",
                    refund_date,
                ),
            )

    conn.commit()
    log.info("✓ Datos insertados correctamente.")


# ────────────────────────────────────────────────────────────────────
# Estadísticas
# ────────────────────────────────────────────────────────────────────


def print_stats(conn: sqlite3.Connection) -> None:
    """Muestra resumen de filas por tabla."""
    cur = conn.cursor()
    tables = [
        "categories",
        "suppliers",
        "products",
        "users",
        "departments",
        "employees",
        "orders",
        "order_items",
        "reviews",
        "inventory",
        "transactions",
    ]
    print("\n" + "=" * 50)
    print("  RESUMEN DE LA BASE DE DATOS")
    print("=" * 50)
    total = 0
    for t in tables:
        count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total += count
        print(f"  {t:<20s} {count:>8,} filas")
    print("-" * 50)
    print(f"  {'TOTAL':<20s} {total:>8,} filas")
    print("=" * 50)

    # Algunas consultas de ejemplo
    print("\n  Consultas de verificación rápida:")
    print("  ---------------------------------")

    row = cur.execute("SELECT AVG(price) FROM products").fetchone()
    print(f"  · Precio medio de productos: {row[0]:.2f} €")

    row = cur.execute(
        "SELECT SUM(total) FROM orders WHERE status='delivered'"
    ).fetchone()
    print(f"  · Revenue (pedidos entregados): {row[0]:,.2f} €")

    row = cur.execute("SELECT AVG(rating) FROM reviews").fetchone()
    print(f"  · Valoración media de reseñas: {row[0]:.2f} / 5")

    row = cur.execute("SELECT COUNT(DISTINCT user_id) FROM orders").fetchone()
    print(f"  · Usuarios con al menos 1 pedido: {row[0]:,}")

    row = cur.execute(
        "SELECT c.name, COUNT(*) cnt FROM products p "
        "JOIN categories c ON c.id=p.category_id "
        "GROUP BY c.name ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    print(f"  · Categoría con más productos: {row[0]} ({row[1]})")

    size_bytes = os.path.getsize(conn.execute("PRAGMA database_list").fetchone()[2])
    print(f"\n  Tamaño del fichero: {size_bytes / 1024 / 1024:.2f} MB\n")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea una base de datos SQLite de ejemplo para Custom-RAG.",
    )
    parser.add_argument(
        "--path",
        default=DB_DEFAULT_PATH,
        help=f"Ruta del fichero .db (por defecto: {DB_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe la BD si ya existe.",
    )
    args = parser.parse_args()

    db_path = Path(args.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        if args.force:
            log.warning("Sobrescribiendo base de datos existente: %s", db_path)
            db_path.unlink()
        else:
            log.error(
                "El fichero %s ya existe. Usa --force para sobrescribirlo.",
                db_path,
            )
            sys.exit(1)

    log.info("Creando base de datos en: %s", db_path.resolve())
    conn = sqlite3.connect(str(db_path))

    try:
        log.info("Creando esquema …")
        conn.executescript(SCHEMA_SQL)
        populate(conn)
        print_stats(conn)
    except Exception:
        log.exception("Error al crear la base de datos.")
        conn.close()
        if db_path.exists():
            db_path.unlink()
        sys.exit(1)
    finally:
        conn.close()

    log.info("✓ Base de datos creada exitosamente en: %s", db_path.resolve())


if __name__ == "__main__":
    main()
