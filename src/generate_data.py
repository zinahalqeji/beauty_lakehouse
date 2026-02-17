#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/generate_data.py

Generate synthetic e-commerce data (customers, products, orders, order_items)
with a deterministic product_type -> category mapping.

Configured for:
- 10 000 customers
- 2 000 products
- 100 000 orders

Outputs (local):
  data/raw/customers.csv
  data/raw/products.csv
  data/raw/orders.csv
  data/raw/order_items.csv
  data/raw/metadata.json
"""

import os
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
SEED = 42
N_CUSTOMERS = 10_000
N_PRODUCTS = 2_000
N_ORDERS = 100_000
MIN_ITEMS_PER_ORDER = 1
MAX_ITEMS_PER_ORDER = 6

OUTPUT_DIR = "data/raw"
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

# -------------------------
# Setup
# -------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(SEED)
fake = Faker("sv_SE")
Faker.seed(SEED)

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=3 * 365)

SWEDISH_CITIES = [
    "Stockholm",
    "Göteborg",
    "Malmö",
    "Uppsala",
    "Västerås",
    "Örebro",
    "Linköping",
    "Helsingborg",
    "Jönköping",
    "Norrköping",
    "Lund",
    "Umeå",
    "Gävle",
    "Borås",
    "Södertälje",
    "Eskilstuna",
    "Halmstad",
    "Växjö",
    "Karlstad",
    "Täby",
]

# Authoritative mapping product_type -> category
PRODUCT_TYPE_TO_CATEGORY = {
    # Hair care
    "Shampoo": "Shampoo",
    "Conditioner": "Conditioner",
    "Hair Mask": "Hair Mask",
    "Leave-in Treatment": "Hair Treatment",
    "Scalp Serum": "Hair Treatment",
    "Dry Shampoo": "Shampoo",
    "Hair Oil": "Hair Treatment",
    "Hair Serum": "Hair Treatment",
    # Body care
    "Body Lotion": "Body Care",
    "Body Wash": "Body Care",
    "Body Scrub": "Body Care",
    "Hand Cream": "Hand Care",
    # Face care
    "Face Cleanser": "Face Care",
    "Face Cream": "Face Care",
    "Face Serum": "Face Care",
    "Toner": "Face Care",
    "BB Cream": "Face Care",
    # Makeup
    "Foundation": "Makeup",
    "Blush": "Makeup",
    "Mascara": "Makeup",
    "Lip Balm": "Makeup",
    "Lipstick": "Makeup",
    # Nail care / tools
    "Nail Polish": "Nail Care",
    "Base Coat": "Nail Care",
    "Top Coat": "Nail Care",
    "Cuticle Oil": "Nail Care",
    "Nail Strengthener": "Nail Care",
    "Nail File": "Nail Tools",
    "Nail Clippers": "Nail Tools",
    "Nail Brush": "Nail Tools",
}

PRODUCT_TYPES = list(PRODUCT_TYPE_TO_CATEGORY.keys())
PAYMENT_TYPES = ["card", "invoice", "paypal", "swish"]
ORDER_STATUSES = ["completed", "cancelled", "returned"]


def random_date_between(start: datetime, end: datetime) -> str:
    delta = end - start
    rand_days = np.random.randint(0, delta.days + 1)
    return (start + timedelta(days=int(rand_days))).date().isoformat()


# -------------------------
# Generate customers
# -------------------------
print("Generating customers...")
customers = []
for i in range(1, N_CUSTOMERS + 1):
    name = fake.name().split()
    first = name[0]
    last = name[-1]
    signup = random_date_between(START_DATE, END_DATE)
    age = int(np.clip(np.random.normal(35, 10), 18, 90))
    city = np.random.choice(SWEDISH_CITIES)
    customers.append(
        {
            "customer_id": i,
            "first_name": first,
            "last_name": last,
            "email": f"user{i}@example.com",
            "signup_date": signup,
            "city": city,
            "age": age,
        }
    )

customers_df = pd.DataFrame(customers)
customers_path = os.path.join(OUTPUT_DIR, "customers.csv")
customers_df.to_csv(customers_path, index=False)
print(f"Saved customers: {customers_path} ({len(customers_df)} rows)")

signup_map = {}
for _, row in customers_df.iterrows():
    try:
        signup_map[int(row["customer_id"])] = pd.to_datetime(row["signup_date"]).date()
    except Exception:
        signup_map[int(row["customer_id"])] = START_DATE.date()

# -------------------------
# Generate products
# -------------------------
print("Generating products...")
adjectives = [
    "Hydra",
    "Silk",
    "Pure",
    "Gentle",
    "Revive",
    "Nourish",
    "Balance",
    "Glow",
    "Radiant",
    "Calming",
    "Repair",
    "Botanical",
    "Fresh",
    "Velvet",
    "Luxe",
    "Bright",
    "Soothing",
    "Clarifying",
]
sizes = ["30ml", "50ml", "75ml", "100ml", "150ml", "200ml", "250ml"]

prices = np.round(np.random.lognormal(mean=2.8, sigma=0.8, size=N_PRODUCTS), 2)

products = []
for i in range(1, N_PRODUCTS + 1):
    ptype = np.random.choice(PRODUCT_TYPES)
    category = PRODUCT_TYPE_TO_CATEGORY[ptype]
    price = float(prices[i - 1])
    cost = round(price * np.random.uniform(0.40, 0.70), 2)
    stock = int(np.random.poisson(120))
    product_name = f"{np.random.choice(adjectives)} {ptype} {np.random.choice(sizes)}"
    products.append(
        {
            "product_id": i,
            "product_name": product_name,
            "product_type": ptype,
            "category": category,
            "price": price,
            "cost": cost,
            "available_stock": stock,
        }
    )

products_df = pd.DataFrame(products)
products_path = os.path.join(OUTPUT_DIR, "products.csv")
products_df.to_csv(products_path, index=False)
print(f"Saved products: {products_path} ({len(products_df)} rows)")

price_map = products_df.set_index("product_id")["price"].to_dict()

# Popularity weights (Zipf-like)
ranks = np.arange(1, N_PRODUCTS + 1)
zipf_weights = 1.0 / ranks
zipf_weights = zipf_weights / zipf_weights.sum()

# -------------------------
# Generate orders and order_items
# -------------------------
print("Generating orders and order_items...")
orders_path = os.path.join(OUTPUT_DIR, "orders.csv")
order_items_path = os.path.join(OUTPUT_DIR, "order_items.csv")

orders_columns = [
    "order_id",
    "customer_id",
    "order_date",
    "total_amount",
    "payment_type",
    "status",
]
order_items_columns = [
    "order_item_id",
    "order_id",
    "product_id",
    "quantity",
    "unit_price",
    "line_total",
]

with (
    open(orders_path, "w", encoding="utf-8") as f_orders,
    open(order_items_path, "w", encoding="utf-8") as f_items,
):
    f_orders.write(",".join(orders_columns) + "\n")
    f_items.write(",".join(order_items_columns) + "\n")

    order_item_id = 0

    item_choices = np.arange(MIN_ITEMS_PER_ORDER, MAX_ITEMS_PER_ORDER + 1)
    base_probs = np.array([0.50, 0.25, 0.15, 0.07, 0.02, 0.01])[: len(item_choices)]
    base_probs = base_probs / base_probs.sum()

    for order_id in tqdm(range(1, N_ORDERS + 1), desc="orders"):
        try:
            customer_id = int(np.random.randint(1, N_CUSTOMERS + 1))
            signup_dt = signup_map.get(customer_id, START_DATE.date())

            if signup_dt >= END_DATE.date():
                order_date = signup_dt.isoformat()
            else:
                delta_days = (END_DATE.date() - signup_dt).days
                rand_days = np.random.randint(0, delta_days + 1)
                order_date = (signup_dt + timedelta(days=int(rand_days))).isoformat()

            payment_type = np.random.choice(PAYMENT_TYPES, p=[0.6, 0.15, 0.15, 0.1])
            status = np.random.choice(ORDER_STATUSES, p=[0.95, 0.03, 0.02])

            n_items = int(np.random.choice(item_choices, p=base_probs))
            product_ids = np.random.choice(
                np.arange(1, N_PRODUCTS + 1),
                size=n_items,
                replace=False,
                p=zipf_weights,
            )

            line_totals = []
            for pid in product_ids:
                order_item_id += 1
                quantity = int(
                    np.random.choice(
                        [1, 1, 1, 2, 2, 3],
                        p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05],
                    )
                )
                unit_price = price_map.get(int(pid))
                if unit_price is None:
                    raise KeyError(f"product_id {pid} not found in price_map")
                discount = np.random.choice(
                    [0.0, 0.0, 0.05, 0.1], p=[0.8, 0.1, 0.08, 0.02]
                )
                unit_price_after = round(unit_price * (1 - discount), 2)
                line_total = round(quantity * unit_price_after, 2)
                line_totals.append(line_total)

                f_items.write(
                    f"{order_item_id},{order_id},{pid},{quantity},{unit_price_after},{line_total}\n"
                )

            total_amount = round(sum(line_totals), 2)
            f_orders.write(
                f"{order_id},{customer_id},{order_date},{total_amount},{payment_type},{status}\n"
            )

        except Exception as e:
            print(f"Error generating order {order_id}: {e}", file=sys.stderr)
            continue

# -------------------------
# Save metadata
# -------------------------
metadata = {
    "seed": SEED,
    "n_customers": N_CUSTOMERS,
    "n_products": N_PRODUCTS,
    "n_orders": N_ORDERS,
    "min_items_per_order": MIN_ITEMS_PER_ORDER,
    "max_items_per_order": MAX_ITEMS_PER_ORDER,
    "generated_at": datetime.utcnow().isoformat() + "Z",
}
with open(METADATA_PATH, "w", encoding="utf-8") as mf:
    json.dump(metadata, mf, indent=2)

print(f"Saved customers: {customers_path}")
print(f"Saved products: {products_path}")
print(f"Saved orders: {orders_path}")
print(f"Saved order_items: {order_items_path}")
print(f"Saved metadata: {METADATA_PATH}")
print("Generation complete.")
