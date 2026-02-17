import pandas as pd
import os
from datetime import datetime

DATA_DIR = "data/raw"

# Expected schemas
EXPECTED = {
    "customers.csv": [
        "customer_id",
        "first_name",
        "last_name",
        "email",
        "signup_date",
        "city",
        "age",
    ],
    "products.csv": [
        "product_id",
        "product_name",
        "product_type",
        "category",
        "price",
        "cost",
        "available_stock",
    ],
    "orders.csv": [
        "order_id",
        "customer_id",
        "order_date",
        "total_amount",
        "payment_type",
        "status",
    ],
    "order_items.csv": [
        "order_item_id",
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "line_total",
    ],
}

# Authoritative mapping
PRODUCT_TYPE_TO_CATEGORY = {
    "Shampoo": "Shampoo",
    "Conditioner": "Conditioner",
    "Hair Mask": "Hair Mask",
    "Leave-in Treatment": "Hair Treatment",
    "Scalp Serum": "Hair Treatment",
    "Dry Shampoo": "Shampoo",
    "Hair Oil": "Hair Treatment",
    "Hair Serum": "Hair Treatment",
    "Body Lotion": "Body Care",
    "Body Wash": "Body Care",
    "Body Scrub": "Body Care",
    "Hand Cream": "Hand Care",
    "Face Cleanser": "Face Care",
    "Face Cream": "Face Care",
    "Face Serum": "Face Care",
    "Toner": "Face Care",
    "BB Cream": "Face Care",
    "Foundation": "Makeup",
    "Blush": "Makeup",
    "Mascara": "Makeup",
    "Lip Balm": "Makeup",
    "Lipstick": "Makeup",
    "Nail Polish": "Nail Care",
    "Base Coat": "Nail Care",
    "Top Coat": "Nail Care",
    "Cuticle Oil": "Nail Care",
    "Nail Strengthener": "Nail Care",
    "Nail File": "Nail Tools",
    "Nail Clippers": "Nail Tools",
    "Nail Brush": "Nail Tools",
}


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {len(df)} rows")
    return df


def check_schema(df, expected_cols, name):
    missing = set(expected_cols) - set(df.columns)
    extra = set(df.columns) - set(expected_cols)
    if missing:
        print(f"❌ {name}: Missing columns: {missing}")
    if extra:
        print(f"❌ {name}: Unexpected columns: {extra}")
    if not missing and not extra:
        print(f"✔ {name}: Schema OK")


def main():
    print("\n=== Loading datasets ===")
    customers = load_csv("customers.csv")
    products = load_csv("products.csv")
    orders = load_csv("orders.csv")
    items = load_csv("order_items.csv")

    print("\n=== Schema validation ===")
    for name, cols in EXPECTED.items():
        df = load_csv(name)
        check_schema(df, cols, name)

    print("\n=== Referential integrity ===")
    # orders -> customers
    missing_customers = set(orders.customer_id) - set(customers.customer_id)
    print(
        "✔ orders.customer_id OK"
        if not missing_customers
        else f"❌ Missing customers: {missing_customers}"
    )

    # order_items -> orders
    missing_orders = set(items.order_id) - set(orders.order_id)
    print(
        "✔ order_items.order_id OK"
        if not missing_orders
        else f"❌ Missing orders: {missing_orders}"
    )

    # order_items -> products
    missing_products = set(items.product_id) - set(products.product_id)
    print(
        "✔ order_items.product_id OK"
        if not missing_products
        else f"❌ Missing products: {missing_products}"
    )

    print("\n=== Business logic checks ===")
    # price >= cost
    bad_price = products[products.price < products.cost]
    print(
        "✔ price >= cost"
        if bad_price.empty
        else f"❌ {len(bad_price)} products have price < cost"
    )

    # category matches product_type
    mismatches = products[
        products.apply(
            lambda r: PRODUCT_TYPE_TO_CATEGORY.get(r.product_type) != r.category, axis=1
        )
    ]
    print(
        "✔ category mapping OK"
        if mismatches.empty
        else f"❌ {len(mismatches)} category mismatches"
    )

    # order_date >= signup_date
    merged = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id")
    merged["order_date"] = pd.to_datetime(merged.order_date)
    merged["signup_date"] = pd.to_datetime(merged.signup_date)
    bad_dates = merged[merged.order_date < merged.signup_date]
    print(
        "✔ order_date >= signup_date"
        if bad_dates.empty
        else f"❌ {len(bad_dates)} invalid order dates"
    )

    # line_total = quantity * unit_price
    items["calc"] = items.quantity * items.unit_price
    bad_line_total = items[abs(items.calc - items.line_total) > 0.01]
    print(
        "✔ line_total correct"
        if bad_line_total.empty
        else f"❌ {len(bad_line_total)} incorrect line_totals"
    )

    print("\n=== Uniqueness checks ===")

    def check_unique(df, col):
        if df[col].is_unique:
            print(f"✔ {col} unique")
        else:
            print(f"❌ {col} has duplicates")

    check_unique(customers, "customer_id")
    check_unique(products, "product_id")
    check_unique(orders, "order_id")
    check_unique(items, "order_item_id")

    print("\n=== Null checks ===")
    for name, df in [
        ("customers", customers),
        ("products", products),
        ("orders", orders),
        ("order_items", items),
    ]:
        nulls = df.isnull().sum()
        bad = nulls[nulls > 0]
        if bad.empty:
            print(f"✔ {name}: No nulls")
        else:
            print(f"❌ {name}: Nulls found\n{bad}")

    print("\n=== Validation complete ===")


if __name__ == "__main__":
    main()
