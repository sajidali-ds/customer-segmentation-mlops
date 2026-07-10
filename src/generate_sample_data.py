import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 800
N_TRANSACTIONS = 25000
COUNTRIES = ["United Kingdom", "Germany", "France", "Spain", "Netherlands",
             "Ireland", "Belgium", "Portugal", "Italy", "Australia"]

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.75),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 3.75),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22632", "HAND WARMER RED POLKA DOT", 1.85),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("21212", "PACK OF 72 RETRO SPOT CAKE CASES", 0.55),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
    ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
    ("22150", "3 STRIPEY MICE FELTCRAFT", 1.65),
]

# Give customers heterogeneous behavior profiles so clustering finds real structure
CUSTOMER_PROFILES = []
for cust_id in range(12000, 12000 + N_CUSTOMERS):
    profile_type = np.random.choice(
        ["champion", "loyal", "at_risk", "new", "lost"],
        p=[0.12, 0.28, 0.20, 0.20, 0.20]
    )
    country = np.random.choice(COUNTRIES, p=[0.55, 0.08, 0.08, 0.06, 0.06,
                                              0.05, 0.04, 0.03, 0.03, 0.02])
    CUSTOMER_PROFILES.append({"CustomerID": cust_id, "type": profile_type, "country": country})

profile_df = pd.DataFrame(CUSTOMER_PROFILES)

def days_ago_for_type(ptype):
    if ptype == "champion":
        return np.random.randint(0, 15)
    if ptype == "loyal":
        return np.random.randint(5, 40)
    if ptype == "at_risk":
        return np.random.randint(60, 150)
    if ptype == "new":
        return np.random.randint(0, 20)
    if ptype == "lost":
        return np.random.randint(180, 365)

def n_orders_for_type(ptype):
    if ptype == "champion":
        return np.random.randint(15, 40)
    if ptype == "loyal":
        return np.random.randint(8, 18)
    if ptype == "at_risk":
        return np.random.randint(3, 9)
    if ptype == "new":
        return np.random.randint(1, 3)
    if ptype == "lost":
        return np.random.randint(1, 4)

END_DATE = datetime(2024, 12, 9)
rows = []
invoice_counter = 536365

for _, cust in profile_df.iterrows():
    n_orders = n_orders_for_type(cust["type"])
    recency_days = days_ago_for_type(cust["type"])
    last_order_date = END_DATE - timedelta(days=int(recency_days))

    for order_i in range(n_orders):
        # Spread earlier orders further back in time
        order_date = last_order_date - timedelta(days=int(np.random.exponential(20) * (n_orders - order_i)))
        if order_date < datetime(2023, 12, 1):
            order_date = datetime(2023, 12, 1) + timedelta(days=np.random.randint(0, 30))

        invoice_counter += 1
        n_items = np.random.randint(1, 8)
        for _ in range(n_items):
            stock_code, desc, unit_price = random.choice(PRODUCTS)
            qty = np.random.randint(1, 12)
            # champions/loyal buy slightly higher unit prices & quantities on average
            if cust["type"] in ("champion", "loyal"):
                qty = int(qty * np.random.uniform(1.0, 1.8))
            rows.append({
                "InvoiceNo": str(invoice_counter),
                "StockCode": stock_code,
                "Description": desc,
                "Quantity": qty,
                "InvoiceDate": order_date.strftime("%m/%d/%Y %H:%M"),
                "UnitPrice": unit_price,
                "CustomerID": cust["CustomerID"],
                "Country": cust["country"],
            })

df = pd.DataFrame(rows)

# Inject a small amount of realistic noise: cancellations (invoice starting with 'C')
# and a few negative-quantity returns, exactly like the real dataset
n_cancel = int(len(df) * 0.02)
cancel_idx = np.random.choice(df.index, n_cancel, replace=False)
df.loc[cancel_idx, "Quantity"] = -df.loc[cancel_idx, "Quantity"].abs()
df.loc[cancel_idx, "InvoiceNo"] = "C" + df.loc[cancel_idx, "InvoiceNo"]

# A few missing CustomerIDs (also realistic in the real dataset)
n_missing = int(len(df) * 0.01)
missing_idx = np.random.choice(df.index, n_missing, replace=False)
df.loc[missing_idx, "CustomerID"] = np.nan

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

out_path = "data/raw/online_retail.csv"
df.to_csv(out_path, index=False)
print(f"Generated {len(df):,} transaction rows for {N_CUSTOMERS} customers -> {out_path}")
print(df.head())
