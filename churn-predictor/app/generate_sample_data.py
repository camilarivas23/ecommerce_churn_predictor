"""
Generate Sample E-Commerce Data
Produces synthetic products, carts, and users for training when DummyJSON is unavailable.
"""
import json
import os
import random
from datetime import datetime, timedelta


def generate_sample_data(output_dir: str, n_users: int = 50, n_carts: int = 500):
    """Generate sample e-commerce data."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    categories = ["smartphones", "laptops", "fragrances", "groceries", "home-decoration",
                   "furniture", "sunglasses", "automotive", "motorcycle", "skin-care",
                   "sports-accessories", "sports-accessories", "tops", "womens-dresses",
                   "womens-shoes", "mens-shirts", "mens-shoes", "mens-watches",
                   "womens-watches", "womens-bags", "womens-jewellery"]

    brands = ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG", "HP", "Dell",
              "Calvin Klein", "Coach", "Gucci", "Prada", "Versace", "Zara", "H&M"]

    # Generate products
    products = []
    for i in range(100):
        cat = random.choice(categories)
        brand = random.choice(brands)
        price = round(random.uniform(10, 500), 2)
        discount = round(random.uniform(0, 30), 1)
        rating = round(random.uniform(3.0, 5.0), 1)
        products.append({
            "id": i + 1,
            "title": f"{brand} {cat.title()} Item {i+1}",
            "price": price,
            "discountPercentage": discount,
            "stock": random.randint(10, 200),
            "brand": brand,
            "category": cat,
            "rating": rating,
            "sku": f"SKU-{i+1:04d}",
            "weight": round(random.uniform(0.1, 5.0), 1),
            "tags": [cat, brand.lower()],
        })

    # Generate users
    users = []
    for uid in range(n_users):
        created_days_ago = random.randint(30, 730)
        users.append({
            "id": uid + 1,
            "firstName": f"User{uid+1}",
            "lastName": f"Test",
            "email": f"user{uid+1}@example.com",
            "age": random.randint(18, 65),
            "gender": random.choice(["male", "female"]),
            "phone": f"+1-555-{uid+1:04d}",
            "birthDate": f"1990-01-{(uid % 28) + 1:02d}",
            "address": {"address": f"{uid+1} Main St", "city": "Anytown", "state": "CA", "postalCode": "90210"},
            "company": {"name": f"Company {uid+1}", "department": "Engineering"},
        })

    # Generate carts
    reference_date = datetime.now()
    carts = []
    # Assign some users to be churned (only old carts)
    churned_user_ids = random.sample([u["id"] for u in users], k=n_users // 3)

    for cart_id in range(n_carts):
        user = random.choice(users)
        n_items = random.randint(1, 8)
        cart_products = random.sample(products, min(n_items, len(products)))
        items = []
        for p in cart_products:
            items.append({
                "id": p["id"],
                "title": p["title"],
                "price": p["price"],
                "quantity": random.randint(1, 5),
                "total": round(p["price"] * random.randint(1, 5), 2),
                "discountPercentage": p["discountPercentage"],
                "discountedPrice": round(p["price"] * (1 - p["discountPercentage"] / 100), 2),
            })

        # Spread carts over time (some recent, some old)
        # Heavier weight on old days to create churned users
        if user["id"] in churned_user_ids:
            days_ago = random.randint(40, 120)
        else:
            days_ago = random.choices(
                range(0, 30),
                weights=[15] * 10 + [8] * 20,
                k=1
            )[0]
        cart_date = reference_date - timedelta(days=days_ago)

        carts.append({
            "id": cart_id + 1,
            "products": items,
            "total": round(sum(i["total"] for i in items), 2),
            "discountedTotal": round(sum(i["discountedPrice"] * i["quantity"] for i in items), 2),
            "userId": user["id"],
            "totalProducts": len(items),
            "totalQuantity": sum(i["quantity"] for i in items),
            "date": int(cart_date.timestamp() * 1000),
        })

    # Save
    ts = datetime.now().isoformat()

    with open(os.path.join(output_dir, "products.json"), "w") as f:
        json.dump({"products": products, "fetched_at": ts}, f, indent=2)

    with open(os.path.join(output_dir, "users.json"), "w") as f:
        json.dump({"users": users, "fetched_at": ts}, f, indent=2)

    with open(os.path.join(output_dir, "carts.json"), "w") as f:
        json.dump({"carts": carts, "fetched_at": ts}, f, indent=2)

    with open(os.path.join(output_dir, "categories.json"), "w") as f:
        json.dump({"categories": categories, "fetched_at": ts}, f, indent=2)

    print(f"Generated: {len(products)} products, {len(users)} users, {len(carts)} carts")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    generate_sample_data("data/raw")
