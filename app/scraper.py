"""
DummyJSON API Client - E-Commerce Data Collection
Fetches products, carts, and users for churn prediction.
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests


class DummyJSONClient:
    """Client for DummyJSON API - fetches e-commerce data."""

    BASE_URL = "https://dummyjson.com"

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.environ.get("DATA_DIR", "data/raw")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        os.makedirs(data_dir, exist_ok=True)

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to fetch {url}: {e}")

    def fetch_products(self, limit: int = 0) -> List[Dict]:
        """Fetch all products from DummyJSON."""
        if limit > 0:
            data = self._get("/products", {"limit": limit, "select": "id,title,price,discountPercentage,stock,brand,category,rating,sku,weight,tags"})
            return data.get("products", [])
        all_products = []
        skip = 0
        while True:
            data = self._get("/products", {"limit": 100, "skip": skip, "select": "id,title,price,discountPercentage,stock,brand,category,rating,sku,weight,tags"})
            products = data.get("products", [])
            if not products:
                break
            all_products.extend(products)
            skip += len(products)
            if skip >= data.get("total", 0):
                break
        return all_products

    def fetch_carts(self) -> List[Dict]:
        """Fetch all carts from DummyJSON, remap to create multi-order users."""
        import random
        random.seed(42)
        all_carts = []
        skip = 0
        while True:
            data = self._get("/carts", {"limit": 100, "skip": skip})
            carts = data.get("carts", [])
            if not carts:
                break
            all_carts.extend(carts)
            skip += len(carts)
            if skip >= data.get("total", 0):
                break

        # DummyJSON gives 1 cart per user. Remap so users have 1-6 orders.
        user_ids = list(set(c["userId"] for c in all_carts))
        random.shuffle(user_ids)
        n_users = len(user_ids)

        # Assign order counts: some users get 1, some 2-6
        order_counts = {}
        for uid in user_ids:
            order_counts[uid] = random.choices([1, 2, 3, 4, 5, 6], weights=[30, 25, 20, 15, 7, 3], k=1)[0]

        # Build new cart list: duplicate carts across users to match their order count
        new_carts = []
        cart_id = 1
        now = datetime.now()
        for uid in user_ids:
            # Find original carts for this user (usually just 1)
            original_carts = [c for c in all_carts if c["userId"] == uid]
            n_orders = order_counts[uid]
            for i in range(n_orders):
                base = original_carts[i % len(original_carts)]
                cart = dict(base)
                cart["id"] = cart_id
                cart["userId"] = uid
                days_ago = random.choices(
                    range(0, 90),
                    weights=[15]*10 + [8]*20 + [3]*30 + [1]*30,
                    k=1
                )[0]
                cart["date"] = int((now - timedelta(days=days_ago)).timestamp() * 1000)
                new_carts.append(cart)
                cart_id += 1

        print(f"  Remapped to {len(new_carts)} carts across {n_users} users")
        return new_carts

    def fetch_users(self) -> List[Dict]:
        """Fetch all users from DummyJSON."""
        all_users = []
        skip = 0
        while True:
            data = self._get("/users", {"limit": 100, "skip": skip, "select": "id,firstName,lastName,email,age,gender,phone,birthDate,address,company"})
            users = data.get("users", [])
            if not users:
                break
            all_users.extend(users)
            skip += len(users)
            if skip >= data.get("total", 0):
                break
        return all_users

    def fetch_categories(self) -> List[Dict]:
        """Fetch all product categories."""
        return self._get("/products/category-list")

    def collect_all(self) -> Dict[str, List]:
        """Collect all data and save to disk."""
        print("Fetching products...")
        products = self.fetch_products()
        print(f"  Got {len(products)} products")

        print("Fetching carts...")
        carts = self.fetch_carts()
        print(f"  Got {len(carts)} carts")

        print("Fetching users...")
        users = self.fetch_users()
        print(f"  Got {len(users)} users")

        print("Fetching categories...")
        categories = self.fetch_categories()
        print(f"  Got {len(categories)} categories")

        self._save(products, carts, users, categories)
        return {"products": products, "carts": carts, "users": users, "categories": categories}

    def _save(self, products, carts, users, categories):
        """Save raw data to JSON files."""
        timestamp = datetime.now().isoformat()

        products_file = os.path.join(self.data_dir, "products.json")
        with open(products_file, "w") as f:
            json.dump({"products": products, "fetched_at": timestamp}, f, indent=2)

        carts_file = os.path.join(self.data_dir, "carts.json")
        with open(carts_file, "w") as f:
            json.dump({"carts": carts, "fetched_at": timestamp}, f, indent=2)

        users_file = os.path.join(self.data_dir, "users.json")
        with open(users_file, "w") as f:
            json.dump({"users": users, "fetched_at": timestamp}, f, indent=2)

        categories_file = os.path.join(self.data_dir, "categories.json")
        with open(categories_file, "w") as f:
            json.dump({"categories": categories, "fetched_at": timestamp}, f, indent=2)

        print(f"Saved all data to {self.data_dir}/")


def main():
    data_dir = os.environ.get("DATA_DIR", "data/raw")
    client = DummyJSONClient(data_dir=data_dir)
    client.collect_all()


if __name__ == "__main__":
    main()
