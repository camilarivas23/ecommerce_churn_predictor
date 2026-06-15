"""
E-Commerce Feature Engineering Module
Generate features for predicting user churn based on cart behavior and product engagement.
Features: Frequency + Magnitude only (no date-based features).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class EcommerceChurnFeatureEngineer:
    """Generate per-user features for churn prediction."""

    def __init__(self, carts_df: pd.DataFrame, products_df: pd.DataFrame,
                 users_df: pd.DataFrame, reference_date: datetime = None):
        self.carts_df = carts_df.copy()
        self.products_df = products_df.copy()
        self.users_df = users_df.copy()
        self.reference_date = reference_date or datetime.now()

        if 'date' in self.carts_df.columns:
            self.carts_df['date'] = pd.to_datetime(self.carts_df['date'], unit='ms', errors='coerce')

        self._prepare_cart_data()

    def _prepare_cart_data(self):
        """Flatten cart products and merge with product metadata."""
        rows = []
        for _, cart in self.carts_df.iterrows():
            user_id = cart.get('userId')
            cart_date = cart.get('date')
            for item in cart.get('products', []):
                product_id = item.get('id')
                qty = item.get('quantity', 1)
                prod = self.products_df[self.products_df['id'] == product_id]
                if len(prod) > 0:
                    prod = prod.iloc[0]
                    price = prod.get('price', 0)
                    discount = prod.get('discountPercentage', 0)
                    rating = prod.get('rating', 0)
                    category = prod.get('category', 'unknown')
                    brand = prod.get('brand', 'unknown')
                else:
                    price = item.get('price', 0)
                    discount = 0
                    rating = 0
                    category = 'unknown'
                    brand = 'unknown'

                discounted_price = price * (1 - discount / 100) if discount else price
                rows.append({
                    'user_id': user_id,
                    'cart_id': cart.get('id'),
                    'cart_date': cart_date,
                    'product_id': product_id,
                    'quantity': qty,
                    'price': price,
                    'discount_pct': discount,
                    'discounted_price': discounted_price,
                    'total_item_cost': discounted_price * qty,
                    'rating': rating,
                    'category': category,
                    'brand': brand,
                })

        self.cart_items_df = pd.DataFrame(rows)

    def _compute_recency_features(self, user_id, user_carts) -> dict:
        """Compute recency features from cart dates."""
        dates = user_carts['date'].dropna()
        if len(dates) == 0:
            return {
                'days_since_last_cart': 999,
                'is_active_last_7d': 0,
                'is_active_last_30d': 0,
            }
        last_date = dates.max()
        if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
            last_date = last_date.replace(tzinfo=None)
        days_since = (self.reference_date - last_date).days
        return {
            'days_since_last_cart': days_since,
            'is_active_last_7d': 1 if days_since <= 7 else 0,
            'is_active_last_30d': 1 if days_since <= 30 else 0,
        }

    def generate_all_features(self) -> pd.DataFrame:
        """Generate features across 3 domains: Recency, Frequency, Magnitude."""
        if self.cart_items_df.empty:
            return pd.DataFrame()

        user_ids = self.carts_df['userId'].unique()
        features_list = []

        for user_id in user_ids:
            user_carts = self.carts_df[self.carts_df['userId'] == user_id]
            user_items = self.cart_items_df[self.cart_items_df['user_id'] == user_id]
            if len(user_items) == 0:
                continue

            feature_dict = {'user_id': user_id}

            # ===== RECENCY FEATURES (time-based + binary) =====
            feature_dict.update(self._compute_recency_features(user_id, user_carts))

            # ===== FREQUENCY FEATURES (aggregation + ratio) =====
            feature_dict['total_orders'] = user_items['cart_id'].nunique()
            feature_dict['total_items_purchased'] = int(user_items['quantity'].sum())
            feature_dict['unique_products_bought'] = user_items['product_id'].nunique()
            feature_dict['avg_cart_size'] = round(user_items.groupby('cart_id')['quantity'].sum().mean(), 2) if feature_dict['total_orders'] > 0 else 0
            feature_dict['unique_categories_bought'] = user_items['category'].nunique()

            # ===== MAGNITUDE FEATURES (aggregation + ratio) =====
            feature_dict['total_spent'] = round(user_items['total_item_cost'].sum(), 2)
            feature_dict['avg_order_value'] = round(user_items.groupby('cart_id')['total_item_cost'].sum().mean(), 2) if feature_dict['total_orders'] > 0 else 0
            feature_dict['avg_product_rating'] = round(user_items['rating'].mean(), 2)
            feature_dict['avg_discount_pct'] = round(user_items['discount_pct'].mean(), 2)
            feature_dict['avg_price_per_item'] = round(user_items['price'].mean(), 2)

            features_list.append(feature_dict)

        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        return features_df

    def generate_churn_labels(self, features_df: pd.DataFrame,
                              threshold_days: int = 30) -> pd.DataFrame:
        """Generate churn labels from cart dates (for training only)."""
        labels = []
        for _, row in features_df.iterrows():
            user_carts = self.carts_df[self.carts_df['userId'] == row['user_id']]
            dates = user_carts['date'].dropna()
            if len(dates) == 0:
                churn = 1
            else:
                last_date = dates.max()
                if hasattr(last_date, 'tzinfo') and last_date.tzinfo is not None:
                    last_date = last_date.replace(tzinfo=None)
                days_since = (self.reference_date - last_date).days
                churn = 1 if days_since > threshold_days else 0
            labels.append({'user_id': row['user_id'], 'churn': churn})
        return pd.DataFrame(labels)
