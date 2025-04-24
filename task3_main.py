
# === Task 3: System Integration & Interface ===

# --- Imports ---
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import math

# === Task 1 Logic: Frequent Itemset Mining ===

# === Task 1 Updated: Frequent Itemset Mining using FP-Growth ===
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

def get_frequent_patterns(user_id, df, min_support=0.02):
    """
    Generate frequent itemsets using FP-Growth for a given user.
    
    Args:
        user_id (str): ID of the user.
        df (DataFrame): Full transaction dataset.
        min_support (float): Minimum support threshold for frequent itemsets.
    
    Returns:
        DataFrame: Frequent itemsets for the given user.
    """
    user_data = df[df['User_id'] == user_id]
    if user_data.empty:
        return pd.DataFrame(columns=['itemsets', 'support'])
    
    # Group by transactions (e.g., by date)
    transactions = user_data.groupby('Date')['itemDescription'].apply(list).tolist()

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apply FP-Growth
    frequent_itemsets = fpgrowth(transaction_df, min_support=min_support, use_colnames=True)
    frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

    return frequent_itemsets


    item_counts = defaultdict(int)
    total_transactions = len(transactions)

    for items in transactions:
        unique_items = set(items)
        for combo in combinations(unique_items, 2):
            item_counts[tuple(sorted(combo))] += 1

    # Convert to DataFrame
    patterns = pd.DataFrame([
        {'items': k, 'count': v, 'support': v / total_transactions}
        for k, v in item_counts.items() if v / total_transactions >= min_support
    ])
    patterns.sort_values(by='support', ascending=False, inplace=True)

    return patterns

# === Task 2 Logic: Collaborative Filtering Recommender ===
class CollaborativeFilteringRecommender:
    def __init__(self, recency_weighting=True, decay_factor=0.95):
        self.user_item_data = None
        self.user_similarity_matrix = None
        self.user_map = {}
        self.reverse_user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        self.user_item_matrix = None
        self.recency_weighting = recency_weighting
        self.decay_factor = decay_factor
        self.fitted = False

    def fit(self, df):
        user_item_dict = defaultdict(dict)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        for _, row in df.iterrows():
            user, item, date = row['User_id'], row['itemDescription'], row['Date']
            timestamp = date.timestamp()
            weight = self.decay_factor ** ((df['Date'].max() - date).days) if self.recency_weighting else 1
            user_item_dict[user][item] = user_item_dict[user].get(item, 0) + weight

        self.user_map = {user: idx for idx, user in enumerate(user_item_dict)}
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}

        item_set = set()
        for items in user_item_dict.values():
            item_set.update(items.keys())
        self.item_map = {item: idx for idx, item in enumerate(item_set)}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}

        user_item_matrix = np.zeros((len(self.user_map), len(self.item_map)))

        for user, items in user_item_dict.items():
            for item, score in items.items():
                user_item_matrix[self.user_map[user]][self.item_map[item]] = score

        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = cosine_similarity(user_item_matrix)
        self.fitted = True

    def recommend(self, user_id, top_n=5, exclude_items=None):
        if user_id not in self.user_map:
            return []

        user_idx = self.user_map[user_id]
        user_similarities = self.user_similarity_matrix[user_idx]
        scores = user_similarities @ self.user_item_matrix

        user_items = set(np.where(self.user_item_matrix[user_idx] > 0)[0])
        if exclude_items:
            user_items.update(self.item_map.get(i) for i in exclude_items if i in self.item_map)

        scores[user_items] = 0  # mask seen items
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.reverse_item_map[idx] for idx in top_indices if scores[idx] > 0]

# === Task 3 Logic: User Interface and Integration ===

def get_user_input():
    user_id = input("Enter user_id: ").strip()
    method = input("Enter recommendation method ('with' or 'without'): ").strip().lower()
    while method not in ['with', 'without']:
        print("Invalid option. Please enter 'with' or 'without'.")
        method = input("Enter recommendation method ('with' or 'without'): ").strip().lower()
    return user_id, method

def run_interface(data):
    print("Welcome to the Grocery Recommender System!")

    # Initialize model
    recommender = CollaborativeFilteringRecommender()
    recommender.fit(data)

    while True:
        user_id, method = get_user_input()
        if user_id not in data['User_id'].values:
            print(f"User {user_id} not found. Showing popular items.")
            popular_items = data['itemDescription'].value_counts().head(5).index.tolist()
            print("Recommended Items:", popular_items)
        else:
            history = data[data['User_id'] == user_id]['itemDescription'].unique().tolist()
            if method == 'with':
                patterns = get_frequent_patterns(user_id, data)
                candidate_items = set()
                for items in patterns['items']:
                    candidate_items.update(items)
                candidate_items = candidate_items - set(history)
                print(f"Top items from frequent patterns: {list(candidate_items)[:5]}")
            else:
                recs = recommender.recommend(user_id, exclude_items=history)
                print(f"Top 5 Recommendations for user {user_id}:")
                for i, item in enumerate(recs, 1):
                    print(f"{i}. {item}")

        cont = input("Do you want to continue? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Thank you for using the system. Goodbye!")
            break

# === Main Entry Point ===
if __name__ == "__main__":
    data = pd.read_csv("Groceries data train.csv")
    run_interface(data)
