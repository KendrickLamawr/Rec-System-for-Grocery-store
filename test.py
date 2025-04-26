import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# Load data
train_path = 'Groceries data train.csv'
test_path = 'Groceries data test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Rename column to ensure consistency
train_df = train_df.rename(columns={"User_id": "user_id"})

# Convert dates to datetime format
train_df["Date"] = pd.to_datetime(train_df["Date"], format='%d/%m/%Y')
test_df["Date"] = pd.to_datetime(test_df["Date"], format='%d/%m/%Y')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nColumns:", list(train_df.columns))
print("\nMissing values (train):\n", train_df.isna().sum())
print("\nMissing values (test):\n", test_df.isna().sum())

# Basic EDA on daily transactions
train_daily = train_df.groupby('Date').size().reset_index(name='Transactions')
test_daily = test_df.groupby('Date').size().reset_index(name='Transactions')

plt.figure(figsize=(14, 5))
plt.plot(train_daily['Date'], train_daily['Transactions'], label='Train set')
plt.plot(test_daily['Date'], test_daily['Transactions'], label='Test set')

plt.title('Daily Transactions: Train vs Test Sets')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.legend()
plt.axvline(x=test_daily['Date'].min(), color='r', linestyle='--', label='Train/Test Split')
plt.tight_layout()
plt.show()

# Get the most frequent items
top20_train = train_df["itemDescription"].value_counts().head(20)
top20_test = test_df["itemDescription"].value_counts().head(20)

fig, ax = plt.subplots(figsize=(8,6))
top20_train[::-1].plot(kind="barh", ax=ax)
ax.set_title("Top 20 items in TRAIN set")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
top20_test[::-1].plot(kind="barh", ax=ax, color="grey")
ax.set_title("Top 20 items in TEST set")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.show()

# Count unique users
n_users_train = train_df["user_id"].nunique()
n_users_test = test_df["user_id"].nunique()
print(f"Unique users in train: {n_users_train:,}")
print(f"Unique users in test: {n_users_test:,}")

# Distribution of items per user
items_per_user = train_df.groupby("user_id")["itemDescription"].count()

plt.figure(figsize=(8,4))
plt.hist(items_per_user, bins=30)
plt.title("Distribution of #items per user (TRAIN)")
plt.xlabel("#items across the year")
plt.ylabel("Users")
plt.tight_layout()
plt.show()

# Day of week analysis
dow_map = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
train_dow = train_df["day_of_week"].value_counts().sort_index()
test_dow = test_df["day_of_week"].value_counts().sort_index()

x = np.arange(7)
width = 0.35
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(x - width/2, train_dow.values, width, label="train")
ax.bar(x + width/2, test_dow.values, width, label="test", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(dow_map)
ax.set_title("Items sold by day of week")
ax.set_ylabel("Item count")
ax.legend()
plt.tight_layout()
plt.show()

# Create baskets (purchases by user and date)
train_baskets = (
    train_df
      .groupby(["user_id", "Date"])
      .size()
      .reset_index(name="items_in_basket")
)

# Distribution of baskets per user
baskets_per_user = train_baskets.groupby("user_id").size()

plt.figure(figsize=(8,4))
plt.hist(baskets_per_user, bins=30, color="#F79F24")
plt.title("Distribution of #baskets per user (TRAIN)")
plt.xlabel("#baskets across the year")
plt.ylabel("Users")
plt.tight_layout()
plt.show()

# Count unique users and items
unique_users = train_df['user_id'].unique()
unique_items = train_df['itemDescription'].unique()
print("Number of unique users:", len(unique_users))
print("Number of unique items:", len(unique_items))

# Date ranges
min_date_train = train_df['Date'].min()
max_date_train = train_df['Date'].max()
print(f"Training data spans from {min_date_train.strftime('%Y-%m-%d')} to {max_date_train.strftime('%Y-%m-%d')}")
print(f"Total days in training: {(max_date_train - min_date_train).days + 1}")

min_date_test = test_df['Date'].min()
max_date_test = test_df['Date'].max()
print(f"Test data spans from {min_date_test.strftime('%Y-%m-%d')} to {max_date_test.strftime('%Y-%m-%d')}")
print(f"Total days in test: {(max_date_test - min_date_test).days + 1}")

#################################################################
#                  TASK 1: PATTERN MINING                       #
#################################################################

def create_transactions_df(df):
    """
    Convert raw purchase data into transaction format suitable for pattern mining.
    Each row represents a single transaction (user-date combination) with items purchased.
    
    Args:
        df (DataFrame): Raw purchase data with user_id, Date, and itemDescription
        
    Returns:
        DataFrame: Transaction data in basket format
    """
    # Group by user_id and Date to create baskets
    transactions = df.groupby(['user_id', 'Date'])['itemDescription'].apply(list).reset_index()
    return transactions

def create_one_hot_encoded(transactions):
    """
    Convert transaction data into one-hot encoded format for Apriori algorithm.
    
    Args:
        transactions (DataFrame): Transaction data with lists of items
        
    Returns:
        DataFrame: One-hot encoded transaction data
    """
    # Create dictionary of transaction items
    transaction_dict = {}
    
    for idx, row in transactions.iterrows():
        user_id = row['user_id']
        date = row['Date']
        items = row['itemDescription']
        
        # Create unique transaction ID
        transaction_id = f"{user_id}_{date}"
        transaction_dict[transaction_id] = items
    
    # Get all unique items
    all_items = set()
    for items in transaction_dict.values():
        all_items.update(items)
    
    # Create one-hot encoded DataFrame
    encoded_df = pd.DataFrame(index=transaction_dict.keys(), columns=list(all_items), dtype=int)
    encoded_df = encoded_df.fillna(0)
    
    # Fill the DataFrame
    for transaction_id, items in transaction_dict.items():
        for item in items:
            encoded_df.at[transaction_id, item] = 1
    
    return encoded_df

def mine_frequent_patterns(transactions_encoded, min_support=0.01, use_colnames=True):
    """
    Mine frequent itemsets using the Apriori algorithm.
    
    Args:
        transactions_encoded (DataFrame): One-hot encoded transaction data
        min_support (float): Minimum support threshold
        use_colnames (bool): Whether to use column names in the result
        
    Returns:
        DataFrame: Frequent itemsets with support
    """
    print(f"Mining frequent patterns with min_support={min_support}...")
    start_time = time.time()
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(
        transactions_encoded,
        min_support=min_support,
        use_colnames=use_colnames,
        verbose=1
    )
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets in {time.time() - start_time:.2f} seconds")
    return frequent_itemsets

def generate_association_rules(frequent_itemsets, min_confidence=0.5, min_lift=1.0):
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets (DataFrame): Frequent itemsets from Apriori
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        
    Returns:
        DataFrame: Association rules with metrics
    """
    print(f"Generating association rules with min_confidence={min_confidence}, min_lift={min_lift}...")
    
    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    print(f"Generated {len(rules)} rules")
    return rules

def score_patterns_for_user(user_id, rules, user_history):
    """
    Score pattern importance/quality for a specific user.
    
    Args:
        user_id (int): User ID
        rules (DataFrame): Association rules
        user_history (list): List of items in user's purchase history
        
    Returns:
        DataFrame: Scored recommendations for the user
    """
    # Filter rules where antecedents (left side) are in user history
    user_rules = []
    
    for _, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        
        # Check if all antecedent items are in user history
        if all(item in user_history for item in antecedents):
            # Check if consequent items are not in user history
            new_items = [item for item in consequents if item not in user_history]
            
            if new_items:  # If there are new items to recommend
                for item in new_items:
                    user_rules.append({
                        'item': item,
                        'confidence': rule['confidence'],
                        'lift': rule['lift'],
                        'support': rule['support'],
                        # Custom score combining metrics - can be adjusted
                        'score': 0.4 * rule['confidence'] + 0.4 * rule['lift'] + 0.2 * rule['support']
                    })
    
    # Convert to DataFrame and sort by score
    if user_rules:
        recommendations = pd.DataFrame(user_rules)
        # Aggregate by item taking the max score
        recommendations = recommendations.groupby('item').agg({
            'confidence': 'max',
            'lift': 'max',
            'support': 'max',
            'score': 'max'
        }).reset_index()
        
        recommendations = recommendations.sort_values('score', ascending=False)
        return recommendations
    else:
        return pd.DataFrame(columns=['item', 'confidence', 'lift', 'support', 'score'])

def get_user_purchase_history(df, user_id):
    """
    Get purchase history for a specific user.
    
    Args:
        df (DataFrame): Purchase data
        user_id (int): User ID
        
    Returns:
        list: List of unique items in user's purchase history
    """
    # Filter by user_id and get unique item descriptions
    user_items = df[df['user_id'] == user_id]['itemDescription'].unique().tolist()
    return user_items

def recency_weighted_patterns(df, user_id, rules, decay_factor=0.01):
    """
    Apply recency weighting to patterns for a user.
    Higher scores for patterns based on more recent purchases.
    
    Args:
        df (DataFrame): Purchase data with timestamps
        user_id (int): User ID
        rules (DataFrame): Association rules
        decay_factor (float): Decay factor for time weighting
        
    Returns:
        DataFrame: Recency-weighted scored recommendations
    """
    # Get user's purchase history with timestamps
    user_df = df[df['user_id'] == user_id].copy()
    
    if user_df.empty:
        return pd.DataFrame(columns=['item', 'score'])
    
    # Get the most recent purchase date
    most_recent_date = user_df['Date'].max()
    
    # Calculate days difference for each purchase
    user_df['days_diff'] = (most_recent_date - user_df['Date']).dt.days
    
    # Apply exponential decay based on time
    user_df['weight'] = np.exp(-decay_factor * user_df['days_diff'])
    
    # Group by item and calculate weighted importance
    item_weights = user_df.groupby('itemDescription')['weight'].sum().reset_index()
    item_weights = dict(zip(item_weights['itemDescription'], item_weights['weight']))
    
    # Get user's history
    user_history = user_df['itemDescription'].unique().tolist()
    
    # Get base recommendations
    recommendations = score_patterns_for_user(user_id, rules, user_history)
    
    if recommendations.empty:
        return recommendations
    
    # Apply additional weighting based on purchase recency
    for idx, row in recommendations.iterrows():
        item = row['item']
        # Adjust score based on recency weights of related items
        # This is a heuristic approach that can be refined
        related_items_weight = 0
        
        # Find rules where this item is in consequents
        for _, rule in rules.iterrows():
            if item in rule['consequents']:
                antecedents = list(rule['antecedents'])
                for ant_item in antecedents:
                    if ant_item in item_weights:
                        related_items_weight += item_weights[ant_item] * rule['confidence']
        
        # Update score with recency weight
        if related_items_weight > 0:
            recommendations.at[idx, 'recency_score'] = related_items_weight
            recommendations.at[idx, 'final_score'] = 0.7 * row['score'] + 0.3 * related_items_weight
        else:
            recommendations.at[idx, 'recency_score'] = 0
            recommendations.at[idx, 'final_score'] = row['score']
    
    # Sort by final score
    recommendations = recommendations.sort_values('final_score', ascending=False)
    return recommendations

def process_pattern_mining(train_df, min_support=0.001, min_confidence=0.2, min_lift=1.0):
    """
    Execute the full pattern mining process.
    
    Args:
        train_df (DataFrame): Training data
        min_support (float): Minimum support threshold
        min_confidence (float): Minimum confidence threshold
        min_lift (float): Minimum lift threshold
        
    Returns:
        tuple: (transactions, frequent_itemsets, rules)
    """
    # Create transaction data
    transactions = create_transactions_df(train_df)
    print(f"Created {len(transactions)} transactions")
    
    # Create one-hot encoded format for Apriori
    transactions_encoded = create_one_hot_encoded(transactions)
    print(f"Created one-hot encoded matrix with shape {transactions_encoded.shape}")
    
    # Mine frequent patterns
    frequent_itemsets = mine_frequent_patterns(transactions_encoded, min_support)
    
    # Generate association rules
    rules = generate_association_rules(frequent_itemsets, min_confidence, min_lift)
    
    return transactions, frequent_itemsets, rules

#################################################################
#            COLLABORATIVE FILTERING INTEGRATION                #
#################################################################

def compute_time_weighted_user_item_matrix(df, half_life_days=180):
    """
    Compute user-item matrix with time decay weighting.
    
    Args:
        df (DataFrame): Purchase data
        half_life_days (int): Half-life in days for time decay
        
    Returns:
        DataFrame: User-item matrix with time decay weights
    """
    # Calculate decay constant
    lambda_val = np.log(2) / half_life_days
    
    # Get maximum date
    max_date = df['Date'].max()
    
    # Calculate days difference
    days_diff = (max_date - df['Date']).dt.days
    
    # Calculate weights
    df = df.copy()
    df['weight'] = np.exp(-lambda_val * days_diff)
    
    # Create user-item matrix
    user_item_matrix = pd.pivot_table(
        df,
        values='weight',
        index='user_id',
        columns='itemDescription',
        aggfunc='sum',
        fill_value=0
    )
    
    return user_item_matrix

def compute_similarities(matrix, method='user'):
    """
    Compute similarity matrix for users or items.
    
    Args:
        matrix (DataFrame): User-item matrix
        method (str): 'user' for user-based or 'item' for item-based
        
    Returns:
        DataFrame: Similarity matrix
    """
    if method == 'user':
        sim_matrix = cosine_similarity(matrix)
        return pd.DataFrame(sim_matrix, index=matrix.index, columns=matrix.index)
    else:  # item-item
        sim_matrix = cosine_similarity(matrix.T)
        return pd.DataFrame(sim_matrix, index=matrix.columns, columns=matrix.columns)

def generate_cf_recommendations(user_item_matrix, sim_matrix, user_id, method='user', top_n=5):
    """
    Generate recommendations using collaborative filtering.
    
    Args:
        user_item_matrix (DataFrame): User-item matrix
        sim_matrix (DataFrame): Similarity matrix
        user_id (int): User ID
        method (str): 'user' for user-based or 'item' for item-based
        top_n (int): Number of recommendations to generate
        
    Returns:
        Series: Top N recommended items with scores
    """
    if user_id not in user_item_matrix.index:
        return None
    
    if method == 'user':
        # Get similar users
        similar_scores = sim_matrix.loc[user_id]
        similar_users = similar_scores.nlargest(top_n + 1).index[1:]
        
        # Get items bought by similar users
        user_items = user_item_matrix.loc[similar_users]
        
        # Weight items by user similarity
        similar_user_weights = similar_scores[similar_users].values.reshape(-1, 1)
        
        # Calculate the sum of weights for normalization
        weight_sum = similar_user_weights.sum()
        
        # Calculate weighted mean by dividing by sum of weights
        weighted_items = pd.Series(
            (user_items.values * similar_user_weights).sum(axis=0) / weight_sum,
            index=user_item_matrix.columns
        )
        
        # Remove items already bought by the user
        user_items = user_item_matrix.loc[user_id]
        weighted_items[user_items > 0] = 0
        
        return weighted_items.nlargest(top_n)
    
    else:  # method == 'item'
        user_vec = user_item_matrix.loc[user_id]
        bought_items = user_vec[user_vec > 0]  # index = purchased items
        candidates = user_vec[user_vec == 0].index  # unseen items
        
        scores = {}
        
        for target in candidates:
            # similarities between target item and each purchased item
            sims = sim_matrix.loc[target, bought_items.index]
            # keep top-k positive similarities
            sims_top = sims[sims > 0].sort_values(ascending=False).head(top_n)
            if sims_top.empty:
                continue
                
            sim_vals = sims_top.values
            purchase_w = bought_items[sims_top.index].values
            
            # similarity-weighted score
            scores[target] = np.dot(sim_vals, purchase_w) / sim_vals.sum()
        
        if not scores:
            return None
        
        return pd.Series(scores).sort_values(ascending=False).head(top_n)

def generate_hybrid_recommendations(user_id, user_item_matrix, sim_matrix_item, rules, 
                                   train_df, alpha=0.5, top_n=5, cf_method='item'):
    """
    Generate hybrid recommendations combining collaborative filtering and association rules.
    
    Args:
        user_id (int): User ID
        user_item_matrix (DataFrame): User-item matrix
        sim_matrix_item (DataFrame): Item similarity matrix
        rules (DataFrame): Association rules
        train_df (DataFrame): Training data
        alpha (float): Weight for CF recommendations (1-alpha for association rules)
        top_n (int): Number of recommendations to generate
        cf_method (str): 'user' or 'item' for CF method
        
    Returns:
        DataFrame: Top N hybrid recommendations with scores
    """
    # Handle users with no purchase history
    if user_id not in user_item_matrix.index:
        # For new users, recommend most popular items
        popular_items = train_df['itemDescription'].value_counts().head(top_n)
        return pd.DataFrame({
            'item': popular_items.index,
            'score': popular_items.values / popular_items.values.max(),
            'source': 'popularity'
        })
    
    # Get user's purchase history
    user_history = get_user_purchase_history(train_df, user_id)
    
    # Get CF recommendations
    cf_recs = generate_cf_recommendations(
        user_item_matrix, 
        sim_matrix_item,
        user_id,
        method=cf_method,
        top_n=top_n*2  # Get more to allow for overlap/filtering
    )
    
    # If CF fails, use only association rules
    if cf_recs is None or cf_recs.empty:
        cf_items = []
        cf_scores = {}
    else:
        cf_items = cf_recs.index.tolist()
        cf_scores = cf_recs.to_dict()
    
    # Get association rule recommendations
    pattern_recs = recency_weighted_patterns(train_df, user_id, rules)
    
    # If pattern mining fails, use only CF
    if pattern_recs.empty:
        pattern_items = []
        pattern_scores = {}
    else:
        pattern_items = pattern_recs['item'].tolist()
        pattern_scores = dict(zip(pattern_recs['item'], pattern_recs['final_score']))
    
    # Combine recommendations
    all_items = set(cf_items) | set(pattern_items)
    
    hybrid_scores = []
    for item in all_items:
        cf_score = cf_scores.get(item, 0)
        pattern_score = pattern_scores.get(item, 0)
        
        # Normalize scores (optional if scores are already normalized)
        if cf_score > 0:
            cf_score = cf_score / max(cf_scores.values()) if cf_scores else 0
        if pattern_score > 0:
            pattern_score = pattern_score / max(pattern_scores.values()) if pattern_scores else 0
        
        # Calculate hybrid score
        hybrid_score = alpha * cf_score + (1 - alpha) * pattern_score
        
        # Determine source
        if cf_score > 0 and pattern_score > 0:
            source = 'hybrid'
        elif cf_score > 0:
            source = 'cf'
        else:
            source = 'pattern'
        
        hybrid_scores.append({
            'item': item,
            'score': hybrid_score,
            'cf_score': cf_score,
            'pattern_score': pattern_score,
            'source': source
        })
    
    # Convert to DataFrame and sort
    hybrid_df = pd.DataFrame(hybrid_scores)
    hybrid_df = hybrid_df.sort_values('score', ascending=False).head(top_n)
    
    return hybrid_df

#################################################################
#                 EVALUATION METRICS                            #
#################################################################

def create_evaluation_data(train_df, test_df, split_date=None, dev_ratio=0.2):
    """
    Create evaluation data by splitting training data or using test data.
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Test data
        split_date (datetime): Date to split training data (None to use ratio)
        dev_ratio (float): Ratio for dev set if split_date is None
        
    Returns:
        tuple: (train_subset, validation_data)
    """
    if split_date is None:
        # Use a time-based split instead of random split to maintain temporal integrity
        max_date = train_df['Date'].max()
        min_date = train_df['Date'].min()
        total_days = (max_date - min_date).days
        split_days = int(total_days * (1 - dev_ratio))
        split_date = min_date + timedelta(days=split_days)
    
    # Split based on date
    train_subset = train_df[train_df['Date'] < split_date].copy()
    validation_data = train_df[train_df['Date'] >= split_date].copy()
    
    return train_subset, validation_data

def get_actual_purchases(df, user_id, min_date=None):
    """
    Get actual purchases for a user after a specified date.
    
    Args:
        df (DataFrame): Purchase data
        user_id (int): User ID
        min_date (datetime): Minimum date to consider purchases
        
    Returns:
        list: Items purchased by the user
    """
    if min_date is not None:
        user_purchases = df[(df['user_id'] == user_id) & (df['Date'] >= min_date)]['itemDescription'].unique().tolist()
    else:
        user_purchases = df[df['user_id'] == user_id]['itemDescription'].unique().tolist()
    
    return user_purchases

def precision_at_k(recommended_items, actual_items, k=5):
    """
    Calculate precision@k.
    
    Args:
        recommended_items (list): List of recommended items
        actual_items (list): List of actual items purchased
        k (int): Number of top recommendations to consider
        
    Returns:
        float: Precision@k
    """
    if not recommended_items or not actual_items:
        return 0.0
    
    # Take top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    
    # Count relevant items
    relevant = [item for item in recommended_items if item in actual_items]
    
    return len(relevant) / len(recommended_items) if recommended_items else 0.0

def recall_at_k(recommended_items, actual_items, k=5):
    """
    Calculate recall@k.
    
    Args:
        recommended_items (list): List of recommended items
        actual_items (list): List of actual items purchased
        k (int): Number of top recommendations to consider
        
    Returns:
        float: Recall@k
    """
    if not recommended_items or not actual_items:
        return 0.0
    
    # Take top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    
    # Count relevant items
    relevant = [item for item in recommended_items if item in actual_items]
    
    return len(relevant) / len(actual_items) if actual_items else 0.0

def hit_rate_at_k(recommended_items, actual_items, k=5):
    """
    Calculate hit rate@k (1 if at least one recommendation is relevant, 0 otherwise).
    
    Args:
        recommended_items (list): List of recommended items
        actual_items (list): List of actual items purchased
        k (int): Number of top recommendations to consider
        
    Returns:
        float: Hit rate@k (1.0 or 0.0)
    """
    if not recommended_items or not actual_items:
        return 0.0
    
    # Take top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    
    # Check if any recommended item is in actual items
    for item in recommended_items:
        if item in actual_items:
            return 1.0
    
    return 0.0

def ndcg_at_k(recommended_items, actual_items, k=5):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        recommended_items (list): List of recommended items
        actual_items (list): List of actual items purchased
        k (int): Number of top recommendations to consider
        
    Returns:
        float: NDCG@k
    """
    if not recommended_items or not actual_items:
        return 0.0
    
    # Take top k recommendations
    if len(recommended_items) > k:
        recommended_items = recommended_items[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in actual_items:
            # Use binary relevance (1 if relevant, 0 if not)
            rel = 1.0
            # Position in list is i+1 (1-indexed)
            dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Calculate ideal DCG (IDCG)
    idcg = 0.0
    # Ideal ranking would have all relevant items at the top
    n_rel = min(len(actual_items), k)
    for i in range(n_rel):
        idcg += 1.0 / np.log2(i + 2)
    
    # Return NDCG
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_recommendations(users, recommendations_func, validation_data, train_data=None, k=5, verbose=True):
    """
    Evaluate recommendations using various metrics.
    
    Args:
        users (list): List of user IDs to evaluate
        recommendations_func (function): Function to generate recommendations
        validation_data (DataFrame): Validation data with actual purchases
        train_data (DataFrame): Training data used for recommendations (optional)
        k (int): Number of recommendations to evaluate
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Evaluation metrics
    """
    precision_sum = 0.0
    recall_sum = 0.0
    hit_rate_sum = 0.0
    ndcg_sum = 0.0
    
    # Keep track of users with recommendations
    users_with_recs = 0
    
    # Minimum date for validation purchases
    min_date = None
    if train_data is not None:
        min_date = train_data['Date'].max()
    
    # Track per-user metrics
    all_metrics = []
    
    for user_id in tqdm(users, desc="Evaluating users"):
        # Get recommendations
        try:
            recommendations = recommendations_func(user_id)
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue
            
        if recommendations is None or len(recommendations) == 0:
            continue
        
        # Convert to list of items if DataFrame
        if isinstance(recommendations, pd.DataFrame):
            recommended_items = recommendations['item'].tolist()
        elif isinstance(recommendations, pd.Series):
            recommended_items = recommendations.index.tolist()
        else:
            recommended_items = recommendations
        
        # Get actual purchases
        actual_items = get_actual_purchases(validation_data, user_id, min_date)
        
        if not actual_items:
            continue
        
        # Calculate metrics
        precision = precision_at_k(recommended_items, actual_items, k)
        recall = recall_at_k(recommended_items, actual_items, k)
        hit_rate = hit_rate_at_k(recommended_items, actual_items, k)
        ndcg = ndcg_at_k(recommended_items, actual_items, k)
        
        # Store metrics
        all_metrics.append({
            'user_id': user_id,
            'precision': precision,
            'recall': recall,
            'hit_rate': hit_rate,
            'ndcg': ndcg,
            'n_recs': len(recommended_items),
            'n_actual': len(actual_items)
        })
        
        # Update sums
        precision_sum += precision
        recall_sum += recall
        hit_rate_sum += hit_rate
        ndcg_sum += ndcg
        users_with_recs += 1
    
    # Calculate averages
    if users_with_recs > 0:
        precision_avg = precision_sum / users_with_recs
        recall_avg = recall_sum / users_with_recs
        hit_rate_avg = hit_rate_sum / users_with_recs
        ndcg_avg = ndcg_sum / users_with_recs
    else:
        precision_avg = recall_avg = hit_rate_avg = ndcg_avg = 0.0
    
    if verbose:
        print(f"Evaluated {users_with_recs} users with recommendations")
        print(f"Precision@{k}: {precision_avg:.4f}")
        print(f"Recall@{k}: {recall_avg:.4f}")
        print(f"Hit Rate@{k}: {hit_rate_avg:.4f}")
        print(f"NDCG@{k}: {ndcg_avg:.4f}")
    
    return {
        'precision': precision_avg,
        'recall': recall_avg,
        'hit_rate': hit_rate_avg,
        'ndcg': ndcg_avg,
        'users_evaluated': users_with_recs,
        'all_metrics': pd.DataFrame(all_metrics)
    }

#################################################################
#           HYPERPARAMETER TUNING AND MAIN LOGIC                #
#################################################################

def tune_half_life_lambda(train_df, half_life_values, min_support=0.001, min_confidence=0.2):
    """
    Tune the half-life lambda parameter for time decay.
    
    Args:
        train_df (DataFrame): Training data
        half_life_values (list): Half-life values to test (in days)
        min_support (float): Minimum support for pattern mining
        min_confidence (float): Minimum confidence for association rules
        
    Returns:
        tuple: (best_half_life, evaluation_results)
    """
    print("Tuning half-life parameter...")
    
    # Create train/validation split
    train_subset, validation_data = create_evaluation_data(train_df, None, dev_ratio=0.2)
    
    # Process pattern mining (same for all lambda values)
    _, _, rules = process_pattern_mining(train_subset, min_support, min_confidence)
    
    # Evaluate each half-life value
    results = []
    
    for half_life in half_life_values:
        print(f"\nEvaluating half-life = {half_life} days")
        
        # Compute user-item matrix with this half-life
        user_item_matrix = compute_time_weighted_user_item_matrix(train_subset, half_life)
        
        # Compute similarity matrices
        sim_matrix_item = compute_similarities(user_item_matrix, method='item')
        
        # Sample users for evaluation (to speed up tuning)
        sample_users = validation_data['user_id'].unique()
        if len(sample_users) > 100:  # Limit for faster tuning
            np.random.seed(42)  # For reproducibility
            sample_users = np.random.choice(sample_users, 100, replace=False)
        
        # Create recommendation function closure
        def recommend_for_user(user_id):
            return generate_hybrid_recommendations(
                user_id, 
                user_item_matrix, 
                sim_matrix_item, 
                rules,
                train_subset,
                alpha=0.7
            )
        
        # Evaluate
        metrics = evaluate_recommendations(sample_users, recommend_for_user, validation_data, train_subset)
        
        # Store results
        results.append({
            'half_life': half_life,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'hit_rate': metrics['hit_rate'],
            'ndcg': metrics['ndcg']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best half-life (based on NDCG)
    best_idx = results_df['ndcg'].idxmax()
    best_half_life = results_df.loc[best_idx, 'half_life']
    
    print(f"\nBest half-life: {best_half_life} days")
    print(f"Best NDCG: {results_df.loc[best_idx, 'ndcg']:.4f}")
    
    return best_half_life, results_df

def create_final_model(train_df, half_life, min_support=0.001, min_confidence=0.2):
    """
    Create the final model using the entire training set.
    
    Args:
        train_df (DataFrame): Training data
        half_life (int): Optimal half-life value
        min_support (float): Minimum support for pattern mining
        min_confidence (float): Minimum confidence for association rules
        
    Returns:
        tuple: (user_item_matrix, sim_matrix_item, rules)
    """
    print("Creating final model...")
    
    # Process pattern mining
    _, _, rules = process_pattern_mining(train_df, min_support, min_confidence)
    
    # Compute user-item matrix
    user_item_matrix = compute_time_weighted_user_item_matrix(train_df, half_life)
    
    # Compute similarity matrices
    sim_matrix_item = compute_similarities(user_item_matrix, method='item')
    
    return user_item_matrix, sim_matrix_item, rules

def evaluate_models_on_test(train_df, test_df, user_item_matrix, sim_matrix_item, rules, half_life):
    """
    Evaluate models on the test set.
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Test data
        user_item_matrix (DataFrame): User-item matrix
        sim_matrix_item (DataFrame): Item similarity matrix
        rules (DataFrame): Association rules
        half_life (int): Half-life value
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating models on test set...")
    
    # Sample users from test set
    test_users = test_df['user_id'].unique()
    if len(test_users) > 200:  # Limit for faster evaluation
        np.random.seed(42)  # For reproducibility
        test_users = np.random.choice(test_users, 200, replace=False)
    
    # Create recommendation functions
    def cf_recommend(user_id):
        recs = generate_cf_recommendations(
            user_item_matrix, 
            sim_matrix_item, 
            user_id, 
            method='item'
        )
        if recs is not None:
            return pd.DataFrame({'item': recs.index, 'score': recs.values})
        return None
    
    def pattern_recommend(user_id):
        return recency_weighted_patterns(train_df, user_id, rules)
    
    def hybrid_recommend(user_id):
        return generate_hybrid_recommendations(
            user_id, 
            user_item_matrix, 
            sim_matrix_item, 
            rules,
            train_df,
            alpha=0.7
        )
    
    # Evaluate each model
    print("\nEvaluating CF model:")
    cf_metrics = evaluate_recommendations(test_users, cf_recommend, test_df, train_df)
    
    print("\nEvaluating Pattern Mining model:")
    pattern_metrics = evaluate_recommendations(test_users, pattern_recommend, test_df, train_df)
    
    print("\nEvaluating Hybrid model:")
    hybrid_metrics = evaluate_recommendations(test_users, hybrid_recommend, test_df, train_df)
    
    # Compare results
    comparison = pd.DataFrame({
        'CF': [cf_metrics['precision'], cf_metrics['recall'], cf_metrics['hit_rate'], cf_metrics['ndcg']],
        'Pattern': [pattern_metrics['precision'], pattern_metrics['recall'], pattern_metrics['hit_rate'], pattern_metrics['ndcg']],
        'Hybrid': [hybrid_metrics['precision'], hybrid_metrics['recall'], hybrid_metrics['hit_rate'], hybrid_metrics['ndcg']]
    }, index=['Precision', 'Recall', 'Hit Rate', 'NDCG'])
    
    print("\nModel Comparison:")
    print(comparison)
    
    return {
        'cf': cf_metrics,
        'pattern': pattern_metrics,
        'hybrid': hybrid_metrics,
        'comparison': comparison
    }

def recommend_items_for_user(user_id, user_item_matrix, sim_matrix_item, rules, train_df, with_patterns=True):
    """
    Generate recommendations for a user with or without patterns.
    
    Args:
        user_id (int): User ID
        user_item_matrix (DataFrame): User-item matrix
        sim_matrix_item (DataFrame): Item similarity matrix
        rules (DataFrame): Association rules
        train_df (DataFrame): Training data
        with_patterns (bool): Whether to use patterns or not
        
    Returns:
        DataFrame: Recommendations
    """
    if with_patterns:
        # Use hybrid recommendations
        recs = generate_hybrid_recommendations(
            user_id, 
            user_item_matrix, 
            sim_matrix_item, 
            rules,
            train_df,
            alpha=0.7
        )
    else:
        # Use only collaborative filtering
        cf_recs = generate_cf_recommendations(
            user_item_matrix, 
            sim_matrix_item, 
            user_id, 
            method='item'
        )
        
        if cf_recs is not None:
            recs = pd.DataFrame({'item': cf_recs.index, 'score': cf_recs.values, 'source': 'cf'})
        else:
            # For users with no history, use popular items
            popular_items = train_df['itemDescription'].value_counts().head(5)
            recs = pd.DataFrame({
                'item': popular_items.index,
                'score': popular_items.values / popular_items.values.max(),
                'source': 'popularity'
            })
    
    return recs

def simple_text_interface(user_item_matrix, sim_matrix_item, rules, train_df):
    """
    Simple text interface for the recommendation system.
    
    Args:
        user_item_matrix (DataFrame): User-item matrix
        sim_matrix_item (DataFrame): Item similarity matrix
        rules (DataFrame): Association rules
        train_df (DataFrame): Training data
    """
    print("Welcome to the Grocery Recommendation System!")
    print("---------------------------------------------")
    
    while True:
        # Get user ID
        user_id_input = input("\nEnter user ID (or 'q' to quit): ")
        
        if user_id_input.lower() == 'q':
            break
        
        try:
            user_id = float(user_id_input)
        except ValueError:
            print("Invalid user ID. Please enter a number.")
            continue
        
        # Check if user exists
        if user_id not in user_item_matrix.index and user_id not in train_df['user_id'].unique():
            print(f"User {user_id} not found in the database.")
            use_anyway = input("Would you like to get recommendations anyway? (y/n): ")
            if use_anyway.lower() != 'y':
                continue
        
        # Get with/without patterns option
        pattern_input = input("Generate recommendations with patterns? (y/n): ")
        with_patterns = pattern_input.lower() == 'y'
        
        # Generate recommendations
        recommendations = recommend_items_for_user(
            user_id, 
            user_item_matrix, 
            sim_matrix_item, 
            rules, 
            train_df, 
            with_patterns
        )
        
        # Display recommendations
        if recommendations is not None and not recommendations.empty:
            print(f"\nTop 5 recommendations for user {user_id}:")
            for i, (_, row) in enumerate(recommendations.head(5).iterrows(), 1):
                item = row['item']
                score = row['score']
                source = row.get('source', 'unknown')
                print(f"{i}. {item} (score: {score:.4f}, source: {source})")
        else:
            print(f"No recommendations found for user {user_id}.")

#################################################################
#                   MAIN EXECUTION                              #
#################################################################

# Set random seed for reproducibility
np.random.seed(42)

# 1. Tune half-life parameter
half_life_values = [30, 60, 90, 180, 365]
best_half_life, tuning_results = tune_half_life_lambda(
    train_df, 
    half_life_values,
    min_support=0.001,
    min_confidence=0.2
)

# 2. Create final model with best parameters
user_item_matrix, sim_matrix_item, rules = create_final_model(
    train_df,
    best_half_life,
    min_support=0.001,
    min_confidence=0.2
)

# 3. Evaluate models on test set
test_metrics = evaluate_models_on_test(
    train_df,
    test_df,
    user_item_matrix,
    sim_matrix_item,
    rules,
    best_half_life
)

# 4. Plot evaluation results
plt.figure(figsize=(10, 6))
metrics = ['Precision', 'Recall', 'Hit Rate', 'NDCG']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, test_metrics['comparison']['CF'], width, label='CF')
ax.bar(x, test_metrics['comparison']['Pattern'], width, label='Pattern')
ax.bar(x + width, test_metrics['comparison']['Hybrid'], width, label='Hybrid')

ax.set_ylabel('Score')
ax.set_title('Model Comparison on Test Set')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()

# 5. Plot tuning results
plt.figure(figsize=(10, 6))
for metric in ['precision', 'recall', 'hit_rate', 'ndcg']:
    plt.plot(tuning_results['half_life'], tuning_results[metric], marker='o', label=metric.capitalize())
plt.xlabel('Half-life (days)')
plt.ylabel('Metric Value')
plt.title('Hyperparameter Tuning Results')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. Run simple text interface
simple_text_interface(user_item_matrix, sim_matrix_item, rules, train_df)
