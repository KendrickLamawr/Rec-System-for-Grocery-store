import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import math

class CollaborativeFilteringRecommender:
    """
    Implements a User-Based Collaborative Filtering recommender system.

    Handles recommendation generation based purely on user similarity
    and provides a structure for integrating frequent pattern-based recommendations.
    Adjusted for 'itemDescription' and 'Date' (DD/MM/YYYY) columns.
    """

    def __init__(self, recency_weighting=True, decay_factor=0.95):
        """
        Initializes the recommender.

        Args:
            recency_weighting (bool): Whether to apply time decay to user interactions.
            decay_factor (float): Factor for time decay (e.g., 0.95 means older items
                                   get slightly less weight). Should be between 0 and 1.
                                   Lower value means faster decay. Only used if
                                   recency_weighting is True.
        """
        self.user_item_data = None # Stores user purchase history {user_id: {item_id: timestamp}}
        self.item_user_data = None # Stores item purchase history {item_id: {user_id: timestamp}}
        self.user_similarity_matrix = None
        self.user_map = None # Maps user_id to matrix index
        self.reverse_user_map = None # Maps matrix index to user_id
        self.item_map = None # Maps item_id (originally itemDescription) to matrix index
        self.reverse_item_map = None # Maps matrix index to item_id
        self.user_item_matrix = None # Sparse or dense matrix representation
        self.popular_items = None # List of most popular items for cold starts
        self.recency_weighting = recency_weighting
        self.decay_factor = decay_factor
        self.fitted = False
        print(f"Initialized Recommender: Recency Weighting={self.recency_weighting}, Decay Factor={self.decay_factor}")

    def _preprocess_data(self, df):
        """
        Preprocesses the raw transaction data.
        Expects columns ['user_id', 'itemDescription', 'Date'].
        Renames 'itemDescription' to 'item_id'.
        Parses 'Date' as DD/MM/YYYY.

        Args:
            df (pd.DataFrame): DataFrame with columns ['user_id', 'itemDescription', 'Date'].

        Returns:
            pd.DataFrame: Preprocessed DataFrame with 'item_id' and 'timestamp'.
                          Returns None if essential columns are missing.
        """
        print("Preprocessing data...")
        required_cols = ['user_id', 'itemDescription', 'Date']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: Input DataFrame missing one or more required columns: {required_cols}")
            return None

        # Rename item description column
        df = df.rename(columns={'itemDescription': 'item_id'})
        df =df.rename(columns = {'User_id':'user_id'})

        # Ensure correct data types
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)

        # Convert date to datetime objects, specifying the format
        original_rows = len(df)
        try:
            # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
            df['timestamp'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        except Exception as e:
            # Catch potential errors even with specified format
            print(f"Error converting 'Date' column to datetime: {e}")
            raise # Re-raise the exception if format string itself is wrong etc.

        # Handle rows where date parsing failed
        rows_with_bad_dates = df['timestamp'].isna().sum()
        if rows_with_bad_dates > 0:
            print(f"Warning: Could not parse 'Date' for {rows_with_bad_dates} rows. These rows will be dropped.")
            df = df.dropna(subset=['timestamp'])

        # Sort by timestamp to establish recency correctly
        df = df.sort_values(by='timestamp')

        print(f"Data preprocessed. Shape after cleaning: {df.shape} (started with {original_rows} rows)")
        return df[['user_id', 'item_id', 'timestamp']] # Return only necessary columns


    def _calculate_time_decay_weight(self, transaction_timestamp, reference_timestamp):
        """
        Calculates a weight based on the time difference.
        More recent transactions get higher weights.

        Args:
            transaction_timestamp (datetime): Timestamp of the transaction.
            reference_timestamp (datetime): The most recent timestamp in the dataset,
                                           used as the reference point.

        Returns:
            float: A weight between 0 and 1.
        """
        if not self.recency_weighting:
            return 1.0 # No decay if weighting is off

        time_diff_days = (reference_timestamp - transaction_timestamp).days
        # Apply exponential decay: weight = decay_factor ^ (time_diff_days / period)
        # Using 30 days as a period for monthly decay approx
        time_diff_days = max(0, time_diff_days) # Ensure non-negative difference
        weight = math.pow(self.decay_factor, time_diff_days / 30)
        return max(0.01, weight) # Ensure a minimum weight to avoid zero values


    def fit(self, df):
        """
        Trains the recommender system on the provided transaction data.

        Args:
            df (pd.DataFrame): Training data with columns ['user_id', 'itemDescription', 'Date'].
        """
        print("Fitting the recommender...")
        processed_df = self._preprocess_data(df.copy())

        if processed_df is None or processed_df.empty:
             print("Error: Preprocessing failed or resulted in empty data. Cannot fit.")
             return # Stop fitting if data is bad

        # --- Create Mappings ---
        unique_users = processed_df['user_id'].unique()
        unique_items = processed_df['item_id'].unique() # Use renamed column

        self.user_map = {user_id: i for i, user_id in enumerate(unique_users)}
        self.reverse_user_map = {i: user_id for user_id, i in self.user_map.items()}
        self.item_map = {item_id: i for i, item_id in enumerate(unique_items)}
        self.reverse_item_map = {i: item_id for item_id, i in self.item_map.items()}
        print(f"Created mappings: {len(unique_users)} users, {len(unique_items)} items.")

        # --- Store User/Item Interactions with Timestamps ---
        self.user_item_data = defaultdict(dict)
        self.item_user_data = defaultdict(dict)
        max_timestamp = processed_df['timestamp'].max() # Reference for decay

        for _, row in processed_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id'] # Use renamed column
            timestamp = row['timestamp']
            # Store the *latest* timestamp for each user-item interaction
            self.user_item_data[user_id][item_id] = timestamp
            self.item_user_data[item_id][user_id] = timestamp

        # --- Build User-Item Matrix (Weighted by Recency if enabled) ---
        n_users = len(unique_users)
        n_items = len(unique_items)
        # Using a dense matrix for simplicity here, consider sparse for large datasets
        self.user_item_matrix = np.zeros((n_users, n_items))

        print("Building user-item matrix...")
        for user_id, items in self.user_item_data.items():
            if user_id in self.user_map:
                user_idx = self.user_map[user_id]
                for item_id, timestamp in items.items():
                    if item_id in self.item_map:
                        item_idx = self.item_map[item_id]
                        weight = self._calculate_time_decay_weight(timestamp, max_timestamp)
                        # Use the weight as the interaction value
                        self.user_item_matrix[user_idx, item_idx] = weight

        # --- Calculate User Similarity ---
        print("Calculating user similarity...")
        # Cosine similarity: measures the cosine of the angle between two non-zero vectors.
        self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        # Set diagonal to 0 to avoid self-similarity influencing recommendations
        np.fill_diagonal(self.user_similarity_matrix, 0)
        print("User similarity matrix calculated.")

        # --- Calculate Popular Items (for cold start) ---
        print("Calculating popular items...")
        item_counts = processed_df['item_id'].value_counts() # Use renamed column
        # Sort by count descending, take top N (e.g., 20)
        self.popular_items = item_counts.head(20).index.tolist()
        print(f"Top 5 popular items: {self.popular_items[:5]}")

        self.fitted = True
        print("Recommender fitting complete.")


    def _get_popular_items(self, k=5):
        """Returns the top k most popular items."""
        if self.popular_items is None:
            print("Warning: Popular items not calculated. Call fit() first.")
            return []
        return self.popular_items[:k]

    def get_recommendations_cf(self, user_id, k=5, N_similar_users=50):
        """
        Generates top-k recommendations for a user using User-Based CF.

        Args:
            user_id (str): The ID of the user to generate recommendations for.
            k (int): The number of recommendations to return.
            N_similar_users (int): The number of similar users to consider.

        Returns:
            list: A list of top-k recommended item_ids (item descriptions).
                  Returns popular items if the user is new or has no history.
        """
        print(f"\nGenerating CF recommendations for user: {user_id}")
        if not self.fitted:
            raise RuntimeError("Recommender has not been fitted. Call fit() first.")

        user_id = str(user_id) # Ensure consistent type

        # --- Handle New or Unknown Users ---
        if user_id not in self.user_map:
            print(f"User '{user_id}' not found in training data. Returning popular items.")
            return self._get_popular_items(k)

        user_idx = self.user_map[user_id]

        # --- Get User's Purchase History ---
        purchased_items = set(self.user_item_data.get(user_id, {}).keys())
        if not purchased_items:
             print(f"User '{user_id}' has no purchase history in training data. Returning popular items.")
             return self._get_popular_items(k)
        print(f"User '{user_id}' has purchased {len(purchased_items)} items.")

        # --- Find Similar Users ---
        user_similarities = self.user_similarity_matrix[user_idx]
        # Get indices of top N similar users (excluding self)
        # Ensure N_similar_users doesn't exceed number of users - 1
        max_similar = self.user_similarity_matrix.shape[0] - 1
        N_similar_users = min(N_similar_users, max_similar)
        similar_user_indices = np.argsort(user_similarities)[::-1][1:N_similar_users+1]


        # --- Calculate Recommendation Scores ---
        recommendation_scores = defaultdict(float)
        print(f"Considering top {N_similar_users} similar users...")

        for similar_user_idx in similar_user_indices:
            similar_user_id = self.reverse_user_map[similar_user_idx]
            similarity_score = user_similarities[similar_user_idx]

            # Optimization: Skip users with zero similarity
            if similarity_score <= 0:
                continue

            # Get items purchased by the similar user
            similar_user_items = self.user_item_data.get(similar_user_id, {})

            for item_id, timestamp in similar_user_items.items():
                # Only recommend items NOT already purchased by the target user
                if item_id not in purchased_items:
                    # The score is the sum of similarities of users who bought the item
                    # Weighted by the interaction value (which includes recency if enabled)
                    # Need item index for the matrix lookup
                    if item_id in self.item_map:
                         item_idx = self.item_map[item_id]
                         interaction_weight = self.user_item_matrix[similar_user_idx, item_idx]
                         recommendation_scores[item_id] += similarity_score * interaction_weight
                    # else: item might only exist in test set, ignore


        # --- Sort and Select Top K ---
        sorted_recommendations = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)

        # Extract just the item IDs
        recommended_item_ids = [item_id for item_id, score in sorted_recommendations]

        print(f"Generated {len(recommended_item_ids)} potential recommendations.")
        final_recommendations = recommended_item_ids[:k]
        print(f"Top {k} CF recommendations: {final_recommendations}")
        return final_recommendations


    def get_recommendations_patterns(self, user_id, frequent_patterns, k=5):
        """
        Generates recommendations based on frequent patterns (Task 1 output).

        Args:
            user_id (str): The ID of the user.
            frequent_patterns (object): The output from the pattern mining task.
                                        The exact format needs to be defined/agreed upon
                                        with Task 1. Example: list of tuples:
                                        (antecedent_set<str>, consequent_item<str>, score<float>).
            k (int): Number of recommendations.

        Returns:
            list: Top-k recommended item_ids (item descriptions) based on patterns.
        """
        print(f"\nGenerating Pattern recommendations for user: {user_id}")
        if not self.fitted:
             raise RuntimeError("Recommender has not been fitted. Call fit() first.")

        user_id = str(user_id)
        if user_id not in self.user_map:
            print(f"User '{user_id}' not found. Cannot generate pattern recommendations.")
            return [] # Return empty list for unknown users

        # --- Get User's Purchase History ---
        user_purchases = set(self.user_item_data.get(user_id, {}).keys())
        if not user_purchases:
             print(f"User '{user_id}' has no purchase history. Cannot generate pattern recommendations.")
             return [] # Return empty list for users with no history

        # --- Placeholder Logic for Pattern Matching ---
        pattern_recommendation_scores = defaultdict(float)
        print("Matching user history against frequent patterns...")

        if frequent_patterns: # Check if patterns were actually provided
             for pattern in frequent_patterns:
                 # --- Adapt this section based on the ACTUAL structure of frequent_patterns ---
                 try:
                     # Assuming pattern is ( {set of antecedent strings}, consequent_string, score_float )
                     antecedent_set, consequent_item, score = pattern
                     # Ensure types are correct (may not be needed if Task 1 guarantees format)
                     antecedent_set = set(map(str, antecedent_set))
                     consequent_item = str(consequent_item)
                     score = float(score)
                 except (TypeError, ValueError, IndexError) as e:
                     print(f"Warning: Skipping invalid pattern format: {pattern}. Error: {e}")
                     continue
                 # ---------------------------------------------------------------------------

                 # Check if user bought all items in the antecedent
                 if antecedent_set.issubset(user_purchases):
                     # Check if the consequent item was NOT already bought
                     if consequent_item not in user_purchases:
                         # Add score (using max score if item recommended by multiple patterns)
                         pattern_recommendation_scores[consequent_item] = max(
                             pattern_recommendation_scores[consequent_item], score
                         )
        else:
            print("No frequent patterns provided.")


        # --- Sort and Select Top K ---
        sorted_pattern_recs = sorted(pattern_recommendation_scores.items(), key=lambda item: item[1], reverse=True)
        pattern_rec_ids = [item_id for item_id, score in sorted_pattern_recs]

        print(f"Generated {len(pattern_rec_ids)} potential pattern recommendations.")
        final_pattern_recs = pattern_rec_ids[:k]
        print(f"Top {k} Pattern recommendations: {final_pattern_recs}")
        return final_pattern_recs


    def get_combined_recommendations(self, user_id, frequent_patterns, k=5, cf_weight=0.5, pattern_weight=0.5, N_similar_users=50):
        """
        Generates recommendations by combining CF and Pattern-based results.

        Args:
            user_id (str): The user ID.
            frequent_patterns (object): Output from Task 1.
            k (int): Number of recommendations.
            cf_weight (float): Weight for CF scores.
            pattern_weight (float): Weight for Pattern scores.
            N_similar_users (int): Number of similar users for CF.

        Returns:
            list: Top-k combined recommended item_ids (item descriptions).
        """
        print(f"\nGenerating Combined recommendations for user: {user_id}")
        user_id = str(user_id)

        # --- Handle New Users ---
        if not self.fitted:
             print("Recommender not fitted. Returning popular items.")
             return self._get_popular_items(k)
        if user_id not in self.user_map:
             print(f"User '{user_id}' not found. Returning popular items.")
             return self._get_popular_items(k)

        # --- Get CF Recommendation Scores ---
        cf_scores = defaultdict(float)
        user_idx = self.user_map[user_id]
        purchased_items = set(self.user_item_data.get(user_id, {}).keys())
        user_similarities = self.user_similarity_matrix[user_idx]
        max_similar = self.user_similarity_matrix.shape[0] - 1
        N_similar_users = min(N_similar_users, max_similar)
        similar_user_indices = np.argsort(user_similarities)[::-1][1:N_similar_users+1]

        for similar_user_idx in similar_user_indices:
            similar_user_id = self.reverse_user_map[similar_user_idx]
            similarity_score = user_similarities[similar_user_idx]
            if similarity_score <= 0: continue
            similar_user_items = self.user_item_data.get(similar_user_id, {})
            for item_id, timestamp in similar_user_items.items():
                if item_id not in purchased_items:
                     if item_id in self.item_map: # Check if item exists in training map
                        item_idx = self.item_map[item_id]
                        interaction_weight = self.user_item_matrix[similar_user_idx, item_idx]
                        cf_scores[item_id] += similarity_score * interaction_weight

        # Normalize CF scores
        max_cf_score = max(cf_scores.values()) if cf_scores else 1.0 # Avoid division by zero
        normalized_cf_scores = {item: score / max_cf_score for item, score in cf_scores.items()}


        # --- Get Pattern Recommendation Scores ---
        pattern_scores = defaultdict(float)
        user_purchases = set(self.user_item_data.get(user_id, {}).keys()) # Re-get just in case
        if frequent_patterns and user_purchases:
             for pattern in frequent_patterns:
                 try:
                     antecedent_set, consequent_item, score = pattern
                     antecedent_set = set(map(str, antecedent_set))
                     consequent_item = str(consequent_item)
                     score = float(score)
                 except (TypeError, ValueError, IndexError) as e:
                     # print(f"Warning: Skipping invalid pattern format: {pattern}. Error: {e}") # Optional: reduce verbosity
                     continue

                 if antecedent_set.issubset(user_purchases) and consequent_item not in user_purchases:
                     pattern_scores[consequent_item] = max(pattern_scores[consequent_item], score)

        # Normalize Pattern scores
        max_pattern_score = max(pattern_scores.values()) if pattern_scores else 1.0 # Avoid division by zero
        normalized_pattern_scores = {item: score / max_pattern_score for item, score in pattern_scores.items()}

        # --- Combine Scores ---
        combined_scores = defaultdict(float)
        all_recommended_items = set(normalized_cf_scores.keys()) | set(normalized_pattern_scores.keys())

        for item_id in all_recommended_items:
            cf_s = normalized_cf_scores.get(item_id, 0)
            pat_s = normalized_pattern_scores.get(item_id, 0)
            # Ensure item exists in training map before assigning score (safety check)
            if item_id in self.item_map:
                combined_scores[item_id] = (cf_weight * cf_s) + (pattern_weight * pat_s)

        # --- Sort and Select Top K ---
        # Filter out any potential items not in the original training item map if necessary?
        # Might happen if patterns introduce items not seen in training transactions.
        # combined_scores = {item: score for item, score in combined_scores.items() if item in self.item_map}

        sorted_combined_recs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        combined_rec_ids = [item_id for item_id, score in sorted_combined_recs]

        print(f"Generated {len(combined_rec_ids)} potential combined recommendations.")
        final_combined_recs = combined_rec_ids[:k]
        print(f"Top {k} Combined recommendations: {final_combined_recs}")
        return final_combined_recs


    def evaluate(self, test_df, k=5, frequent_patterns=None, mode='cf'):
        """
        Evaluates the recommender system using Precision@k and Recall@k.

        Args:
            test_df (pd.DataFrame): Test data with columns ['user_id', 'itemDescription', 'Date'].
            k (int): The number of recommendations to evaluate (e.g., @5).
            frequent_patterns (object): Output from Task 1 (needed for 'pattern' or 'combined' modes).
                                        Should be generated ONLY from the training data.
            mode (str): Evaluation mode: 'cf', 'pattern', or 'combined'.

        Returns:
            dict: A dictionary containing evaluation metrics (e.g., {'precision@k': ..., 'recall@k': ...}).
                  Returns None if evaluation cannot be performed.
        """
        print(f"\nEvaluating recommendations (mode: {mode})...")
        if not self.fitted:
            print("Error: Recommender has not been fitted. Cannot evaluate.")
            return None

        # --- Prepare Test Data ---
        print("Preprocessing test data for evaluation...")
        eval_df = self._preprocess_data(test_df.copy()) # Use same preprocessing
        if eval_df is None or eval_df.empty:
             print("Error: Test data preprocessing failed or resulted in empty data. Cannot evaluate.")
             return None

        # Group actual purchases in the (preprocessed) test set by user
        actual_purchases = eval_df.groupby('user_id')['item_id'].apply(set).to_dict()
        test_users = eval_df['user_id'].unique()

        all_precisions = []
        all_recalls = []

        if mode != 'cf' and frequent_patterns is None:
            print(f"Warning: Mode is '{mode}' but frequent_patterns not provided. Evaluation might be inaccurate or fail.")


        print(f"Evaluating on {len(test_users)} users from the test set...")
        users_evaluated = 0
        for user_id in test_users:
            # user_id is already string from preprocessing
            if user_id not in actual_purchases:
                continue # Skip users with no actual purchases in test set

            # Get recommendations based on the specified mode
            recommendations = []
            if mode == 'cf':
                # Only generate recommendations for users seen in training
                if user_id in self.user_map:
                    recommendations = self.get_recommendations_cf(user_id, k=k)
                # else: user is new in test set, CF gives popular items, which might not be ideal for eval
                # Or maybe we should skip evaluating users not in training set? For now, allow popular.
                elif user_id not in self.user_map:
                     recommendations = self._get_popular_items(k) # Or [] ?

            elif mode == 'pattern':
                 # Only generate recommendations for users seen in training
                 if user_id in self.user_map:
                    recommendations = self.get_recommendations_patterns(user_id, frequent_patterns, k=k)
                 # else: new user, pattern gives [], which is fine for eval

            elif mode == 'combined':
                 # Combined handles new users by returning popular items
                 recommendations = self.get_combined_recommendations(user_id, frequent_patterns, k=k)
            else:
                print(f"Error: Invalid evaluation mode '{mode}'. Use 'cf', 'pattern', or 'combined'.")
                return None

            actual = actual_purchases[user_id]
            # Filter actual items to only those known during training? Optional, depends on goal.
            # actual = {item for item in actual if item in self.item_map}

            hits = len(set(recommendations) & actual)

            # --- Calculate Precision@k and Recall@k ---
            precision_at_k = hits / k if k > 0 else 0
            recall_at_k = hits / len(actual) if len(actual) > 0 else 0

            all_precisions.append(precision_at_k)
            all_recalls.append(recall_at_k)
            users_evaluated += 1

        # --- Calculate Average Metrics ---
        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0

        print(f"Evaluation complete for {users_evaluated} users.")
        print(f"Average Precision@{k}: {avg_precision:.4f}")
        print(f"Average Recall@{k}: {avg_recall:.4f}")

        return {
            f'precision@{k}': avg_precision,
            f'recall@{k}': avg_recall
        }

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Load Data (Using new filenames and columns)
    train_file = 'Groceries data train.csv'
    test_file = 'Groceries data test.csv'
    try:
        # Load using original column names
        train_df = pd.read_csv(train_file) # Let preprocessing handle date parsing
        print(f"Training data loaded successfully from {train_file}.")
        print(f"Train columns: {train_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Error: {train_file} not found. Creating dummy data...")
        # Create dummy data if file not found, matching expected input columns
        data = {
            'user_id': ['1', '1', '1', '2', '2', '3', '3', '3', '3', '4', '4', '1', '5'],
            'itemDescription': ['yogurt', 'soda', 'root vegetables', 'yogurt', 'whole milk', 'soda', 'root vegetables', 'hard cheese', 'butter', 'yogurt', 'butter', 'whole milk', 'other vegetables'],
            'Date': [ # Using DD/MM/YYYY format
                '01/01/2023', '02/01/2023', '05/01/2023', '01/01/2023', '03/01/2023',
                '02/01/2023', '05/01/2023', '06/01/2023', '01/02/2023', '10/01/2023',
                '05/02/2023', '10/02/2023', '15/01/2023'
            ]
            # Add other columns if needed by other parts of the system, but CF only needs these 3
        }
        train_df = pd.DataFrame(data)
        print(f"Dummy training data created. Columns: {train_df.columns.tolist()}")


    # 2. Initialize and Fit Recommender
    # Try with and without recency weighting
    recommender = CollaborativeFilteringRecommender(recency_weighting=True, decay_factor=0.9)
    recommender.fit(train_df) # Pass the DataFrame with original column names

    # --- Dummy frequent patterns (replace with actual output from Task 1) ---
    # Format: [( {antecedent_items}, consequent_item, score ), ...]
    # Items must match the 'itemDescription' values from the data
    dummy_patterns = [
        ({'yogurt', 'soda'}, 'root vegetables', 0.9),
        ({'root vegetables'}, 'hard cheese', 0.8),
        ({'butter'}, 'waffles', 0.7), # 'waffles' might not be in dummy data
        ({'yogurt', 'whole milk'}, 'other vegetables', 0.85)
    ]
    print(f"\nUsing dummy patterns: {dummy_patterns}")

    # 3. Get Recommendations (CF only)
    user_to_recommend = '1'
    if recommender.fitted:
        cf_recs = recommender.get_recommendations_cf(user_id=user_to_recommend, k=5)

        user_to_recommend_new = '99' # User not in training data
        cf_recs_new = recommender.get_recommendations_cf(user_id=user_to_recommend_new, k=5)

        # 4. Get Recommendations (Patterns)
        pattern_recs = recommender.get_recommendations_patterns(user_id=user_to_recommend, frequent_patterns=dummy_patterns, k=5)

        # 5. Get Combined Recommendations
        combined_recs = recommender.get_combined_recommendations(
            user_id=user_to_recommend,
            frequent_patterns=dummy_patterns,
            k=5,
            cf_weight=0.6, # Give slightly more weight to CF
            pattern_weight=0.4
        )
    else:
        print("\nSkipping recommendation generation as fitting failed.")


    # 6. Evaluate (Requires the test set file)
    if recommender.fitted:
        try:
            test_df = pd.read_csv(test_file)
            print(f"\nTest data loaded successfully from {test_file}.")
            print(f"Test columns: {test_df.columns.tolist()}")

            # IMPORTANT: Ensure frequent_patterns used for evaluation are generated ONLY from train_df
            # Evaluate different modes:
            print("\n--- Evaluating CF Mode ---")
            eval_results_cf = recommender.evaluate(test_df.copy(), k=5, mode='cf')
            if eval_results_cf: print(f"CF Evaluation Results: {eval_results_cf}")

            print("\n--- Evaluating Pattern Mode ---")
            # Pass the *same* dummy_patterns generated from training data perspective
            eval_results_pat = recommender.evaluate(test_df.copy(), k=5, frequent_patterns=dummy_patterns, mode='pattern')
            if eval_results_pat: print(f"Pattern Evaluation Results: {eval_results_pat}")

            print("\n--- Evaluating Combined Mode ---")
            eval_results_comb = recommender.evaluate(test_df.copy(), k=5, frequent_patterns=dummy_patterns, mode='combined')
            if eval_results_comb: print(f"Combined Evaluation Results: {eval_results_comb}")

        except FileNotFoundError:
            print(f"\nWarning: {test_file} not found. Skipping evaluation.")
        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")
    else:
         print("\nSkipping evaluation as fitting failed.")
