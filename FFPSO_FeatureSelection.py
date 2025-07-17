import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from typing import List, Tuple, Union

def calculate_feature_score(X: np.ndarray, y: np.ndarray,
                          mi_weight: float = 0.6,
                          corr_weight: float = 0.3,
                          red_weight: float = 0.1,
                          min_unique_vals: int = 2) -> float:
    """Calculate feature importance score based on mutual information, correlation, and redundancy."""
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X and y must be numpy arrays")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if not np.isclose(mi_weight + corr_weight + red_weight, 1.0):
        raise ValueError("Weights must sum to 1.0")

    try:
        # Calculate mutual information score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_scores = mutual_info_classif(X, y)
        avg_mi = np.mean(mi_scores) if len(mi_scores) > 0 else 0

        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            unique_x = len(np.unique(X[:, i]))
            unique_y = len(np.unique(y))

            if unique_x >= min_unique_vals and unique_y >= min_unique_vals:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = pearsonr(X[:, i], y)
                correlations.append(abs(corr))
            else:
                correlations.append(0)
        avg_corr = np.mean(correlations) if correlations else 0

        # Calculate feature redundancy
        n_features = X.shape[1]
        if n_features <= 1:
            avg_redundancy = 0
        else:
            redundancy = 0
            valid_pairs = 0
            for i in range(n_features):
                for j in range(i+1, n_features):
                    if (len(np.unique(X[:, i])) >= min_unique_vals and
                        len(np.unique(X[:, j])) >= min_unique_vals):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            corr, _ = pearsonr(X[:, i], X[:, j])
                        redundancy += abs(corr)
                        valid_pairs += 1

            avg_redundancy = (redundancy / valid_pairs) if valid_pairs > 0 else 0

        # Calculate final score
        score = (mi_weight * avg_mi +
                corr_weight * avg_corr -
                red_weight * avg_redundancy)

        return float(max(0, min(1, score)))  # Clip score between 0 and 1

    except Exception as e:
        warnings.warn(f"Error calculating feature score: {str(e)}")
        return 0.0

class ClassifierFreeSFS:
    """Sequential Forward Selection without using a classifier."""

    def __init__(self, n_features_to_select: int):
        self.n_features_to_select = n_features_to_select
        self.selected_features_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassifierFreeSFS':
        n_features = X.shape[1]
        self.selected_features_ = []
        remaining_features = list(range(n_features))

        for _ in range(min(self.n_features_to_select, n_features)):
            best_feature = None
            best_score = -np.inf

            for feature in remaining_features:
                current_features = self.selected_features_ + [feature]
                score = calculate_feature_score(X[:, current_features], y)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature is not None:
                self.selected_features_.append(best_feature)
                remaining_features.remove(best_feature)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

class ClassifierFreeSFSFireflyPSO:
    """Hybrid feature selection using Classifier-Free SFS and Firefly-PSO."""

    def __init__(self, n_particles: int = 20, n_fireflies: int = 10, n_iterations: int = 50,
                 alpha: float = 0.5, beta: float = 0.2, gamma: float = 1.0,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 initial_features: int = 20):
        self.n_particles = n_particles
        self.n_fireflies = n_fireflies
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.initial_features = initial_features

    def optimize(self, X: np.ndarray, y: np.ndarray, n_features: int) -> List[int]:
        # Initial feature selection using Classifier-Free SFS
        sfs = ClassifierFreeSFS(n_features_to_select=self.initial_features)
        X_initial = sfs.fit_transform(X, y)
        initial_feature_indices = sfs.selected_features_

        print(f"Initial features selected: {initial_feature_indices}")

        # Initialize particles and fireflies
        n_features_total = len(initial_feature_indices)
        particles = np.random.rand(self.n_particles, n_features_total)
        velocities = np.zeros_like(particles)
        fireflies = np.random.rand(self.n_fireflies, n_features_total)

        pbest = particles.copy()
        pbest_fitness = np.zeros(self.n_particles)
        gbest = np.zeros(n_features_total)
        gbest_fitness = -np.inf

        for iteration in range(self.n_iterations):
            # PSO update
            for i in range(self.n_particles):
                selected_features = self._select_top_features(particles[i], n_features)
                fitness = calculate_feature_score(X_initial[:, selected_features], y)

                if fitness > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = fitness

                if fitness > gbest_fitness:
                    gbest = particles[i].copy()
                    gbest_fitness = fitness

            # Update particle positions and velocities
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                               self.c1 * r1 * (pbest[i] - particles[i]) +
                               self.c2 * r2 * (gbest - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)

            # Firefly update
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if calculate_feature_score(X_initial[:, self._select_top_features(fireflies[j], n_features)], y) > \
                       calculate_feature_score(X_initial[:, self._select_top_features(fireflies[i], n_features)], y):
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta * np.exp(-self.gamma * r**2)
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + \
                                      self.alpha * (np.random.rand(n_features_total) - 0.5)
                        fireflies[i] = np.clip(fireflies[i], 0, 1)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Best score: {gbest_fitness:.4f}")

        final_features = self._select_top_features(gbest, n_features)
        return [initial_feature_indices[i] for i in final_features]

    def _select_top_features(self, particle: np.ndarray, n_features: int) -> List[int]:
        return particle.argsort()[-n_features:][::-1].tolist()

def load_nsl_kdd_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the NSL-KDD dataset."""
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"
    ]

    df = pd.read_csv(filepath, header=None, names=columns)
    df = df.drop('difficulty', axis=1)

    le = LabelEncoder()
    categorical_cols = ['protocol_type', 'service', 'flag', 'label']
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Create binary attack flag
    df['attackflag'] = (df['label'] != le.transform(['normal'])[0]).astype(int)

    X = df.drop(['label', 'attackflag'], axis=1)
    y = df['attackflag']

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def evaluate_final_model(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        selected_features: List[int]) -> Tuple[float, float]:
    """Evaluate the final model using selected features."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    clf.fit(X_train_selected, y_train)

    train_pred = clf.predict(X_train_selected)
    test_pred = clf.predict(X_test_selected)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_precision=precision_score(y_train,train_pred)
    test_precision=precision_score(y_test,test_pred)
    train_recall=recall_score(y_train,train_pred)
    test_recall=recall_score(y_test,test_pred)
    train_f1=f1_score(y_train,train_pred)
    test_f1=f1_score(y_test,test_pred)

    return train_acc, test_acc, train_precision, test_precision, train_recall, test_recall, train_f1, test_f1



def main():
    # Load NSL-KDD dataset
    print("Loading NSL-KDD dataset...")
    """
    #if separate train and test datasets: 
    X_train, y_train = load_nsl_kdd_data('/KDDTrain.csv')
    X_test, y_test = load_nsl_kdd_data('/KDDTest.csv')"""

    X,y=load_nsl_kdd_data('/KDDTrain.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and run the hybrid algorithm
    print("Running hybrid feature selection...")
    cfsfs_fpso = ClassifierFreeSFSFireflyPSO(
        n_particles=20, n_fireflies=10, n_iterations=20,
        alpha=0.5, beta=0.2, gamma=1.0, w=0.7, c1=1.5, c2=1.5,
        initial_features=20
    )

    final_features = cfsfs_fpso.optimize(X_train.values, y_train.values, n_features=10)

    print("\nFinal selected features:")
    for i, feature in enumerate(final_features, 1):
        print(f"{i}. {X_train.columns[feature]}")

    # Evaluate the final model
    final_train_acc, final_test_acc, final_train_pre, final_test_pre, final_train_recall, final_test_recall, final_train_f1, final_test_f1 = evaluate_final_model(
        X_train, y_train, X_test, y_test, final_features
    )

    print(f"\nFinal Results:")
    print(f"Training Accuracy: {final_train_acc:.4f}")
    print(f"Test Accuracy: {final_test_acc:.4f}")
    print(f"Training precision: {final_train_pre:.4f}")
    print(f"Test precision: {final_test_pre:.4f}")
    print(f"Training recall: {final_train_recall:.4f}")
    print(f"Test recall: {final_test_recall:.4f}")
    print(f"Training F1: {final_train_f1:.4f}")
    print(f"Test F1: {final_test_f1:.4f}")

if __name__ == "__main__":
    main()
