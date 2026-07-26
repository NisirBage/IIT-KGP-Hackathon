"""Shared benchmark configuration -- every model in Phase 4 uses these exact settings."""
from sklearn.model_selection import RepeatedKFold

RNG_SEED = 42
N_SPLITS = 5
N_REPEATS = 10  # 50 folds total, identical for every model -> paired statistical tests valid

def make_cv():
    """Fresh RepeatedKFold instance -- same seed every call, so every model in the
    benchmark sees byte-identical fold assignments (required for paired comparisons)."""
    return RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RNG_SEED)
