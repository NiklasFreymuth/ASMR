# Training scalars. Includes metrics and losses
ACCURACY = "acc"
TOTAL_LOSS = "loss"
CROSS_ENTROPY = "ce"
BINARY_CROSS_ENTROPY = "bce"
MEAN_SQUARED_ERROR = "mse"
ROOT_MEAN_SQUARED_ERROR = "rmse"
MEAN_ABSOLUTE_ERROR = "mae"

# Training data
SAMPLES = "samples"
TARGETS = "targets"
WEIGHTS = "weights"

# Different Network outputs
NODES = "nodes"
EDGES = "edges"
PREDICTIONS = "predictions"
LOGITS = "logits"
LATENT_FEATURES = "latent_features"

# Recording
FINAL = "final"
FIGURES = "figure"
VIDEO_ARRAYS = "videos"
SCALARS = "scalars"
NETWORK_HISTORY = "network_history"
ADDITIONAL_PLOTS = "additional_plots"
TABLES = "TABLES"

# Heterogeneous modules, config-values
CONCAT_AGGR = "concatenation"
AGGR_AGGR = "aggregation"
SRC = "src"
DEST = "dest"

################
# Environments #
################

# Environment graphs
AGENT = "agent"
EVADER = "evader"
BOXES = "boxes"
ELEMENT = "element"
VERTEX = "vertex"
GAUSSIAN = "gaussian"

# Dynamics
DIRECT = "Direct"
UNICYCLEVEL = "UnicycleVel"
UNICYCLEACCEL = "UnicycleAccel"

# Reward decompositions
PENALTY = "penalty"
REWARD = "reward"
RETURN = "return"
REMAINING_ERROR = "weighted_remaining_error"
ERROR_TIMES_AGENTS = "error_times_agents"
APPROXIMATION_GAIN = "approximation_gain"
LOG_REMAINING_ERROR = "log_remaining_error"
REACHED_ELEMENT_LIMITS = "reached_element_limits"
REFINEMENT_STD = "refinement_std"
VELOCITY_PENALTY = "velocity_penalty"
ELEMENT_PENALTY = "element_penalty"
ELEMENT_PENALTY_LAMBDA = "element_penalty_lambda"
ELEMENT_LIMIT_PENALTY = "element_limit_penalty"
NUM_AGENTS = "num_agents"
DELTA_ELEMENTS = "delta_elements"
NUM_ELEMENTS = "num_elements"
AVG_TOTAL_REFINEMENTS = "avg_total_refinements"
AVG_STEP_REFINEMENTS = "avg_step_refinements"
GLOBAL_REWARD = "global_reward"

# Dones
IS_TRUNCATED = "is_truncated"

# spatial credit assignment
AGENT_MAPPING = "agent_mapping"

# Environment observation cache
LAST_OBSERVATION = "last_observation"
NORMALIZED_AGENT_DISTANCES = "normalized_agent_distances"
NORMALIZED_EVADER_DISTANCES = "normalized_evader_distances"
VISIBILITY_RATIO = "visibility_ratio"
