import numpy as np
from dataset import get_num_neurons, COL_NEURON
from intrinsic_dimension import intrinsic_dimension_cpu


def compute_id_matrix(sessions, args, ttable, latents, target_outcome):
    """

    Arguments:
        sessions: matlab structure
        args: Args dataclass
        ttable: TrialTable
        latents: np.ndarray of shape (n_signals, latent_size)
        target_outcome: int, 0 (early), 1 (correct) or 2 (late)
    Returns:
        np.ndarray of shape (n_sessions, max_neurons)
    """
    max_neurons = ttable.trial_table[:, COL_NEURON].max()
    results = np.zeros((len(sessions), max_neurons))

    for session_id in range(len(sessions)):
        n_neurons = get_num_neurons(sessions, args.cue_or_mov, session_id)
        for neuron_id in range(n_neurons):
            # estimate ID for certain session and neuron
            latents_arr = []
            sample_i = 0
            for i in range(len(ttable.trial_table)):
                session_id_new, trial_id_new, outcome, RT, n_neurons = ttable.trial_table[i]

                # join correct and late
                if outcome == 2:
                    outcome = 0

                for neuron_id_new in range(n_neurons):
                    if session_id_new == session_id and neuron_id_new == neuron_id and outcome == target_outcome:
                        latent = latents[sample_i]
                        latents_arr.append(latent)
                    sample_i += 1
            latents_arr = np.stack(latents_arr)  # (n_points, n_features)
            results[session_id, neuron_id] = intrinsic_dimension_cpu(latents_arr)
    return results
