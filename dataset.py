from pathlib import Path

# for saving norm params
import ast
import pprint

import numpy as np
import scipy
import torch
import torch.utils.data as data  # dataset shuffling, sample batching, ...

def get_outcome(sessions, session_id):
    """
    Get reaction time and outcome for trials within a session.

    Parameters
    ----------
    sessions: matlab structure
    session_id: int

    Returns
    -------
    (reaction_time, outcome)
        reaction_time: np.ndarray (trial_num)
        Time from cue to release. Equal to press_to_release if the mouse didn't wait for the cue.

        outcome: np.ndarray (trial_num)
        0 if early, 1 if correct, 2 if late.
    """
    total_trials = len(sessions[session_id]['trials'])
    outcome = np.zeros((total_trials), dtype=int)  # 0 - early, 1 - correct, 2 - late
    reaction_time = np.zeros((total_trials))
    press_to_release = np.zeros((total_trials))

    # convert reaction time and press_to_release from matlab to numpy
    for trial_id in range(len(sessions[session_id]['trials'])):
        reaction_time[trial_id] = sessions[session_id]['trials'][trial_id, 0]['RT'][0, 0]
        press_to_release[trial_id] = sessions[session_id]['trials'][trial_id, 0]['press_to_release'][0, 0]

    # calculate if release was early, correct or later
    equal = np.isclose(press_to_release, reaction_time)
    outcome[equal] = 0
    outcome[~equal & (reaction_time < 1.0)] = 1
    outcome[~equal & (reaction_time > 1.0)] = 2

    return reaction_time, outcome

def convolve(signal: np.ndarray, win_size: int):
    """
    Gaussian filter with window size = 6 sigma.

    Arguments:
        signal: np.ndarray, shape (n_steps)

    Returns:
        np.ndarray, shape (n_steps)
    """

    # TODO padding with mean
    std = win_size / 6 # rule of 3 sigma
    win = scipy.signal.windows.gaussian(win_size, std)
    sig = scipy.signal.convolve(signal, win, mode='same') / sum(win)
    return sig


def load_trial(sessions, cue_or_mov, session_id: int, trial_id: int, neuron_id: int, n_subsample: int) -> np.ndarray:
    """
    Convert spike indices to continuous signal.

    Arguments:
        sessions: matlab structure

    Returns:
        np.ndarray, shape (n_steps)
    """
    time = sessions[session_id]['time'][0]  # 1-D

    n_steps = len(time)
    sig = np.zeros((n_steps))

    sig_shape = sessions[session_id]['trials'][trial_id, 0][cue_or_mov][neuron_id, 0].shape
    if sig_shape[1] == 0:
        # !!! no spikes present. The result array already contains all zeros
        pass
    else:
        spikes = sessions[session_id]['trials'][trial_id, 0][cue_or_mov][neuron_id, 0][:,
                 0] - 1  # indices begin from 1!
        sig[spikes] = 1

    sig = convolve(sig, 1000)

    # subsample
    start = 0
    stop = n_steps - 1
    indices = np.linspace(start, stop, n_subsample).astype(int)  # guarantee sequence length
    sig = sig[indices]
    assert len(sig) == n_subsample

    return sig


def normalize(sig, m=None, std=None):
    """
    Subtract mean and divide by standard deviation.

    sig: (n_steps)
    """
    if m is None:
        m = np.mean(sig)
        std = np.std(sig)
    eps = 1e-4
    return (sig - m) / (std + eps)


def get_num_neurons(sessions, cue_or_mov, session_id):
    """
    Arguments:
        sessions: matlab structure
        cue_or_mov: "spikes_cue" or "spikes_mov"
        session_id: int

    Returns:
        int
    """
    first_trial_id = 0
    trial = sessions[session_id]['trials'][first_trial_id, 0]
    return len(trial[cue_or_mov])


# Index of all trials.
# columns: session_id, trial_id, outcome, reaction time, n_neurons
#              0          1         2           3            4

COL_SESS = 0
COL_TRIAL = 1
COL_OUT = 2
COL_REACT = 3
COL_NEURON = 4

class TrialTable:
    def __init__(self, args):
        """

        Arguments:
            args: global config
        """

        # save smoothed signals here
        self.precompute_file = args.data_dir + "/precompute_smooth_%s_%d.npy" % (args.brain_region, args.subsample_length)

        # save per session normalization
        self.norm_file = args.data_dir + "/precompute_norm_%s.txt" % args.brain_region

    def precompute_smoothing(self, sessions, cue_or_mov, args):
        """
        - create an table of trials for easy random access later in dataloader
        - compute smoothed signals if not computed and save to disk
        - load smoothed signals from disk
        - convert index to dict
        """

        self.trial_table = []  # columns: session_id, trial_id, outcome, reaction time, n_neurons
        self.precomputed = {}  # (session_id, trial_id, neuron_id) -> signal

        # ================ create an table of trials with columns:
        # session_id, trial_id, outcome, reaction time, n_neurons
        for session_id in range(len(sessions)):
            trial_num = len(sessions[session_id]['trials'])
            reaction_time, outcome = get_outcome(sessions, session_id)  # (trial_num), (trial_num)
            for trial_id in range(trial_num):
                # later we will select only successful trials
                trial = sessions[session_id]['trials'][trial_id, 0]
                n_neurons = len(trial[cue_or_mov])
                self.trial_table.append([session_id, trial_id, outcome[trial_id], reaction_time[trial_id], n_neurons])

        self.trial_table = np.array(self.trial_table, dtype=object)
        print("trial table shape:", self.trial_table.shape)

        # TEST: number of neurons must be same inside each session
        for session_id in range(len(sessions)):
            table_session = self.trial_table[self.trial_table[:, COL_SESS] == session_id]
            assert np.all(table_session[:, COL_NEURON] == table_session[0, COL_NEURON])


        # ================ compute smoothing if not computed
        if not Path(self.precompute_file).exists():
            print("precompute smoothed signals to", self.precompute_file)
            precomputed = []

            # iterate over all signals
            for i in range(len(self.trial_table)):
                session_id, trial_id, _out, _rt, n_neurons = self.trial_table[i]
                for neuron_id in range(n_neurons):
                    precomputed.append(load_trial(sessions, cue_or_mov, session_id, trial_id, neuron_id, args.subsample_length))

            precomputed = np.stack(precomputed).astype(np.float16)
            with open(self.precompute_file, "wb") as f:
                np.save(f, precomputed)

        # load smoothed signals
        print("load smoothed signals from", self.precompute_file)
        with open(self.precompute_file, "rb") as f:
            precomputed = np.load(f)
        print("precomputed shape:", precomputed.shape)

        # ================ convert table to dict
        sig_i = 0
        for i in range(len(self.trial_table)):
            session_id, trial_id, _out, _rt, n_neurons = self.trial_table[i]
            for neuron_id in range(n_neurons):
                self.precomputed[(session_id, trial_id, neuron_id)] = precomputed[sig_i]
                sig_i += 1

    def precompute_norm(self, sessions, cue_or_mov, args):
        norm_params = {}  # (session_id, neuron_id) -> []

        if not Path(self.norm_file).exists():
            # compute norm if not computed.
            # important that we don't process more than one session at a time!

            print("precompute norm to", self.norm_file)
            for session_id in range(len(sessions)):
                trial_num = len(sessions[session_id]['trials'])
                #RT, outcome = get_outcome(sessions, session_id)

                n_neurons =  get_num_neurons(sessions, cue_or_mov, session_id)
                for trial_id in range(trial_num):
                    for neuron_id in range(n_neurons):
                        key = (session_id, neuron_id)
                        if not key in norm_params:
                            norm_params[key] = []
                        norm_params[key].append(load_trial(sessions, cue_or_mov, session_id, trial_id, neuron_id, args.subsample_length))

                for neuron_id in range(n_neurons):
                    key = (session_id, neuron_id)
                    session_array = np.concatenate(norm_params[key])

                    # mean
                    m = float(np.mean(session_array))

                    # std
                    std = float(np.std(session_array))

                    norm_params[key] = (m, std)

            with open(self.norm_file, "w") as f:
                pprint.pp(norm_params, stream=f)

        else:
            print("load norm from", self.norm_file)
            with open(self.norm_file, "r") as f:
                norm_params = ast.literal_eval(f.read())

        self.norm_params = norm_params


class NeuronDataset(data.Dataset):
    def __init__(self, sessions, ttable, cue_or_mov, is_validation=False):
        """
        Create a list of (session_id, trial_id, neuron_id) tuples. We must have random access to them for training.
        """
        self.ttable = ttable
        self.trials = []

        if not is_validation:
            # training
            # filter based on session, trial number and outcome
            session_ids = range(0, len(sessions) // 2)  # 15 sessions

            for session_id in session_ids:  # range(len(sessions))
                trial_num = len(sessions[session_id]['trials'])
                RT, outcome = get_outcome(sessions, session_id)
                for trial_id in range(0, trial_num):
                    # === skip early outcomes?
                    # if outcome[trial_id] == 0:
                    #     continue
                    trial = sessions[session_id]['trials'][trial_id, 0]
                    n_neurons = len(trial[cue_or_mov])
                    for neuron_id in range(n_neurons):
                        self.trials.append((session_id, trial_id, neuron_id))

        else:
            # validation
            session_ids = range(len(sessions) // 2, len(sessions))

            for session_id in session_ids:  # range(len(sessions))
                trial_num = len(sessions[session_id]['trials'])
                RT, outcome = get_outcome(sessions, session_id)
                for trial_id in range(1, trial_num):
                    # === skip early outcomes?
                    #if outcome[trial_id] == 0:
                    #    continue
                    trial = sessions[session_id]['trials'][trial_id, 0]
                    n_neurons = len(trial[cue_or_mov])
                    for neuron_id in range(n_neurons):
                        self.trials.append((session_id, trial_id, neuron_id))

    def __getitem__(self, idx):
        """
        return: input (subsample_length), output (subsample_length)
        """
        session_id, trial_id, neuron_id = self.trials[idx]
        # sig = load_trial(sessions, session_id, trial_id, neuron_id, subsample_length)
        sig = self.ttable.precomputed[(session_id, trial_id, neuron_id)]

        # add feature dimension and normalize
        sig = sig[:, None]  # (n_steps, feat)
        m, std = self.ttable.norm_params[(session_id, neuron_id)]
        sig_norm = normalize(sig, m, std)

        return (
            torch.tensor(sig_norm, dtype=torch.float),
            torch.tensor(sig_norm, dtype=torch.float)
        )

    def __len__(self):
        return len(self.trials)