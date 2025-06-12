import numpy as np
import torch
from dataset import normalize


def load_sample(ttable, session_id, trial_id, neuron_id):
    """
    Returns:
        signal: torch.tensor (n_steps, 1)
    """
    # load time series
    sig = ttable.precomputed[(session_id, trial_id, neuron_id)]
    #sig = load_trial(sessions, session_id, trial_id, neuron_id, subsample_length)

    # add feature dimension
    sig = sig[:, None] # (n_steps, feat)

    # normalize and convert to torch tensor
    m, std = ttable.norm_params[(session_id, neuron_id)]
    norm = normalize(sig, m, std)
    return torch.tensor(norm, dtype=torch.float)

def predict(model, args, inp):
    """
    Predict for one signal.

    Arguments:
        inp: torch tensor (b, n_steps, n_features)

    Returns:
        (signal: tensor (b, n_steps), latent: tensor (b, latent_size))
    """
    with torch.no_grad():
        # add batch dimension
        #inputs = torch.tensor(inp, dtype=torch.float)[None, :, :].to(args.DEVICE)
        inputs = torch.tensor(inp, dtype=torch.float).to(args.DEVICE)

        # forward, produces output signal of shape (b, n_steps, n_features)
        # and latents of shape (b, latent_size)
        outputs, latent = model(inputs)

        # remove batch and feature dimensions, move to CPU
        #return (outputs[0, :, 0].cpu().detach(), latent[0].cpu().detach())
        return (outputs[:, :, 0].cpu().detach(), latent[:].cpu().detach())


def evaluate_and_save(ttable, args, model):
    model.eval()

    results = []
    latents = []

    inp_batch = []
    batch_size = 1000
    for i in range(len(ttable.trial_table)):
        session_id, trial_id, _, _, n_neurons = ttable.trial_table[i]

        for neuron_id in range(n_neurons):
            inp_batch.append(load_sample(ttable, session_id, trial_id, neuron_id))

            if len(inp_batch) >= batch_size:
                pred, latent = predict(model, args, torch.stack(inp_batch))

                for sample_i in range(len(pred)):
                    results.append(pred[sample_i].numpy())
                    latents.append(latent[sample_i].numpy())

                inp_batch = []

    if len(inp_batch) > 0:
        pred, latent = predict(model, args, torch.stack(inp_batch))

        for sample_i in range(len(pred)):
            results.append(pred[sample_i].numpy())
            latents.append(latent[sample_i].numpy())

        inp_batch = []


    predictions = np.stack(results).astype(np.float16)
    latents = np.stack(latents).astype(np.float16)

    with open(args.data_dir + "/predictions_%s.npy" % args.brain_region, "wb") as f:
        np.save(f, predictions)
        np.save(f, latents)


def plot_average(ttable, predictions, args, session_id, neuron_id, fig, ax, target_outcome=1):
    """
    Load predictions relevant to session, neuron and target outcome.
    """
    pred_arr = []
    true_arr = []
    err_arr = []
    sample_i = 0
    for i in range(len(ttable.trial_table)):
        session_id_cur, trial_id_cur, outcome, _, n_neurons = ttable.trial_table[i]

        for neuron_id_cur in range(n_neurons):

            if session_id_cur == session_id and neuron_id_cur == neuron_id and outcome == target_outcome:
                true = load_sample(ttable, session_id_cur, trial_id_cur, neuron_id_cur)[:, 0]
                # pred = predict(load_sample(session_id, trial_id, neuron_id)).numpy()
                pred = predictions[sample_i]
                err = (true - pred) ** 2

                pred_arr.append(pred)
                true_arr.append(true)
                err_arr.append(err)

            sample_i += 1
    pred_arr = np.stack(pred_arr)  # (trial_num, n_steps)
    true_arr = np.stack(true_arr)  # (trial_num, n_steps)
    err_arr = np.stack(err_arr)  # (trial_num, n_steps)

    x = np.linspace(-3, 2, args.subsample_length)
    median_pred = np.median(pred_arr, axis=0)
    median_true = np.median(true_arr, axis=0)
    median_err = np.median(err_arr, axis=0)

    # === plot
    ax.clear()
    ax.plot(x, median_pred, color='b', label='predicted')
    ax.plot(x, median_true, color='g', label='true')
    ax.plot(x, median_err, color='r', label='error')

    # quantile_1_4 = np.quantile(pred_arr, 0.25, axis=0)
    # quantile_3_4 = np.quantile(pred_arr, 0.75, axis=0)
    # ax.fill_between(x, quantile_1_4, quantile_3_4, color='b', alpha=0.2)

    quantile_1_4 = np.quantile(true_arr, 0.25, axis=0)
    quantile_3_4 = np.quantile(true_arr, 0.75, axis=0)
    ax.fill_between(x, quantile_1_4, quantile_3_4, color='g', alpha=0.2)

    quantile_1_4 = np.quantile(err_arr, 0.25, axis=0)
    quantile_3_4 = np.quantile(err_arr, 0.75, axis=0)
    ax.fill_between(x, quantile_1_4, quantile_3_4, color='r', alpha=0.2)

    ax.set_title('Neuron ' + str(neuron_id))
    ax.set_xlabel('time aligned to cue, s', fontsize=10)
    ax.set_ylabel('Smoothed signal', fontsize=10)
    ax.set_xlim((-3, 2))
    ax.margins(0, 0.1)
    ax.legend()

    fig.tight_layout()
    # fig.savefig("plot/median-%02d-%02d-%01d.png"%(session_id, neuron_id, target_outcome))
    fig.canvas.draw_idle()