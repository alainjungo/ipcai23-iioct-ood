import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import definitions as defs


def main():
    pred_dir = os.path.join(defs.SIMULATED_OUT_DIR, '221109-222725_allmethods_ratio')
    snr_pred_dir = os.path.join(defs.SIMULATED_OUT_DIR, '230124-123110_snr_ratio')

    pert_probabilities = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    file_paths = {p: os.path.join(pred_dir, f'scores_and_metrics_{str(p).replace(".", "")}.csv') for p in pert_probabilities}

    results = {}
    for prob, file_path in file_paths.items():
        df = pd.read_csv(file_path, index_col=0)
        ae = df['ae'].to_numpy()

        nb_drop = round(prob * len(df))

        est_score = df['estimation_score'].to_numpy()
        est_confidence = np.abs(est_score - 0.5)  # most uncertain are around 0.5 -> 0.-
        est_inds = np.argsort(est_confidence)
        ae_est_sorted = ae[est_inds]
        est_retained_by_ratio = ae_est_sorted[nb_drop:]

        maha_score = df['maha_score'].to_numpy()
        maha_inds = np.argsort(maha_score)
        ae_maha_sorted = ae[maha_inds]
        maha_retained_by_ratio = ae_maha_sorted[:len(df)-nb_drop]

        maha_img_score = df['maha_img_score'].to_numpy()
        maha_img_inds = np.argsort(maha_img_score)
        ae_maha_img_sorted = ae[maha_img_inds]
        maha_img_retained_by_ratio = ae_maha_img_sorted[:len(df)-nb_drop]

        sup_score = df['supervised_score'].to_numpy()
        sup_inds = np.argsort(sup_score)
        ae_sup_sorted = ae[sup_inds]
        sup_retained_by_ratio = ae_sup_sorted[:len(df)-nb_drop]

        glow_score = df['glow_score'].to_numpy()
        glow_inds = np.argsort(glow_score)
        ae_glow_sorted = ae[glow_inds]
        glow_retained_by_ratio = ae_glow_sorted[:len(df) - nb_drop] # higher score -> more ood (since bits per dimension [from nll])
        # glow_retained_by_ratio = ae_glow_sorted[nb_drop:]

        df_snr = pd.read_csv(os.path.join(snr_pred_dir, os.path.basename(file_path)), index_col=0)
        ae_snr = df_snr['ae'].to_numpy()
        snr = df_snr['snr'].to_numpy()
        snr_inds = np.argsort(snr)
        ae_snr_sorted = ae_snr[snr_inds]
        snr_retained_by_ratio = ae_snr_sorted[:len(df) - nb_drop]

        def extract(arr):
            return arr.mean(), len(arr)/len(df) if len(df) > 0 else 0

        results.setdefault('No-rejection', []).append(extract(ae))
        results.setdefault('MahaAD', []).append(extract(maha_retained_by_ratio))
        results.setdefault('Supervised', []).append(extract(sup_retained_by_ratio))
        results.setdefault('Glow', []).append(extract(glow_retained_by_ratio))
        results.setdefault('Uncertainty', []).append(extract(est_retained_by_ratio))
        results.setdefault('Raw-MahaAD', []).append(extract(maha_img_retained_by_ratio))
        results.setdefault('SNR', []).append(extract(snr_retained_by_ratio))

    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(11,6))

        for i, (name, res) in enumerate(results.items()):
            m, ratio = zip(*res)
            # ax.plot(pert_probabilities, m, label=name, c=f'C{i}', lw=5, alpha=0.75)
            if name == 'No-rejection':
                ax.plot(pert_probabilities, m, label=name, c=f'C{i}', ls='--', lw=3, alpha=0.75, zorder=1)
                # ax2.plot(pert_probabilities, ratio, label=name, c=f'C{i}', ls='--', lw=3, alpha=0.75)
            else:
                ax.plot(pert_probabilities, m, label=name, c=f'C{i}', lw=5, alpha=0.75, zorder=len(results) - i + 1)

        ax.set_ylabel('Mean absolute error (mm)', fontweight='bold')
        ax.set_xlabel('Ratio of perturbed samples', fontweight='bold')
        ax.tick_params(axis='both', which='major')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(visible=True, axis='y', alpha=0.75, zorder=0, color='lightgray')

        # ax.legend(prop={'size': 12})
        ax.legend(loc='center right', borderaxespad=-12, frameon=False)

        plt.tight_layout()
        os.makedirs(defs.PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(defs.PLOT_DIR, 'fig5_reject_samples_by_ratio.pdf'), bbox_inches='tight', pad_inches=0,
                    dpi=400)
        plt.show()


if __name__ == '__main__':
    main()
