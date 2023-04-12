import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import definitions as defs
import retinaqa.eval.metrics as m


def main():
    pred_file = os.path.join(defs.SIMULATED_OUT_DIR, '221109-222725_allmethods_ratio', 'scores_and_metrics_05.csv')
    df = pd.read_csv(pred_file, index_col=0)

    all_perts = ['stripe', 'rect', 'zoom', 'shift', 'noise', 'contr', 'smooth', 'intsh']
    pert_names = ['Stripes', 'Rectangle', 'Zoom', 'Shift', 'Noise', 'Contrast', 'Smoothing', 'Intensity']

    aucs = {}
    for pert in all_perts:

        selection_mask = (df['pert_name'] == pert) | (df['pert_name'] == 'none')
        selection = df[selection_mask]

        maha_res = m.evaluate_ood(selection['pert_name'] != 'none', selection['maha_score'])
        aucs.setdefault('MahaAD', []).append(maha_res['auc'])

        sup_res = m.evaluate_ood(selection['pert_name'] != 'none', selection['supervised_score'])
        aucs.setdefault('Supervised', []).append(sup_res['auc'])

    train_perts = ['shift', 'smooth', 'noise', 'intsh']
    test_perts = [p for p in all_perts if p not in train_perts]

    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.2
        x = np.arange(len(test_perts))
        offsets = np.arange(0, len(aucs)) - 0.5 * (len(aucs) - 1)
        for i, (method, vals) in enumerate(aucs.items()):
            val_selection = [vals[all_perts.index(p)] for p in test_perts]
            ax.bar(x + offsets[i] * width, val_selection, width, label=method, color=f'C{i+1}', zorder=2)
        ax.set_xticks(x, [pert_names[all_perts.index(p)] for p in test_perts], fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylabel('AUROC', fontweight='bold')
        ax.set_ylim([0.5, 1.])
        ax.grid(visible=True, axis='y', alpha=0.75, zorder=0, color='lightgray')

        ax.legend(loc='center right', borderaxespad=-10, frameon=False)
        fig.tight_layout()

        os.makedirs(defs.PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(defs.PLOT_DIR, 'fig6_held_out_perturbation_barplot.pdf'), bbox_inches='tight', dpi=400)

        plt.show()


if __name__ == '__main__':
    main()
