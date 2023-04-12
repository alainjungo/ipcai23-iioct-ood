import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import definitions as defs
import retinaqa.data.helper as data_hlp
import retinaqa.eval.metrics as m


def main():
    pred_file = os.path.join(defs.REAL_OUT_DIR, '221011-140040_allmethods', 'scores_and_metrics.csv')
    snr_pred_file = os.path.join(defs.REAL_OUT_DIR, '230124-122527_snr', 'scores_and_metrics.csv')

    case_df, target = get_case_df(pred_file)

    results = []

    maha_score = case_df['maha_score'].to_numpy()
    auc, ap = get_results(target, maha_score)
    results.append(('MahaAD', auc, ap))

    sup_score = case_df['supervised_score'].to_numpy()
    auc, ap = get_results(target, sup_score)
    results.append(('Supervised', auc, ap))

    glow_score = case_df['glow_score'].to_numpy()
    auc, ap = get_results(target, glow_score)  # higher score -> more ood (since bits per dimension [from nll])
    # auc, ap = get_results(target, -glow_score)
    results.append(('Glow', auc, ap))

    est_score = case_df['estimation_score'].to_numpy()
    est_confidence = np.abs(est_score - 0.5)  # most uncertain are around 0.5 -> 0.-
    # est_uncertainty = est_confidence.max() - est_confidence
    est_uncertainty = 0.5 - est_confidence
    auc, ap = get_results(target, est_uncertainty)
    results.append(('Uncertainty', auc, ap))

    maha_img_score = case_df['maha_img_score'].to_numpy()
    auc, ap = get_results(target, maha_img_score)
    results.append(('Raw-MahaAD', auc, ap))

    snr_case_df, snr_target = get_case_df(snr_pred_file)
    assert (target == snr_target).all()
    snr = snr_case_df['snr'].to_numpy()
    auc, ap = get_results(target, snr)
    results.append(('SNR', auc, ap))

    summary = pd.DataFrame.from_records(results, columns=['', 'AUROC', 'AP'])
    summary = summary.round({'AUROC': 3, 'AP': 3})

    show_table(summary)
    plot(summary)


def show_table(summary):
    print(summary.to_latex(index=False, column_format='l@{\hskip 1em}c@{\hskip 1em}c'))


def plot(summary):
    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.15
        offsets = np.arange(0, len(summary)) - 0.5*(len(summary) - 1)
        x = np.arange(2)
        summary.set_index('', inplace=True)
        for i, (index, series) in enumerate(summary.iterrows()):
            ax.bar(x+offsets[i]*width, series, width, label=index, color=f'C{i+1}', zorder=2)
            for j, (_, value) in enumerate(series.items()):
                ax.text(x[j]+offsets[i]*width, value+0.01, value, rotation=45, rotation_mode='anchor')
        ax.set_xticks(x, summary.columns.tolist(), fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(visible=True, axis='y', alpha=0.75, zorder=0, color='lightgray')
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([0., 1.])
        ax.legend(loc='center right', borderaxespad=-12, frameon=False)
        fig.tight_layout()

        os.makedirs(defs.PLOT_DIR, exist_ok=True)
        plt.savefig(os.path.join(defs.PLOT_DIR, 'fig7_real_ood_barplot.pdf'), bbox_inches='tight', dpi=400)
        plt.show()


def get_results(target, score):
    res = m.evaluate_ood(target, score)
    auc, ap = res['auc'], res['ap']
    return auc, ap


def get_case_df(file_path):
    df = pd.read_csv(file_path, index_col=0)
    # since predictions were repeated to the A-scan level in order to store the estimation as well
    case_df = df.groupby('case_id', as_index=False)[df.columns[1:]].agg('mean')

    assert len(df) / 10 == len(case_df)  # since sequence length is 10 on test set

    case_df.rename(columns={'is_ood': 'is_ood_from_file'}, inplace=True)
    case_df = data_hlp.add_case_information(case_df, simulated_ood_data=False)

    assert (case_df['is_ood'] == case_df['is_ood_from_file']).all()
    target = case_df['is_ood'].to_numpy()

    return case_df, target


if __name__ == '__main__':
    main()
