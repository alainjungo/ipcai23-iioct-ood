import os

import pandas as pd
import matplotlib.pyplot as plt

import definitions as defs
import retinaqa.data.helper as data_hlp


def main():
    file_path = os.path.join(defs.REAL_OUT_DIR, '221011-140040_allmethods', 'scores_and_metrics.csv')

    df = pd.read_csv(file_path, index_col=0)
    case_df = df.groupby('case_id', as_index=False)[df.columns[1:]].agg('mean')
    assert len(df) / 10 == len(case_df)  # true since case_id is unique

    case_df.rename(columns={'is_ood': 'is_ood_from_file'}, inplace=True)
    case_df = data_hlp.add_case_information(case_df, simulated_ood_data=False)
    assert (case_df['is_ood'] == case_df['is_ood_from_file']).all()

    image_dict = data_hlp.load_all_images(defs.DATA_DIR, selection='test')

    case_df.sort_values('maha_score', inplace=True)

    low_score_but_ood = case_df.loc[case_df['is_ood']].iloc[:20]
    plot_cases(low_score_but_ood, image_dict, 'FN')

    high_score_and_ood = case_df.loc[case_df['is_ood']].iloc[-20:][::-1]
    plot_cases(high_score_and_ood, image_dict, 'TP')

    high_score_but_not_ood = case_df.loc[~case_df['is_ood']].iloc[-20:][::-1]
    plot_cases(high_score_but_not_ood, image_dict, 'FP')

    low_score_and_not_ood_fg = case_df.loc[~case_df['is_ood']].iloc[:20]
    plot_cases(low_score_and_not_ood_fg, image_dict, 'TN')


def plot_cases(selection_df: pd.DataFrame, image_dict: dict, title):
    selection_df = selection_df.reset_index()

    fig, axs = plt.subplots(len(selection_df), figsize=(10, 20))

    for i, row in selection_df.iterrows():
        case_id, score = row[['case_id', 'maha_score']]
        sequence = image_dict[case_id]

        axs[i].imshow(sequence, interpolation='nearest', vmax=255, vmin=0, aspect='auto')
        axs[i].spines[:].set_visible(False)
        axs[i].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        axs[i].set_ylabel(f'{case_id} ({score:.3f})', rotation=0, loc='center', labelpad=100)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
