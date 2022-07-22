import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    with open('explanation_generation_rouge.json') as f:
        explanations = json.load(f)
    with open('focused_summarization_rouge.json') as f:
        summaries = json.load(f)

    damping_factors = ['0.8', '0.85', '0.9']
    similarity_thresholds = ['0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9']
    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(12,4)
    axs[0].set_title('Explanation Extraction')
    axs[1].set_title('Focused Summarization: Democrat')
    axs[2].set_title('Focused Summarization: Republican')
    axs[0].set(ylabel='ROUGE-1 F1')

    for i, damping_factor in enumerate(damping_factors):
        explanations_rouge1, explanations_rouge2, explanations_rougel = \
            [explanations[similarity_threshold][damping_factor][0] for similarity_threshold in explanations], \
            [explanations[similarity_threshold][damping_factor][1] for similarity_threshold in explanations], \
            [explanations[similarity_threshold][damping_factor][2] for similarity_threshold in explanations]
        dem_rouge1, dem_rouge2, dem_rougel = \
            [summaries[similarity_threshold][damping_factor]['democrat'][0] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['democrat'][1] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['democrat'][2] for similarity_threshold in summaries]
        rep_rouge1, rep_rouge2, rep_rougel = \
            [summaries[similarity_threshold][damping_factor]['republican'][0] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['republican'][1] for similarity_threshold in summaries], \
            [summaries[similarity_threshold][damping_factor]['republican'][2] for similarity_threshold in summaries]

        axs[0].plot(similarity_thresholds, explanations_rouge1, 'o-', label='DF={}'.format(damping_factor))
        axs[0].set_yticks(np.arange(0.29, 0.39, step=0.01))

        axs[1].plot(similarity_thresholds, dem_rouge1, 'o-', label='DF={}'.format(damping_factor))
        axs[1].set_yticks(np.arange(0.29, 0.39, step=0.01))

        axs[2].plot(similarity_thresholds, rep_rouge1, 'o-', label='DF={}'.format(damping_factor))
        axs[2].set_yticks(np.arange(0.29, 0.39, step=0.01))

        axs[i].set(xlabel='Similarity Threshold')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in axs.flat:
        #     ax.label_outer()

    plt.legend(bbox_to_anchor=(0.6, 0.7), loc='lower left', borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
    main()
