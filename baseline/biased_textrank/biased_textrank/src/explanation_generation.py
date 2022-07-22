import json
from distutils.command.clean import clean

import nltk
from biased_summarization import select_top_k_texts_preserving_order, biased_textrank, get_sbert_embedding, \
    biased_textrank_ablation, vcosine
from rouge import Rouge
import numpy as np

rouge = Rouge()


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def get_forbidden_words():
    return [
        'Fake',
        'Barely True',
        'Barely False'
        'Mostly False',
        'True',
        'False',
        'Half True',
        'Half False',
        'Mostly True',
        'Pants on Fire',
        'Half Flip',
        'Full Flop'
    ]


def remove_forbidden_sentences(text):
    sentences = get_sentences(text)
    forbidden_words = get_forbidden_words()
    forbidden_sentences = []
    for sentence in sentences:
        for forbidden_word in forbidden_words:
            if forbidden_word.lower() in sentence.lower():
                forbidden_sentences.append(sentence)

    return ' '.join([sentence for sentence in sentences if sentence not in forbidden_sentences])


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if s and not s.isspace()]
    return sentences


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (
            ("'", '&#39;'),
            ('"', '&quot;'),
            ('>', '&gt;'),
            ('<', '&lt;'),
            ('&', '&amp;'),
            ('\'', '&rsquo;')
        )
    for code in htmlCodes:
        s = s.replace(code[1], code[0])
    return s


def clean_liar_data(split):
    with open('../../QAforMisinformation/data/liar/{}2.jsonl'.format(split)) as test_file:
        data_set_lines = list(test_file)

    data_set = [json.loads(line) for line in data_set_lines]

    problematic_datapoints_index = []
    for i, item in enumerate(data_set):
        file_name = item['json_file_id']
        try:
            with open('../../QAforMisinformation/data/liar/statements/{}'.format(file_name), encoding='utf-8') as statements_file:
                # item['statements'] = json.load(statements_file)['ruling_comments']
                item['statements'] = remove_html_tags(json.load(statements_file)['ruling_comments'])
            item['statements'] = ' '.join(item['statements'].split())
            item['statements'] = html_decode(remove_forbidden_sentences(item['statements']))
            possible_ruling_tokens = ['Our Ruling', 'Our ruling', 'our ruling', 'Summing up', 'Summing Up',
                                      'summing up']
            ruling_token = None
            for possible_ruling_token in possible_ruling_tokens:
                if possible_ruling_token in item['statements']:
                    ruling_token = possible_ruling_token
                    break
            if ruling_token:
                split_statements = item['statements'].split(ruling_token)
                item['new_justification'] = split_statements[1].strip()
                item['statements'] = split_statements[0].strip()
            else:
                statement_sentences = get_sentences(item['statements'])
                item['new_justification'] = ' '.join(statement_sentences[-5:])
                item['statements'] = ' '.join(statement_sentences[:-5])
        except:
            problematic_datapoints_index.append(i)
            print("Wrong file or file path for {}".format(file_name))

    for i in reversed(problematic_datapoints_index):
        del data_set[i]

    print('saving cleaned {} set file...'.format(split))
    with open('../data/liar/clean_{}2.json'.format(split), 'w') as f:
        f.write(json.dumps(data_set))

    return data_set


def get_liar_data(split):
    with open('../data/liar/clean_{}.json'.format(split)) as liar_file:
        liar_data = json.load(liar_file)

    return liar_data


def generate_textrank_explanations(split):
    dataset = get_liar_data(split)

    for claim in dataset:
        statements = get_sentences(claim['statements'])
        statements_embeddings = get_sbert_embedding(statements)
        bias = claim['claim']
        bias_embedding = get_sbert_embedding(bias)
        try:
            ranking = biased_textrank(statements_embeddings, bias_embedding, biased=False)
        except:
            print(statements)
            print('--------------------')
            print(bias)
        claim['generated_justification_biased'] = ' '.join(select_top_k_texts_preserving_order(statements, ranking, 4))

    print('saving generated {} set file...'.format(split))
    with open('../data/liar/clean_{}.json'.format(split), 'w') as f:
        f.write(json.dumps(dataset))


def evaluate_generated_explanations(split):
    dataset = get_liar_data(split)
    dataset = [claim for claim in dataset if len(get_sentences(claim['statements'])) > 3]

    rouge1 = []
    rouge2 = []
    rougel = []
    for claim in dataset:
        if 'generated_justification_biased' not in claim:
            continue
        reference = claim['new_justification']
        explanation = claim['generated_justification_biased']
        score = rouge.get_scores(explanation, reference)
        rouge1.append(score[0]['rouge-1']['f'])
        rouge2.append(score[0]['rouge-2']['f'])
        rougel.append(score[0]['rouge-l']['f'])

    print('Average ROUGE-1: {}'.format(np.mean(rouge1)))
    print('Average ROUGE-2: {}'.format(np.mean(rouge2)))
    print('Average ROUGE-l: {}'.format(np.mean(rougel)))


def ablation_study(split):
    dataset = get_liar_data(split)

    damping_factors = [0.8, 0.85, 0.9]
    similarity_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    summaries = [{} for item in dataset]
    for i, claim in enumerate(dataset):
        statements = get_sentences(claim['statements'])
        if len(statements) < 4:
            continue
        statements_embeddings = get_sbert_embedding(statements)
        bias = claim['claim']
        bias_embedding = get_sbert_embedding(bias)
        ranking = biased_textrank_ablation(statements_embeddings, bias_embedding, damping_factors=damping_factors, similarity_thresholds=similarity_thresholds)
        for similarity_threshold in similarity_thresholds:
            summaries[i][similarity_threshold] = {}
            for damping_factor in damping_factors:
                _ranking = ranking[similarity_threshold][damping_factor]
                summaries[i][similarity_threshold][damping_factor] = ' '.join(select_top_k_texts_preserving_order(statements, _ranking, 4))

    # saving results
    # with open('explanation_generation_ablation.json', 'w') as f:
    #     f.write(json.dumps(summaries))

    rouge_results = {}
    for similarity_threshold in similarity_thresholds:
        rouge_results[similarity_threshold] = {}
        for damping_factor in damping_factors:
            rouge1 = []
            rouge2 = []
            rougel = []
            for i, claim in enumerate(dataset):
                if len(get_sentences(claim['statements'])) < 4:
                    continue
                reference = claim['new_justification']
                explanation = summaries[i][similarity_threshold][damping_factor]
                score = rouge.get_scores(explanation, reference)
                rouge1.append(score[0]['rouge-1']['f'])
                rouge2.append(score[0]['rouge-2']['f'])
                rougel.append(score[0]['rouge-l']['f'])

            rouge_results[similarity_threshold][damping_factor] = [np.mean(rouge1), np.mean(rouge2), np.mean(rougel)]
            print('Similarity Threshold={}, Damping Factor={}'.format(similarity_threshold, damping_factor))
            print('ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}'.format(np.mean(rouge1), np.mean(rouge2), np.mean(rougel)))
            print('############################')

    with open('explanation_generation_rouge.json', 'w') as f:
        f.write(json.dumps(rouge_results))


def generate_embedding_similarity_explanations(split):
    dataset = get_liar_data(split)

    for claim in dataset:
        statements = get_sentences(claim['statements'])
        statements_embeddings = get_sbert_embedding(statements)
        bias = claim['claim']
        bias_embedding = get_sbert_embedding(bias)
        try:
            similarities = vcosine(bias_embedding, statements_embeddings)
            claim['generated_justification_embedding_similarity'] = ' '.join(
                select_top_k_texts_preserving_order(statements, similarities[0], 4))
        except:
            print(statements)
            print('--------------------')
            print(bias)

    print('saving generated {} set file...'.format(split))
    with open('../data/liar/clean_{}.json'.format(split), 'w') as f:
        f.write(json.dumps(dataset))


if __name__ == "__main__":
    generate_textrank_explanations('test2')
    evaluate_generated_explanations('test2')
    ablation_study('test2')
