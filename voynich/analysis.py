from voynich.voynich_manuscript import VoynichManuscript

import fire
import os
from importlib import resources
import collections
import itertools
import datetime, time
import pickle
import numpy as np
import pandas as pd
from scipy import spatial, stats

import warnings

warnings.simplefilter("ignore")


class Corpus:
    def __init__(self, vm):
        self.corpus_df = self.get_corpus_dataframe(vm)
        self.word_counter = collections.Counter()
        self.vocab = list()
        self.num_words_in_text = None
        self.num_words_in_vocab = None
        self.df_word = self.get_word_dataframe()
        self.df_pattern = self.get_pattern_dataframe()

    def get_corpus_dataframe(self, vm):
        pass

    def get_word_dataframe(self):
        pass

    def get_pattern_dataframe(self):
        pass


class PBodyTypeCorpus(Corpus):
    def __init__(self, vm):
        self.df_corpus = self.get_corpus_dataframe(vm)
        self.word_counter = collections.Counter(
            [word for document in self.df_corpus.document for word in document]
        )
        self.vocab = sorted(list(set(self.word_counter.keys())))
        self.num_words_in_text = sum(self.word_counter.values())
        self.num_words_in_vocab = len(self.vocab)
        self.df_word = self.get_word_dataframe()
        self.df_pattern = self.get_pattern_dataframe()

    def get_corpus_dataframe(self, vm):
        d = collections.defaultdict(list)
        for page_id, page in vm.pages.items():
            document = []
            for body_id, body in page.bodies.items():

                if body.body_type != "P":
                    continue

                for line_id, line in body.lines.items():
                    document += line.words

                d["document_id"].append(body_id)
                d["document"].append(document)

        df = pd.DataFrame(data=d)
        df = df.set_index("document_id")

        return df

    def get_word_dataframe(self):
        word_col, loci_col, count_col, p_col = [], [], [], []

        word_count_dict = collections.defaultdict(int)
        word_to_loci_dict = collections.defaultdict(list)
        for document_id, row in self.df_corpus.iterrows():
            document = row.document
            for word_index, word in enumerate(document):
                word_count_dict[word] += 1
                word_to_loci_dict[word].append((document_id, word_index))

        for word in self.vocab:
            loci = word_to_loci_dict[word]
            count = word_count_dict[word]
            p = float(count / self.num_words_in_text)

            word_col.append(word)
            loci_col.append(loci)
            count_col.append(count)
            p_col.append(p)

        d = {
            "word": word_col,
            "loci": loci_col,
            "count": count_col,
            "p": p_col,
        }
        df = pd.DataFrame(data=d)
        df = df.set_index("word")

        return df

    def get_pattern_dataframe(self):
        (
            pattern_col,
            loci_col,
            count_vocab_col,
            count_text_col,
            p_vocab_col,
            p_text_col,
            word_indices_col,
        ) = ([], [], [], [], [], [], [])

        pattern_witnessed_in_vocab_word_dict = collections.defaultdict(bool)
        pattern_witnessed_in_text_word_dict = collections.defaultdict(bool)
        pattern_count_dict_vocab = collections.defaultdict(int)
        pattern_count_dict_text = collections.defaultdict(int)
        pattern_to_loci_dict = collections.defaultdict(list)
        for document_id, row in self.df_corpus.iterrows():
            document = row.document
            for index, word in enumerate(document):
                for x, y in itertools.combinations(range(len(word) + 1), r=2):
                    pattern = word[x:y]

                    pattern_to_loci_dict[pattern].append((document_id, index))

                    if not pattern_witnessed_in_vocab_word_dict[(word, pattern)]:
                        pattern_count_dict_vocab[pattern] += 1

                    if not pattern_witnessed_in_text_word_dict[
                        (document_id, index, pattern)
                    ]:
                        pattern_count_dict_text[pattern] += 1

                    pattern_witnessed_in_vocab_word_dict[(word, pattern)] = True
                    pattern_witnessed_in_text_word_dict[
                        (document_id, index, pattern)
                    ] = True

        for pattern, count_vocab in pattern_count_dict_vocab.items():
            loci = pattern_to_loci_dict[pattern]
            count_text = pattern_count_dict_text[pattern]
            p_vocab = float(count_vocab / self.num_words_in_vocab)
            p_text = float(count_text / self.num_words_in_text)

            pattern_col.append(pattern)
            loci_col.append(loci)
            count_vocab_col.append(count_vocab)
            count_text_col.append(count_text)
            p_vocab_col.append(p_vocab)
            p_text_col.append(p_text)

        d = {
            "pattern": pattern_col,
            "loci": loci_col,
            "count_vocab": count_vocab_col,
            "count_text": count_text_col,
            "p_vocab": p_vocab_col,
            "p_text": p_text_col,
        }
        df = pd.DataFrame(data=d)
        df = df.set_index("pattern")

        return df


class WordContextDataset:
    def __init__(self, corpus, left_context_index, right_context_index):

        # check input
        # TODO: add message
        assert isinstance(left_context_index, int)
        assert isinstance(right_context_index, int)
        assert left_context_index != 0 and right_context_index != 0
        assert left_context_index <= right_context_index

        # initialize
        self.corpus_class = corpus.__class__.__name__
        self.left_context_index = left_context_index
        self.right_context_index = right_context_index
        self.context_indices = [
            i for i in range(left_context_index, right_context_index + 1) if i != 0
        ]
        self.word_counter = collections.Counter(
            [word for document in corpus.df_corpus.document for word in document]
        )
        self.vocab = sorted(list(set(self.word_counter.keys())))
        self.num_words_in_text = sum(self.word_counter.values())
        self.num_words_in_vocab = len(self.vocab)
        self.target_word_to_contexts_dict = collections.defaultdict(list)
        self.target_word_to_context_word_counter_dict = dict()

        # build variables
        self.build(corpus)

    def build(self, corpus):
        # print message
        print(f"\nCreating word context dataset...")
        print(
            f"Parameters:\n\tcorpus_class={self.corpus_class}\n\tleft_context_index={self.left_context_index}\n\tright_context_index={self.right_context_index}"
        )
        print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}.")

        # create (1) target word to contexts dict and (2) target word to context word counter dict
        print(
            "Creating (1) target word to contexts dict and (2) target word to context word counter dict..."
        )
        for document_id, row in corpus.df_corpus.iterrows():
            document = row.document
            num_words_in_document = len(document)
            for target_index, target_word in enumerate(document):
                if (
                    target_word
                    not in self.target_word_to_context_word_counter_dict.keys()
                ):
                    self.target_word_to_context_word_counter_dict[
                        target_word
                    ] = collections.Counter()
                valid_context_word_indices = [
                    x
                    for x in list(np.array(self.context_indices) + target_index)
                    if x >= 0 and x < num_words_in_document
                ]
                self.target_word_to_contexts_dict[target_word].append(
                    (document_id, valid_context_word_indices)
                )
                for context_index in valid_context_word_indices:
                    context_word = document[context_index]
                    self.target_word_to_context_word_counter_dict[target_word][
                        context_word
                    ] += 1
        print("Done.")

        # create target word + context word frequency matrix
        print("Creating target word + context word frequency matrix...")
        self.target_word_context_word_frequency_matrix = np.zeros(
            (self.num_words_in_vocab, self.num_words_in_vocab)
        )  # row = target_word, column = context word
        for i, target_word in enumerate(self.vocab):
            context_word_counter = self.target_word_to_context_word_counter_dict[
                target_word
            ]
            n = sum(context_word_counter.values())
            for j, context_word in enumerate(self.vocab):
                if n == 0:
                    self.target_word_context_word_frequency_matrix[i, j] = 0.0
                else:
                    self.target_word_context_word_frequency_matrix[i, j] = float(
                        context_word_counter[context_word] / n
                    )

        # calculate word similarity matrix creation time
        print("Calculating word similartiy matrix creation time...")
        SAMPLE_SIZE = 100
        start_time = time.time()
        for i in range(SAMPLE_SIZE):
            _ = self.get_cosine_similarity(
                self.target_word_context_word_frequency_matrix[0, :],
                self.target_word_context_word_frequency_matrix[0, :],
            )
        end_time = time.time()
        expected_insertion_duration = (end_time - start_time) / SAMPLE_SIZE
        num_insertions = self.num_words_in_vocab + (
            self.num_words_in_vocab * (self.num_words_in_vocab - 1) / 2
        )
        num_seconds = num_insertions * expected_insertion_duration
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=num_seconds)
        print(
            f"Word similarity matrix creation should finish at approximately {finish_time.strftime('%H:%M:%S')}"
        )

        # create word similarity matrix
        print(f"Creating word similarity matrix...")
        self.word_similarity_matrix = np.zeros(
            (self.num_words_in_vocab, self.num_words_in_vocab)
        )
        for i, j in itertools.combinations_with_replacement(
            range(self.num_words_in_vocab), r=2
        ):
            score = self.get_cosine_similarity(
                self.target_word_context_word_frequency_matrix[i, :],
                self.target_word_context_word_frequency_matrix[j, :],
            )
            self.word_similarity_matrix[i, j] = score
            self.word_similarity_matrix[j, i] = score
        print("Done.")

        # print message
        print(f"Finished at {datetime.datetime.now().strftime('%H:%M:%S')}.\n")

    @staticmethod
    def get_cosine_similarity(x_i, x_j):
        return 1.0 - spatial.distance.cosine(x_i, x_j)

    def get_most_similar_words(self, word, top_n_similar_words):
        word_similarities = [
            self.word_similarity_matrix[self.vocab.index(word), j]
            for j in range(self.word_similarity_matrix.shape[0])
        ]
        word_indices_sorted = np.flip(np.argsort(word_similarities))

        word_similarities_sorted = [
            (self.vocab[word_index], word_similarities[word_index])
            for word_index in word_indices_sorted
            if (self.vocab[word_index] != word)
            and (not np.isnan(word_similarities[word_index]))
        ]
        if top_n_similar_words:
            word_similarities_sorted = word_similarities_sorted[:top_n_similar_words]

        return word_similarities_sorted


BinomialCDF = collections.namedtuple("BinomialCDF", ["x", "k", "n", "p", "prob"])


class Analysis:
    def __init__(self, corpus, word_context_dataset):
        assert (
            corpus.__class__.__name__ == word_context_dataset.corpus_class
        )  # TODO: add message

        self.corpus_class = word_context_dataset.corpus_class
        self.left_context_index = word_context_dataset.left_context_index
        self.right_context_index = word_context_dataset.right_context_index


class TargetWordContextWordPairAnalysis(Analysis):
    def __init__(self, corpus, word_context_dataset):
        # initialize
        super().__init__(corpus, word_context_dataset)

        # print message
        print(f"\nCreating target word + context word pair analysis...")
        print(
            f"Parameters:\n\tcorpus_class={self.corpus_class}\n\tleft_context_index={self.left_context_index}\n\tright_context_index={self.right_context_index}"
        )
        print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}.")

        # get df
        self.df = self.get_dataframe_of_probabilities_of_witnessed_target_word_context_word_pairs(
            corpus, word_context_dataset
        )

        # print message
        print(f"Finished at {datetime.datetime.now().strftime('%H:%M:%S')}.\n")

    @staticmethod
    def get_binomial_cdf_of_at_least_k_contexts_containing_context_word(
        corpus, contexts, context_word
    ):
        k = sum(
            [
                1
                if context_word
                in [
                    corpus.df_corpus.loc[document_id].document[word_index]
                    for word_index in word_indices
                ]
                else 0
                for document_id, word_indices in contexts
            ]
        )
        n = len(contexts)
        p = corpus.df_word.loc[context_word].p

        if k == 0:
            prob = 1.0
        else:
            prob = 1 - stats.binom.cdf(k - 1, n, p)

        return BinomialCDF(context_word, k, n, p, prob)

    @staticmethod
    def get_dataframe_of_probabilities_of_witnessed_target_word_context_word_pairs(
        corpus, word_context_dataset
    ):
        d = collections.defaultdict(list)
        already_explored_bool_dict = collections.defaultdict(bool)
        for i, target_word in enumerate(word_context_dataset.vocab):
            contexts = word_context_dataset.target_word_to_contexts_dict[target_word]
            unique_context_words = list(
                set(
                    [
                        corpus.df_corpus.loc[document_id].document[word_index]
                        for document_id, word_indices in contexts
                        for word_index in word_indices
                    ]
                )
            )
            for context_word in unique_context_words:
                if already_explored_bool_dict[(target_word, context_word)]:
                    continue
                binomial_cdf = TargetWordContextWordPairAnalysis.get_binomial_cdf_of_at_least_k_contexts_containing_context_word(
                    corpus, contexts, context_word
                )
                d["target_word"].append(target_word)
                d["context_word"].append(context_word)
                d["k"].append(binomial_cdf.k)
                d["n"].append(binomial_cdf.n)
                d["p"].append(binomial_cdf.p)
                d["prob"].append(binomial_cdf.prob)

                already_explored_bool_dict[(target_word, context_word)] = True

        df = pd.DataFrame(data=d)

        return df


class SimilarWordsPatternAnalysis(Analysis):
    def __init__(self, corpus, word_context_dataset, top_n_similar_words):
        # initialize
        super().__init__(corpus, word_context_dataset)
        self.top_n_similar_words = top_n_similar_words

        # print message
        print(f"\nCreating similar words pattern analysis...")
        print(
            f"Parameters:\n\tcorpus_class={self.corpus_class}\n\tleft_context_index={self.left_context_index}\n\tright_context_index={self.right_context_index}\n\ttop_n_similar_words={top_n_similar_words}"
        )
        print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}.")

        # get df
        self.df = (
            self.get_dataframe_of_probabilities_of_witnessed_patterns_in_similar_words(
                corpus, word_context_dataset, top_n_similar_words
            )
        )

        # print message
        print(f"Finished at {datetime.datetime.now().strftime('%H:%M:%S')}.\n")

    @staticmethod
    def get_binomial_cdf_of_at_least_k_words_containing_pattern(
        df_pattern, words, pattern, n
    ):
        pattern = pattern.lower()
        k = sum([1 if pattern in w.lower() else 0 for w in words])
        p = df_pattern.loc[pattern].p_text

        if k == 0:
            prob = 1.0
        else:
            prob = 1 - stats.binom.cdf(k - 1, n, p)

        return BinomialCDF(pattern, k, n, p, prob)

    @staticmethod
    def get_dataframe_of_probabilities_of_witnessed_patterns_in_similar_words(
        corpus, word_context_dataset, top_n_similar_words
    ):
        d = collections.defaultdict(list)
        already_explored_bool_dict = collections.defaultdict(bool)
        for i, word in enumerate(word_context_dataset.vocab):
            word_similarities = word_context_dataset.get_most_similar_words(
                word, top_n_similar_words=top_n_similar_words
            )
            word_similarities = [
                (w, sim) for w, sim in word_similarities if sim != 0.0
            ]  # only consider non-trivial similarities
            similar_words = [x[0] for x in word_similarities]
            for x, y in itertools.combinations(range(len(word) + 1), r=2):
                pattern = word[x:y]
                if already_explored_bool_dict[(word, pattern)]:
                    continue
                binomial_cdf = SimilarWordsPatternAnalysis.get_binomial_cdf_of_at_least_k_words_containing_pattern(
                    corpus.df_pattern, similar_words, pattern, n=top_n_similar_words
                )
                d["word"].append(word)
                d["pattern"].append(pattern)
                d["k"].append(binomial_cdf.k)
                d["n"].append(binomial_cdf.n)
                d["p"].append(binomial_cdf.p)
                d["prob"].append(binomial_cdf.prob)

                already_explored_bool_dict[(word, pattern)] = True

        df = pd.DataFrame(data=d)

        return df


def main(
    left_context_index,
    right_context_index,
    top_n_similar_words,
    corpus_class="PBodyTypeCorpus",
):
    # validate input variables
    try:
        CorpusClass = globals()[corpus_class]
        assert issubclass(CorpusClass, Corpus)
    except:
        raise Exception(f"")  # TODO

    try:
        assert isinstance(top_n_similar_words, int)
        assert top_n_similar_words >= 1
    except:
        raise Exception(
            "'top_n_similar_words' must be an integer greater than or equal to one."
        )

    # get Voynich manuscript
    with resources.path(
        "voynich.data.raw", "voynich_transcription_eva.txt"
    ) as filepath:
        vm_transcription_txt_filepath = filepath
    vm = VoynichManuscript(vm_transcription_txt_filepath)

    # get corpus
    corpus = CorpusClass(vm)

    # get word context dataset
    with resources.path("voynich.data.processed", "word_context_dataset") as filepath:
        word_context_dataset_dirpath = filepath
    if not os.path.isdir(word_context_dataset_dirpath):
        os.mkdir(word_context_dataset_dirpath)
    parameters_dict = {
        "corpus_class": corpus_class,
        "left_context_index": left_context_index,
        "right_context_index": right_context_index,
    }
    filename = f"{'-'.join([str(key)+'='+str(value) for key, value in parameters_dict.items()])}.pkl"
    save_filepath = os.path.join(word_context_dataset_dirpath, filename)
    if os.path.isfile(save_filepath):
        print(f"\nObject already exists: {save_filepath}")
        with open(save_filepath, "rb") as f:
            word_context_dataset = pickle.load(f)
    else:
        word_context_dataset = WordContextDataset(
            corpus, left_context_index, right_context_index
        )
        with open(save_filepath, "wb") as f:
            pickle.dump(word_context_dataset, f)

    # get target word + context word pair analysis
    with resources.path(
        "voynich.data.processed", "target_word_context_word_pair_analysis"
    ) as filepath:
        target_word_context_word_pair_analysis_dirpath = filepath
    if not os.path.isdir(target_word_context_word_pair_analysis_dirpath):
        os.mkdir(target_word_context_word_pair_analysis_dirpath)
    parameters_dict = {
        "corpus_class": corpus_class,
        "left_context_index": left_context_index,
        "right_context_index": right_context_index,
    }
    filename = f"{'-'.join([str(key)+'='+str(value) for key, value in parameters_dict.items()])}.pkl"
    save_filepath = os.path.join(
        target_word_context_word_pair_analysis_dirpath, filename
    )
    if os.path.isfile(save_filepath):
        print(f"\nObject already exists: {save_filepath}")
        with open(save_filepath, "rb") as f:
            target_word_context_word_pair_analysis = pickle.load(f)
    else:
        target_word_context_word_pair_analysis = TargetWordContextWordPairAnalysis(
            corpus, word_context_dataset
        )
        with open(save_filepath, "wb") as f:
            pickle.dump(target_word_context_word_pair_analysis, f)

    # get similar words pattern analysis
    with resources.path(
        "voynich.data.processed", "similar_words_pattern_analysis"
    ) as filepath:
        similar_words_pattern_analysis_dirpath = filepath
    if not os.path.isdir(similar_words_pattern_analysis_dirpath):
        os.mkdir(similar_words_pattern_analysis_dirpath)
    parameters_dict = {
        "corpus_class": corpus_class,
        "left_context_index": left_context_index,
        "right_context_index": right_context_index,
        "top_n_similar_words": top_n_similar_words,
    }
    filename = f"{'-'.join([str(key)+'='+str(value) for key, value in parameters_dict.items()])}.pkl"
    save_filepath = os.path.join(similar_words_pattern_analysis_dirpath, filename)
    if os.path.isfile(save_filepath):
        print(f"\nObject already exists: {save_filepath}")
        with open(save_filepath, "rb") as f:
            similar_words_pattern_analysis = pickle.load(f)
    else:
        similar_words_pattern_analysis = SimilarWordsPatternAnalysis(
            corpus, word_context_dataset, top_n_similar_words=top_n_similar_words
        )
        with open(save_filepath, "wb") as f:
            pickle.dump(similar_words_pattern_analysis, f)


if __name__ == "__main__":
    fire.Fire(main)
