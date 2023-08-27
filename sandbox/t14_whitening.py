'''

An experiment in whitening embeddings so they're normalized, and more semantically relevant.

https://arxiv.org/pdf/2103.15316.pdf

There are 2 things at play here:

1. classification

2. document retrieval

----------
WHITENING

PROS:
- the SVD whitening approach can shrink embeddings

CONS:
- whitening tends to hurt accuracy of cos sim ratings
- SVD is sensitive to epsilon, which keeps matrix from blowing up

'''

import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print()
def safe_div(numerator, denominator, epsilon=1e-9):
    """Safely divide two numbers."""
    sign = np.sign(denominator)
    return numerator / (denominator + epsilon * sign)

class WhiteningK:
    def __init__(self, k=None, epsilon=1e-5):
        self.kernel = None
        self.bias = None
        self.k = k  # Number of principal components to keep
        self.epsilon = epsilon  # Small constant for numerical stability

    def fit(self, vecs):
        """Compute the kernel and bias for whitening transformation."""

        # Check if vecs is a 2D array
        if len(vecs.shape) != 2:
            raise ValueError("Input array should be 2D.")

        # Check if vecs has more than one sample
        if vecs.shape[0] <= 1:
            raise ValueError("Input array should contain more than one sample.")

        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)

        if self.k is not None:
            u = u[:, :self.k]
            s = s[:self.k]

        self.kernel = np.dot(u, np.diag(safe_div(1.0, np.sqrt(s), self.epsilon)))  # Using safe_div for numerical stability
        self.bias = -mu

    def transform(self, vecs):
        """Transform the given vectors using the computed kernel and bias."""

        # Check if fit() has been called
        if self.kernel is None or self.bias is None:
            raise Exception("The fit method must be called before transform.")

        # Check if vecs is a 2D array
        if len(vecs.shape) != 2:
            raise ValueError("Input array should be 2D.")

        vecs = (vecs + self.bias).dot(self.kernel)
        return vecs / np.sqrt(safe_div(vecs ** 2, vecs ** 2).sum(axis=1, keepdims=True))

    def fit_transform(self, vecs):
        """Fit to the data, then transform it."""
        self.fit(vecs)
        return self.transform(vecs)


class DONTUSE_WhiteningK:
    def __init__(self, k, epsilon=1e-0):
        self.k = k
        self.bias = None
        self.kernel = None
        self.epsilon = epsilon  # stabilize div0

    def fit(self, embeddings, use_cov=True):
        self.bias = np.mean(embeddings, axis=0)
        if use_cov:
            # SVD on the covariance of the data. This is slower if the data is
            # small.
            cov_matrix = np.cov(embeddings - self.bias, rowvar=False)
            U, Lambda, _ = np.linalg.svd(cov_matrix)
            Lambda_inv_sqrt = np.diag(
                safe_div(1.0, np.sqrt(Lambda), self.epsilon))
            self.kernel = np.dot(U, Lambda_inv_sqrt)[:, :self.k]

        else:
            # SVD on data set (instead of covariance matrix). Can take longer to
            # calculate if there's a lot of data, but much quicker on smaller
            # datasets
            n = embeddings.shape[0]
            U, S, Vt = np.linalg.svd(
                embeddings - self.bias,
                full_matrices=False)
            Lambda_inv_sqrt = np.diag(
                safe_div(1.0,
                         np.sqrt(S[:self.k]**2 / (n - 1)),
                         self.epsilon))
            self.kernel = np.dot(Vt.T[:, :self.k],
                                 Lambda_inv_sqrt)

    def transform(self, embeddings):
        if self.bias is None or self.kernel is None:
            raise ValueError("fit method should be called before transform.")
        xs = np.dot(embeddings - self.bias, self.kernel)

        # # Don't Normalize
        # return xs

        # Normalize
        return xs / (xs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def fit_transform(self, embeddings):
        self.fit(embeddings)
        return self.transform(embeddings)


# MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
# MODEL_NAME = 'all-mpnet-base-v2'
MODEL_NAME = 'thenlper/gte-large'

try:
    already_loaded
except:
    model = SentenceTransformer(MODEL_NAME)
    already_loaded = True

K = 32
whitener = WhiteningK(K)

data = [
    {'label': 'art', 'text': 'A mosaic is an image made up of small pieces of colored material.'},
    {'label': 'art', 'text': 'Abstract art does not attempt to represent external reality.'},
    {'label': 'art', 'text': 'Art Deco features bold geometric shapes and bright colors.'},
    {'label': 'art', 'text': 'Art Nouveau is characterized by intricate linear designs and flowing curves.'},
    {'label': 'art', 'text': 'Baroque art is known for its detail, emotion, and grandeur.'},
    {'label': 'art', 'text': 'Color theory examines how colors interact and influence perception.'},
    {'label': 'art', 'text': 'Conceptual art focuses on ideas rather than aesthetic form.'},
    {'label': 'art', 'text': 'Cubism was pioneered by Picasso and Braque.'},
    {'label': 'art', 'text': 'Fresco is a technique where paint is applied on wet plaster.'},
    {'label': 'art', 'text': 'Impressionism is a 19th-century art movement.'},
    {'label': 'art', 'text': 'Pointillism uses small dots of color to form an image.'},
    {'label': 'art', 'text': 'Pop art emerged in the 1950s and often uses imagery from popular culture.'},
    {'label': 'art', 'text': 'Renaissance art flourished in Italy in the 15th and 16th centuries.'},
    {'label': 'art', 'text': 'Sculpture involves creating three-dimensional forms.'},
    {'label': 'art', 'text': 'Street art is visual art created in public locations.'},
    {'label': 'biology', 'text': 'A niche is an organism\'s role in an ecosystem.'},
    {'label': 'biology', 'text': 'Animal behavior can be innate or learned.'},
    {'label': 'biology', 'text': 'Biochemistry studies chemical processes in living organisms.'},
    {'label': 'biology', 'text': 'Biodiversity refers to the variety of life on Earth.'},
    {'label': 'biology', 'text': 'Cells are the basic unit of life.'},
    {'label': 'biology', 'text': 'DNA encodes genetic information.'},
    {'label': 'biology', 'text': 'Ecology is the study of how organisms interact with their environment.'},
    {'label': 'biology', 'text': 'Evolution explains the diversity of species.'},
    {'label': 'biology', 'text': 'Genetics studies how traits are passed from parents to offspring.'},
    {'label': 'biology', 'text': 'Metabolism is the set of chemical processes within cells.'},
    {'label': 'biology', 'text': 'Mitosis is cell division in somatic cells.'},
    {'label': 'biology', 'text': 'Natural selection is a mechanism of evolution.'},
    {'label': 'biology', 'text': 'Photosynthesis is how plants convert light into energy.'},
    {'label': 'biology', 'text': 'Proteins are made of amino acids.'},
    {'label': 'biology', 'text': 'Viruses are infectious particles smaller than bacteria.'},
    {'label': 'chemistry', 'text': 'Acids and bases react to form water and salts.'},
    {'label': 'chemistry', 'text': 'Catalysts speed up chemical reactions without being consumed.'},
    {'label': 'chemistry', 'text': 'Chemical reactions involve the rearrangement of atoms.'},
    {'label': 'chemistry', 'text': 'Covalent bonds share electrons between atoms.'},
    {'label': 'chemistry', 'text': 'Ionic bonds form between atoms that transfer electrons.'},
    {'label': 'chemistry', 'text': 'Isotopes are atoms with the same number of protons but different numbers of neutrons.'},
    {'label': 'chemistry', 'text': 'Matter exists in various states including solid, liquid, and gas.'},
    {'label': 'chemistry', 'text': 'Organic chemistry focuses on carbon-containing compounds.'},
    {'label': 'chemistry', 'text': 'Redox reactions involve the transfer of electrons between species.'},
    {'label': 'chemistry', 'text': 'The periodic table organizes chemical elements based on their properties.'},
    {'label': 'computer science', 'text': 'A compiler translates source code into machine code.'},
    {'label': 'computer science', 'text': 'API stands for Application Programming Interface.'},
    {'label': 'computer science', 'text': 'Algorithms are essential for solving computational problems.'},
    {'label': 'computer science', 'text': 'Big Data involves processing large sets of data.'},
    {'label': 'computer science', 'text': 'Cybersecurity focuses on protecting computer systems from theft or damage.'},
    {'label': 'computer science', 'text': 'Data structures include arrays, linked lists, and trees.'},
    {'label': 'computer science', 'text': 'Database normalization reduces data redundancy.'},
    {'label': 'computer science', 'text': 'Front-end development focuses on user interface and user experience.'},
    {'label': 'computer science', 'text': 'Graph theory is foundational for network analysis.'},
    {'label': 'computer science', 'text': 'Machine learning is a subset of artificial intelligence.'},
    {'label': 'computer science', 'text': 'Object-oriented programming uses classes and objects.'},
    {'label': 'computer science', 'text': 'Operating systems manage computer hardware and software.'},
    {'label': 'computer science', 'text': 'Python is a high-level programming language.'},
    {'label': 'computer science', 'text': 'Software development involves coding, debugging, and deployment.'},
    {'label': 'computer science', 'text': 'Web servers host websites and handle requests.'},
    {'label': 'history', 'text': 'Colonialism involved the domination of one country over another.'},
    {'label': 'history', 'text': 'Feudalism was a hierarchical system in medieval Europe.'},
    {'label': 'history', 'text': 'The Age of Enlightenment was characterized by emphasis on reason and science.'},
    {'label': 'history', 'text': 'The Civil Rights Movement aimed to end racial segregation in the United States.'},
    {'label': 'history', 'text': 'The Cold War was a period of political tension between the USA and the USSR.'},
    {'label': 'history', 'text': 'The Industrial Revolution transformed economies from agrarian to industrial.'},
    {'label': 'history', 'text': 'The Renaissance was a period of revival in art, literature, and learning.'},
    {'label': 'history', 'text': 'The Roman Empire was established in 27 BC.'},
    {'label': 'history', 'text': 'The Suez Canal was opened in 1869.'},
    {'label': 'history', 'text': 'World War II ended in 1945.'},
    {'label': 'literature', 'text': 'A haiku is a traditional form of Japanese poetry.'},
    {'label': 'literature', 'text': 'A metaphor is a figure of speech that directly compares two things.'},
    {'label': 'literature', 'text': 'A novella is shorter than a novel but longer than a short story.'},
    {'label': 'literature', 'text': 'Allegory is a story where characters and events symbolize broader ideas.'},
    {'label': 'literature', 'text': 'Epic poems often depict heroic deeds and adventures.'},
    {'label': 'literature', 'text': 'Irony is when the opposite of what you expect to happen occurs.'},
    {'label': 'literature', 'text': 'Narrative perspective can be first-person, third-person, or omniscient.'},
    {'label': 'literature', 'text': 'Satire uses humor to criticize or expose flaws.'},
    {'label': 'literature', 'text': 'Shakespeare wrote both plays and sonnets.'},
    {'label': 'literature', 'text': 'The genre of a story can be fantasy, drama, or comedy.'},
    {'label': 'mathematics', 'text': 'Boolean algebra is the basis of digital logic.'},
    {'label': 'mathematics', 'text': 'Calculus is the mathematical study of continuous change.'},
    {'label': 'mathematics', 'text': 'Combinatorics involves the arrangement and combination of objects.'},
    {'label': 'mathematics', 'text': 'Complex numbers have a real and an imaginary part.'},
    {'label': 'mathematics', 'text': 'Differential equations relate functions and their derivatives.'},
    {'label': 'mathematics', 'text': 'Geometry focuses on properties and dimensions of shapes and spaces.'},
    {'label': 'mathematics', 'text': 'Linear algebra deals with vector spaces and linear equations.'},
    {'label': 'mathematics', 'text': 'Mathematical proofs are logical arguments establishing a truth.'},
    {'label': 'mathematics', 'text': 'Matrices are rectangular arrays of numbers.'},
    {'label': 'mathematics', 'text': 'Number theory studies integers and their properties.'},
    {'label': 'mathematics', 'text': 'Probability theory helps us understand uncertain events.'},
    {'label': 'mathematics', 'text': 'Set theory is a branch of mathematical logic.'},
    {'label': 'mathematics', 'text': 'Statistics is the study of data collection, analysis, and interpretation.'},
    {'label': 'mathematics', 'text': 'Topology studies properties preserved under continuous transformations.'},
    {'label': 'mathematics', 'text': 'Trigonometry deals with the relations between angles and sides of triangles.'},
    {'label': 'physics', 'text': 'Acceleration is the rate of change of velocity.'},
    {'label': 'physics', 'text': 'Classical mechanics describes the motion of macroscopic objects.'},
    {'label': 'physics', 'text': 'Dark matter does not emit light or energy.'},
    {'label': 'physics', 'text': 'Electromagnetism is one of the four fundamental forces.'},
    {'label': 'physics', 'text': 'Entropy is a measure of disorder.'},
    {'label': 'physics', 'text': 'Fluid dynamics studies the flow of liquids and gases.'},
    {'label': 'physics', 'text': 'Friction is the force resisting relative motion.'},
    {'label': 'physics', 'text': 'Newton\'s laws describe the relationship between a body and the forces acting upon it.'},
    {'label': 'physics', 'text': 'Optics is the study of the behavior of light.'},
    {'label': 'physics', 'text': 'Particle physics studies the nature and behavior of subatomic particles.'},
    {'label': 'physics', 'text': 'Quantum mechanics is a fundamental theory in physics.'},
    {'label': 'physics', 'text': 'Sound waves are pressure waves transmitted through a medium.'},
    {'label': 'physics', 'text': 'The theory of relativity transformed our understanding of space and time.'},
    {'label': 'physics', 'text': 'Thermodynamics deals with heat and temperature.'},
    {'label': 'physics', 'text': 'Wavelength is the spatial period of a wave.'},
    {'label': 'psychology', 'text': 'Behaviorism focuses on observable behaviors.'},
    {'label': 'psychology', 'text': 'Clinical psychology deals with the diagnosis and treatment of mental disorders.'},
    {'label': 'psychology', 'text': 'Cognitive psychology studies mental processes like problem-solving.'},
    {'label': 'psychology', 'text': 'Developmental psychology studies psychological growth across the lifespan.'},
    {'label': 'psychology', 'text': 'Freud is known for his theories on the unconscious mind.'},
    {'label': 'psychology', 'text': 'Maslow\'s hierarchy of needs includes physiological and psychological needs.'},
    {'label': 'psychology', 'text': 'Neuropsychology studies the relationship between the brain and behavior.'},
    {'label': 'psychology', 'text': 'Personality psychology explores individual differences in behavior, emotions, and thought patterns.'},
    {'label': 'psychology', 'text': 'Positive psychology focuses on human strengths and well-being.'},
    {'label': 'psychology', 'text': 'Social psychology studies how individuals\' thoughts and actions are influenced by social context.'},
]

pd_data = pd.DataFrame(data)
pd_data = pd_data.sample(frac=1).reset_index(drop=True)

TEST_P = 0.3  # % of data to use for training-test split
split_index = int((1 - TEST_P) * len(pd_data))

# train_df = pd_data.iloc[:split_index].copy()
train_df = pd_data.iloc[:split_index].copy()
test_df = pd_data.iloc[split_index:].copy()


def report(name, xs):
    print(f'{name} | mean: {xs.mean():>.3f}, std: {xs.std():>.3f}, min: {xs.min():>.3f}, max: {xs.max():>.3f}')


##########
# Whitening Training

# Standard Embeddings
train_df['embedding'] = train_df['text'].apply(model.encode)
embeddings = np.vstack(train_df['embedding'].to_numpy())
report('train embeddings', embeddings)

# Whitened Embeddings
w_embeddings = whitener.fit_transform(embeddings)
train_df['w_embedding'] = list(w_embeddings)
report('train w_embeddings', w_embeddings)

report('WHITENER', whitener.kernel)

##########
# UMAP

# UMAP Embeddings
umap_model = UMAP(n_neighbors=10, min_dist=0.1, n_components=64)
umap_embeddings = umap_model.fit_transform(embeddings)
umap_bias = umap_embeddings.mean(axis=0)
umap_std = umap_embeddings.std(axis=0)
umap_embeddings = (umap_embeddings - umap_bias) / (umap_std + 1e-10)
train_df['umap_embedding'] = list(umap_embeddings)
report('train umap_embeddings', umap_embeddings)


##########
# Classifier Training

labels = pd_data['label'].unique()
mean_embeddings = {}
for label in labels:
    w_embeddings = np.vstack(train_df[train_df['label'] == label]['w_embedding'].to_numpy())
    mean_embeddings[label] = w_embeddings.mean(axis=0)


##########
# Build Test Embeddings

# Standard Embeddings
test_df['embedding'] = test_df['text'].apply(model.encode)
embeddings = np.vstack(test_df['embedding'].to_numpy())
report('test embeddings', embeddings)

# Whitened Embeddings
w_embeddings = whitener.transform(embeddings)
test_df['w_embedding'] = list(w_embeddings)
report('test w_embeddings', w_embeddings)

# UMAP Embeddings
umap_test_embeddings = umap_model.transform(embeddings)
umap_test_embeddings = (umap_test_embeddings - umap_bias) / (umap_std + 1e-10)
test_df['umap_embedding'] = list(umap_test_embeddings)
report('test umap_embeddings', umap_test_embeddings)


##########
# Test
#
# We test classification by running the cosine similarity of each test data to
# each of the training data and pick the label from the training data which most
# closely matches the cosine similarity data.

def classify(x, train_we, train_labels):
    cos_similarities = cosine_similarity([x], train_we)[0]
    most_similar_idx = np.argmax(cos_similarities)
    return train_labels[most_similar_idx]

def test_accuracy(test_df, train_df, col_name):
    train_labels = train_df['label'].tolist()
    train_we = np.vstack(train_df[col_name].to_numpy())
    correct = 0
    total = len(test_df)
    for idx, row in test_df.iterrows():
        test_embedding = row[col_name]
        true_label = row['label']
        predicted_label = classify(test_embedding, train_we, train_labels)
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / total
    return accuracy

print()
print(f'Test Embedding       : {test_accuracy(test_df, train_df, "embedding"):.4f}')
print(f'Test W Embedding     : {test_accuracy(test_df, train_df, "w_embedding"):.4f}')
print(f'Test UMAP Embedding  : {test_accuracy(test_df, train_df, "umap_embedding"):.4f}')

mean_embeddings_df = train_df.groupby('label')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
print(f'Test Mean Embedding  : {test_accuracy(test_df, mean_embeddings_df, "embedding"):.4f}')

mean_w_embeddings_df = train_df.groupby('label')['w_embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
print(f'Test Mean W_Embedding: {test_accuracy(test_df, mean_w_embeddings_df, "w_embedding"):.4f}')

mean_umap_embeddings_df = train_df.groupby('label')['umap_embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
print(f'Test Mean UMAP_Embedding: {test_accuracy(test_df, mean_umap_embeddings_df, "umap_embedding"):.4f}')



##########
# Visualization

VISUALIZE = False
if VISUALIZE:

    def cosine_similarity_matrix(mat):
        norm = np.linalg.norm(mat, axis=1)
        norm = np.reshape(norm, (-1, 1))
        normalized_mat = mat / norm
        return np.dot(normalized_mat, normalized_mat.T)

    def plot_heatmap(ax, data, title, xlabel, ylabel):
        # Remove diagonal elements to make the heatmap more interpretable
        data -= np.eye(data.shape[0])
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)

    # Define a list of (dataframe, column_name, title_suffix) tuples
    cases = [
        (train_df, 'embedding', 'Train Original'),
        (train_df, 'w_embedding', 'Train Whitened'),
        (test_df, 'embedding', 'Test Original'),
        (test_df, 'w_embedding', 'Test Whitened')
    ]

    # Create a 2x2 grid for plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Flatten axes for easy iteration
    flattened_axes = axes.flatten()

    for i, (df, column, title_suffix) in enumerate(cases):
        df = df.sort_values(by='label')
        ax = flattened_axes[i]
        embeddings = np.vstack(df[column])
        sim_matrix = cosine_similarity_matrix(embeddings)
        plot_heatmap(ax, sim_matrix, f'Cosine Similarity ({title_suffix})', f'{title_suffix} Embeddings', f'{title_suffix} Embeddings')

    plt.tight_layout()
    plt.show()

    # An Identity Property
    w = whitener.kernel
    cov = np.cov(np.vstack(train_df['embedding']), rowvar=False)
    assertion = w.T @ cov @ w
    plt.imshow(assertion)
    plt.show()
