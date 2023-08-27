from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class SimpleNN:
    def __init__(self, input_dim, label_embedding_dim, string_labels, lr=0.01, epochs=30):
        self.input_dim = input_dim
        self.label_embedding_dim = label_embedding_dim
        self.string_labels = string_labels
        self.num_classes = len(string_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(string_labels)}
        self.epochs = epochs
        self.lr = lr

        self.model = nn.Sequential(

            # One Layer
            nn.Linear(input_dim, label_embedding_dim),
            nn.Tanh(),
            # nn.BatchNorm1d(label_embedding_dim),  # regularization

        )

        self.label_embeddings = torch.randn(self.num_classes, label_embedding_dim)
        self.label_embeddings *= label_embedding_dim / torch.norm(self.label_embeddings, dim=1, keepdim=True)

        self.optimizer = optim.AdamW(list(self.model.parameters()),
                                     lr=lr,
                                     weight_decay=1e-2)

    def cos_loss(self, output, target):
        dot_product = (output * target).sum(dim=-1)
        norm_output = torch.norm(output, dim=-1)
        norm_target = torch.norm(target, dim=-1)
        cosine_similarity = dot_product / (norm_output * norm_target)
        cosine_distance = 1 - cosine_similarity  # Converting similarity to distance
        return cosine_distance.mean()

    def fit(self, X, string_labels):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        label_indices = [self.label_to_index[label] for label in string_labels]
        y_tensor = self.label_embeddings[label_indices]
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = self.cos_loss(output, y_tensor)
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, loss: {loss.detach().cpu()}')
            loss.backward()
            self.optimizer.step()

    def transform(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            output = self.model(X_tensor)
        return output.numpy()

    def fit_transform(self, X, string_labels):
        self.fit(X, string_labels)
        return self.transform(X)

# # Sample usage
# nn = SimpleNN(input_dim=64, label_embedding_dim=64, string_labels=['label_1', 'label_2'])
# train_embeddings = np.random.rand(100, 64)  # Your train embeddings
# train_string_labels = np.random.choice(['label_1', 'label_2'], size=100)  # Random string labels for each data point

# # Fit and transform
# nn.fit(train_embeddings, train_string_labels)
# transformed_data = nn.transform(np.random.rand(10, 64))

# print(transformed_data)



# MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
# MODEL_NAME = 'all-mpnet-base-v2'
MODEL_NAME = 'thenlper/gte-large'

try:
    already_loaded
except:
    model = SentenceTransformer(MODEL_NAME)
    already_loaded = True

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

TEST_P = 0.2  # % of data to use for training-test split
split_index = int((1 - TEST_P) * len(pd_data))

# train_df = pd_data.iloc[:split_index].copy()
train_df = pd_data.iloc[:split_index].copy()
test_df = pd_data.iloc[split_index:].copy()


def report(name, xs):
    print(f'{name} | mean: {xs.mean():>.3f}, std: {xs.std():>.3f}, min: {xs.min():>.3f}, max: {xs.max():>.3f}')


# Standard Embeddings
train_df['embedding'] = train_df['text'].apply(model.encode)
embeddings = np.vstack(train_df['embedding'].to_numpy())
report('train embeddings', embeddings)

# Prepare data for SimpleNN
train_embeddings = np.vstack(train_df['embedding'].to_numpy())
train_string_labels = train_df['label'].tolist()

# Initialize and train SimpleNN
nn = SimpleNN(input_dim=1024, label_embedding_dim=64, string_labels=list(set(train_df['label'])))
nn.fit(train_embeddings, train_string_labels)

test_df['embedding'] = test_df['text'].apply(model.encode)
embeddings = np.vstack(test_df['embedding'].to_numpy())
report('test embeddings', embeddings)
test_embeddings = np.vstack(test_df['embedding'].to_numpy())
test_string_labels = test_df['label'].tolist()


# Transform and evaluate SimpleNN embeddings
transformed_train_embeddings = nn.transform(train_embeddings)
transformed_test_embeddings = nn.transform(test_embeddings)
train_df['nn_embedding'] = list(transformed_train_embeddings)
test_df['nn_embedding'] = list(transformed_test_embeddings)


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
        else:
            pass
            # print(f'miss: pred={predicted_label}, true={true_label}, {row["text"]}')
    accuracy = correct / total
    return accuracy

print()
print(f'Test Embedding       : {test_accuracy(test_df, train_df, "embedding"):.4f}')

mean_embeddings_df = train_df.groupby('label')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
print(f'Test Mean Embedding  : {test_accuracy(test_df, mean_embeddings_df, "embedding"):.4f}')


# Run test_accuracy using transformed_test_embeddings (nn_embedding column)
print(f'Test NN Embedding    : {test_accuracy(test_df, train_df, "nn_embedding"):.4f}')

# If you'd like, you can also calculate the mean of the transformed embeddings
mean_nn_embeddings_df = train_df.groupby('label')['nn_embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
print(f'Test Mean NN_Embedding: {test_accuracy(test_df, mean_nn_embeddings_df, "nn_embedding"):.4f}')
