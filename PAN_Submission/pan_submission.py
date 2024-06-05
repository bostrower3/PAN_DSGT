import json
import torch
import os
import bz2
import nltk
import xgboost
from nltk import sent_tokenize
from nltk import word_tokenize, pos_tag, ne_chunk
import spacy
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import RobertaTokenizer, RobertaModel
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import RobertaTokenizer
import torch_geometric.data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import pandas as pd
import torch_geometric as pyg
import re
import multiprocessing
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import pickle

# Load the English NER model
nlp = spacy.load("en_core_web_trf")


# Path to the compressed file
compressed_file_path = '/app/enwiki_20180420_100d.pkl.bz2'

# Path to save the extracted model file
extracted_file_path = '/app/enwiki_20180420_100d.pkl'

# Extract the model file
with open(extracted_file_path, 'wb') as new_file, bz2.BZ2File(compressed_file_path, 'rb') as file:
    for data in iter(lambda : file.read(100 * 1024), b''):
        new_file.write(data)

from wikipedia2vec import Wikipedia2Vec
wiki2vec = Wikipedia2Vec.load(extracted_file_path)






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer outside of functions

Bert_Base_Dir = '/app/BertBaseUncased'
RoBerta_Base_Dir = '/app/RoBERTA'
tokenizer = BertTokenizer.from_pretrained(Bert_Base_Dir)
model = BertForNextSentencePrediction.from_pretrained(Bert_Base_Dir).to(device)

# Load pre-trained RoBERTa tokenizer and model
Roberta_tokenizer = RobertaTokenizer.from_pretrained(RoBerta_Base_Dir)
Roberta_model = RobertaModel.from_pretrained(RoBerta_Base_Dir).to(device)



def return_json(file_path):
  # Create an empty dictionary to store the data
  json_data_dict = {}

  # Open the JSONL file
  with open(file_path, 'r') as f:
      # Read each line and parse it as JSON
      for index, line in enumerate(f):
          json_data = json.loads(line)

          # Add each JSON object to the dictionary with its index as key
          json_data_dict[json_data['id']] = {'text1:':json_data['text1'], 'text2:':json_data['text2']}
  return json_data_dict




def calculate_nsp_score(sentence1, sentence2, model=model, tokenizer=tokenizer):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
      outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    nsp_score = probabilities[0][0].item()
    return nsp_score

def get_roberta_embedding(sentences, model=Roberta_model, tokenizer=Roberta_tokenizer):
    # Tokenize sentences into individual words
    tokens = tokenizer.tokenize(sentences)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids_tensor = torch.tensor([input_ids]).to(device)

    # Get the Roberta model output for the input tensor
    with torch.no_grad():
        output = model(input_ids_tensor).last_hidden_state[:, 0, :]

    return output
#try detaching  tensor, happens when you
def get_ner_embeddings(sentences, nlp=nlp):
    doc = nlp(sentences)
    ner_embeddings = []
    for ent in doc.ents:
        ent_embedding = get_roberta_embedding(ent.text)
        ner_embeddings.append((ent, ent_embedding))
    return ner_embeddings

def doc_ner_embeddings(document):
    sentences = sent_tokenize(document)
    doc_ents = {}
    prev_sent = ""
    nsp_scores = []
    for i, sentence in enumerate(sentences):
        ner_embeddings = get_ner_embeddings(sentence)
        doc_ents[i] = ner_embeddings
        if i > 0:
            nsp_score = calculate_nsp_score(prev_sent, sentence)
            nsp_scores.append(nsp_score)
        prev_sent = sentence
    return doc_ents, nsp_scores

def wiki2vecs(entity):
    try:
        wiki = torch.tensor(wiki2vec.get_entity_vector(entity))
    except:
        try:
            wiki = torch.tensor(wiki2vec.get_word_vector(entity))
        except:
            wiki = torch.zeros(100)
    return wiki

def CLS_token(document, model=Roberta_model, tokenizer=Roberta_tokenizer):
    input_ids = tokenizer.encode(document, add_special_tokens=True, return_tensors='pt',max_length=512,truncation=True).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state
    cls_representation = hidden_states[:, 0, :]
    return cls_representation


def process_document(texts,CLS_tokens,document_list):
  indx = 0
  for indx in range(len(document_list)):
    pairdoc = {}
    pair_cls = {}
    for key in document_list[indx].keys():
      temp = doc_ner_embeddings(document_list[indx][key])
      check = sum(len(temp[0][key]) for key in temp[0].keys())
      if check < 3:
          print(f"Document {indx} skipped")
          pairdoc[key] = -9
          pair_cls[key] = -9
          continue

      pairdoc[key] = temp
      pair_cls[key] = CLS_token(document_list[indx][key])

    texts[indx]  = pairdoc
    CLS_tokens[indx] = pair_cls
    indx += 1
    print(indx)
  return texts,CLS_tokens



def cosine_sim(vec1, vec2):

  cosine_similarity = F.cosine_similarity(vec1, vec2, dim=1)
  return cosine_similarity

def flat_vec(vec):
  # Flatten the vectors along the specified axis
  return vec

def adjacency_matrix_to_edge_index(adjacency_matrix):
    num_nodes = adjacency_matrix.shape[0]
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


def Return_inputs(combo,similarity = 0.9,wiki2vec = False):
  if combo[0] == -9:
    return -9

  texts = combo[0]
  nsp_scores = combo[1]
  ##Parse Node Pairs
  Node_Pairs = {}
  for i in range(len(texts)):
    for j in range(len(texts)):
      if i == j: #inner Sentence Pairs

        for k in range(len(texts[i])-1):
          for l in range(k+1,len(texts[j])):
            if k!=l:
              pair_1 = str(texts[i][k][0])
              pair_2 = str(texts[j][l][0])
              pair_1Robeta = flat_vec(texts[i][k][1])
              pair_2Robeta = flat_vec(texts[j][l][1])
              if wiki2vec:
                pair_1wiki = wiki2vecs(str(pair_1)).to(device)
                pair_2wiki = wiki2vecs(str(pair_2)).to(device)
                Node_Pairs[(pair_1,pair_2)] = (torch.concat((pair_1Robeta,pair_1wiki.unsqueeze(0)),dim = 1)
                ,torch.concat((pair_2Robeta,pair_2wiki.unsqueeze(0)),dim = 1))
              else:
                Node_Pairs[(pair_1,pair_2)] = (pair_1Robeta,pair_2Robeta)

      if i != j: #inter Sentence Pairs
        for k in range(len(texts[i])):
          for l in range(len(texts[j])):

            pair_1 = str(texts[i][k][0])
            pair_2 = str(texts[j][l][0])
            pair_1Robeta = flat_vec(texts[i][k][1])
            pair_2Robeta = flat_vec(texts[j][l][1])
            #check if similarity between inter sentence pairs is close enough
            sim = cosine_sim(texts[i][k][1],texts[j][l][1])
            if torch.Tensor.cpu(sim).numpy().sum()/sim.size()[0] > similarity:
              if wiki2vec:
                pair_1wiki = wiki2vecs(str(pair_1)).to(device)
                pair_2wiki = wiki2vecs(str(pair_2)).to(device)
                Node_Pairs[(pair_1,pair_2)] = (torch.concat((pair_1Robeta,pair_1wiki.unsqueeze(0)),dim = 1)
                ,torch.concat((pair_2Robeta,pair_2wiki.unsqueeze(0)),dim = 1))
              else:
                Node_Pairs[(pair_1,pair_2)] = (pair_1Robeta,pair_2Robeta)


  ##Remove duplicated reversed pairs
  unique_data = {}
  for key, value in Node_Pairs.items():
    sorted_key = tuple(sorted(key))
    unique_data[sorted_key] = value

  ##create a dictionary for each singular entity
  new_dict = {}

  for embeddings, values in unique_data.items():
      for embed, val in zip(embeddings, values):
          if embed not in new_dict:
              new_dict[embed] = []
          new_dict[embed].append(val)


  ##Have to average contextual representations
  averaged_dict = {}

  for embed, values in new_dict.items():
      # Stack the tensors along a new dimension (dim=0) and calculate the mean along that dimension
      averaged_tensor = torch.stack(values, dim=0).mean(dim=0)
      averaged_dict[embed] = averaged_tensor

  ##Adjaceny Matrix
  # Extract unique entities
  edges_values = unique_data
  entities = set(entity for edge in edges_values.keys() for entity in edge)

  # Create a mapping from entities to indices
  entity_to_index = {entity: idx for idx, entity in enumerate(entities)}

  # Initialize an adjacency matrix with zeros
  num_entities = len(entities)
  adjacency_matrix = np.zeros((num_entities, num_entities))

  # Fill in the adjacency matrix based on the values in your dictionary
  for (entity1, entity2), (val1, val2) in edges_values.items():
      idx1, idx2 = entity_to_index[entity1], entity_to_index[entity2]
      # You can choose how to combine val1 and val2, for example, summing them
      adjacency_matrix[idx1, idx2] += 1
      adjacency_matrix[idx2, idx1] += 1

  ##Degree Matrix
  # Calculate the degree for each node
  degrees = np.sum(adjacency_matrix, axis=1)

  # Create a degree matrix
  degree_matrix = np.diag(degrees)

  #Inverse and set np.inf to 0's
  inversed_d = degree_matrix**(-1/2)
  inversed_d[np.isinf(inversed_d)] = 0

  ##Create Matrix of word embeddings
  H_e = np.zeros((len(averaged_dict),averaged_dict[list(averaged_dict.keys())[0]].shape[1]))
  for i in range(len(averaged_dict)):
    H_e[i] = torch.Tensor.cpu(averaged_dict[list(averaged_dict.keys())[i]])


  A_squiggle = inversed_d@adjacency_matrix@inversed_d


  ##Create padded sequence of entities by sentence
  doocs = []
  lengths = []
  for keys in texts.keys():
    sents = []
    length = 0
    for pair in texts[keys]:
      sents.append(entity_to_index[str(pair[0])])
      length += 1
    lengths.append(length)
    doocs.append(sents)
  #pad_sequence(doocs,batch_first=True)

  return A_squiggle,H_e,doocs,entity_to_index,lengths,adjacency_matrix_to_edge_index(adjacency_matrix),nsp_scores

def contains_nested_list(lst):
    for item in lst:
        if isinstance(item, list):
            return True
    return False


def Dataset_Curation(texts,similarity = 0.9,wiki2vec = False):
  Initial_embeddings = {}
  entity_sentence = {}
  length = {}
  nsp_score = {}
  adj_matrixs = {}

  for key in range(len(texts)):
    Initial_embeddings_pair = {}
    entity_sentence_pair = {}
    length_pair = {}
    nsp_score_pair = {}
    adj_matrixs_pair = {}
    for text in list(texts[key].keys()):
      try:
        A_squiggle,H_e,entity_sentences,ei_index,lengths,adj_matrix,nsp_scores = Return_inputs(texts[key][text],similarity = similarity,wiki2vec = wiki2vec)
        if contains_nested_list(nsp_scores):
          print(key,'continued')
          continue
        if len(nsp_scores) == 0:
          print(key,'continued')
          nsp_scores = nsp_scores[0]
          continue

        Initial_embeddings_pair[text] = H_e
        entity_sentence_pair[text] = entity_sentences
        length_pair[text] = lengths
        nsp_score_pair[text] = nsp_scores
        adj_matrixs_pair[text] = adj_matrix
        print(key,text)
      except:
        Initial_embeddings[text] = -9
        entity_sentence[text] = -9
        length[text] = -9
        nsp_score[text] = -9
        adj_matrixs[text] = -9
        print(key,text, "failed")


    Initial_embeddings[key] = Initial_embeddings_pair
    entity_sentence[key] = entity_sentence_pair
    length[key] = length_pair
    nsp_score[key] = nsp_score_pair
    adj_matrixs[key] = adj_matrixs_pair

  return Initial_embeddings,entity_sentence,length,nsp_score,adj_matrixs

class CleanedGraphData(torch_geometric.data.Dataset):
    def __init__(self, Initial_embeddings,entity_sentences,lengths,nsp_scores,adj_matrix,CLS_tokens,sim = 0.9,wiki2vec = False):
        self.Initial_embeddings = Initial_embeddings
        self.entity_sentences = entity_sentences
        self.lengths = lengths
        self.nsp_scores = nsp_scores
        self.CLS_tokens = CLS_tokens
        self.adj_matrix = adj_matrix
        self.keys = list(Initial_embeddings.keys())

    def len(self):
        return 1

    def get(self, idx):



        key = 'text1:'
        try:
          Graph_data =torch_geometric.data.Data(x=torch.tensor(self.Initial_embeddings[key],dtype=torch.float),edge_index=self.adj_matrix[key],
                                                entity_sentences = self.entity_sentences[key],
                                                lengths = self.lengths[key],
                                                nsp_scores = self.nsp_scores[key],
                                                CLS_tokens = self.CLS_tokens[key]
                                              )
        except:
          Graph_data =torch.tensor(-9)

        key = 'text2:'
        try:
          Graph_data2 =torch_geometric.data.Data(x=torch.tensor(self.Initial_embeddings[key],dtype=torch.float),edge_index=self.adj_matrix[key],
                                              entity_sentences = self.entity_sentences[key],
                                              lengths = self.lengths[key],
                                              nsp_scores = self.nsp_scores[key],
                                              CLS_tokens = self.CLS_tokens[key]
                                             )
        except:
          Graph_data2 =torch.tensor(-9)

        return Graph_data.to(device),Graph_data2.to(device)



def pad_sequences(batch,input):
  max_length = 0
  for doc in batch:
    if len(doc) > max_length:
      max_length = len(doc)
  new_batch = []
  for doc in batch:
    new_batch.append(
        torch.cat((doc.to(device),torch.zeros((max_length-len(doc),input)).to(device) ),dim = 0)
    )
  return torch.stack(new_batch)


def pad_sequences_NSP(botch):
  max_length = 0
  for doc in botch:
    if len(doc) > max_length:
      max_length = len(doc)
  new_botch = []
  for doc in botch:
    if max_length != len(doc):
      new_botch.append(
        torch.cat((torch.tensor(doc,dtype = torch.float64).to(device),torch.zeros((max_length-len(doc))).to(device) ),dim = 0)
    )
    else:
      new_botch.append(torch.tensor(doc,dtype = torch.float64).to(device))
  return torch.stack(new_botch)

# Define a custom GCN model
class MyGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_convs,wiki2vec = False):
        super(MyGCN, self).__init__()
        self.input_dim = input_dim
        if wiki2vec:
            self.input_dim += 100



        #define variable length GCN Blocks
        self.ConvBlocks = nn.ModuleList([
            GCNConv(self.input_dim, self.input_dim,bias = False) for _ in range(num_convs)
        ])

        #Sentence Representation
        self.sentence_linear = nn.Linear(self.input_dim,self.input_dim,bias = True)
        #LSTM for graph-enhanced node representations
        self.lstm = nn.LSTM(self.input_dim,hidden_dim,bidirectional=False,batch_first=True)
        #Linear Layer to reduce dimensionality of LSTM
        self.linear_dim_reduct = nn.Linear(input_dim,256)
        #final Layer
        self.linear = nn.Linear(512,output_dim)


    def forward(self, Data):


        # Compute degree matrix
        deg = degree(Data.edge_index[1], dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[Data.edge_index[1]].view(-1, 1) * deg_inv_sqrt[Data.edge_index[0]].view(-1, 1)

        # Normalize edge weights
        edge_weight = norm

        for conv in self.ConvBlocks:
            Data.x = conv(Data.x, Data.edge_index, edge_weight=edge_weight)
            Data.x = F.relu(Data.x)


        ##Create Offset to match correct index to correct row
        deal = [int(thing) for thing in list(Data.batch)]
        dicter = {}
        for i in range(3):
          for indx,j in enumerate(deal):
            if j in dicter.keys():
              continue
            else:
              dicter[j] = indx


        #lookup entity embeddings from graph_enhanced representations in sequence of each sentence
        """entity_embeddings = []
        for sentence in Data.entity_sentences:
          sent = []
          for entity in sentence:
            sent.append(Data.x[entity])
          entity_embeddings.append(sent)
        """
        entity_embeddings = []
        for Doc_Indx,doc in enumerate(Data.entity_sentences):
          sent_embeddings = []
          for sentence in doc:
            sent = []
            for entity in sentence:
              sent.append(Data.x[dicter[Doc_Indx]+entity-1])
            sent_embeddings.append(sent)

          entity_embeddings.append(sent_embeddings)



        # Average the embeddings using a linear layer
        linear_outputs = [[[self.sentence_linear(emb) for emb in sentence] for sentence in doc] for doc in entity_embeddings]
        relu_outputs = [[[F.relu(output) for output in sentence] for sentence in doc] for doc in linear_outputs]


        # Calculate sum for each sublist of indices
        sums = []
        for indx,relu in enumerate(relu_outputs):
          sumz = []
          for indx2,relu2 in enumerate(relu):

            try:
              sumz.append(torch.sum(torch.stack(relu2),dim = 0).to(device))
            except:
              sumz.append(torch.zeros(self.input_dim).to(device))
          sums.append(sumz)
        #replace zeros with something else?



        #Add small epsilon
        eps = 0.000001
        result = []
        for indx,dealydo in enumerate(sums):
          result.append(
              torch.div(torch.stack(sums[indx]),eps+torch.tensor(Data.lengths[indx], dtype=torch.float).unsqueeze(1).to(device))
          )

        #Pad Sequences for LSTM
        result = pad_sequences(result,self.input_dim)

        ##Pass through to an LSTM
        output, (h_n, c_n) = self.lstm(result)
        #print(output.shape)
        #print(h_n.shape)
        #print(c_n.shape)

        ##NSP Scores:
        NSP_Scores = pad_sequences_NSP(Data.nsp_scores)
        #print(NSP_Scores.shape)
        #print(output.shape)
        # Initialize sliding result
        solution_vector = torch.zeros((output.shape[0],output.shape[2])).to(device)
        for i in range(solution_vector.shape[0]):
            for j in range(NSP_Scores.shape[1]):

                solution_vector[i] += NSP_Scores[i][j] * (output[i][j] + output[i][j+1])

        #print(solution_vector.shape)
        ##Final Layer

        linear_dim_reduct_out = self.linear_dim_reduct(Data.CLS_tokens.squeeze(0))
        print(linear_dim_reduct_out.shape)
        print(solution_vector.squeeze(0).shape)
        Combination = torch.cat((solution_vector.squeeze(0),linear_dim_reduct_out),dim = 0)
        #print(Combination.shape)
        output = self.linear(Combination)
        #print(output)


        return F.sigmoid(F.relu(output))

##########################Jack##############################

#!wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2
#!bzip2 -dk enwiki_20180420_100d.pkl.bz2


def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc

#this function will create a dependency graph of our text.
#Our node values will be the word embeddings of each word in the sentence.
#the edges will be a directed graph based on the dependency structure of the data.
#can eventually experiment with edge values based on the kind of dependency???
def create_dependency_graph(document):

  #notes: get position with child.idx
  #part of speech with child.pos_
  #relation with child.dep_
  document = re.sub('\n', '',document)
  doc = nlp(document)
  sentences = list(doc.sents)

  #initialize our graph. We represent our edges as a dict of lists (for now)
  #this ensures that if we encounter a node for the second time we can quickly
  #check if we have already found its children and not continue the recursion
  #nodes are a list of lists. Each node will be the word embedding of that word
  #replace setting word embeddings with wikipedia to vec once that is set up.
  #when the graph is complete, each document will be a disjoint graph where each

  i = 0
  #our edge dictionary. This will be converted to the necessary list format down the line and determine connections
  edges = {}
  #Our nodes list. This will eventually hold the embeddings of each word in the document
  nodes = [[-1]] * 500
  #The sentence number. We define the position of our words by two coordinates: sentence number, and word number.
  sentence_num = 0
  #positions matrix. Formatted as [[sentence_number, index]]
  #edge features. This will hold information regarding the type of edge that exists between words (see the dict above)
  edge_features = []
  dropping = False
  for sentence in sentences:
    #edge-case: we are going to skip any sentence with punctuation as its root.
    #This may introduce issues, but such sentences are likely outliers anyways
    if sentence.root.is_punct:
      continue

    #first, we construct a dict tying a words index in the sentence to what word it is in the document
    #this will allow us to construct our graph representation.
    #while we are at it, we can create our positions list.
    words = {}
    for word in sentence:
      #once we get to 500, we do not want to continue processing. We will drop the last sentence
      if i == 500:
        dropping = True
        break
      string = str(word)
      if word.is_punct or string == "'s" or string == ',' or string == ' ':
        continue
      else:
        words[word.idx] = i
        i += 1
    #we break the loop if the document gets too long, taking only what we have so far.
    if dropping:
      break
    #make our node list and positions matrix
    for word in sentence:
      string = str(word)
      if word.is_punct or string == "'s" or string == ',' or string == ' ':
        continue
      try:
        nodes[words[word.idx]] = list(wiki2vec_jack.get_word_vector(str(word).lower()))
      except KeyError:
        nodes[words[word.idx]] = [0]*100

    #make our edge list and edge features matrix
    for word in sentence:
      string = str(word)
      if word.is_punct or string == "'s" or string == ',' or string == ' ':
        continue
      word_idx = words[word.idx]
      edges[word_idx] = []
      for child in word.children:
        child_string = str(child)
        if child.is_punct or child_string == "'s" or child_string == ',' or child_string == ' ':
          continue
        child_idx = words[child.idx]
        edges[word_idx].append([word_idx, child_idx])
        edge_features.append(deps[child.dep_])
    sentence_num += 1
  return (nodes, edges, edge_features)

#We use a dict to quickly check for membership while creating our graph. However,
#We need our data to be in a 2D list format. This function handles that
def edge_dict_to_list(edge_dict):
  sources = []
  destinations = []
  for i in range(len(edge_dict.keys())):
    if edge_dict[i]:
      for edge in edge_dict[i]:
        sources.append(edge[0])
        destinations.append(edge[1])
  sources = sources
  destinations = destinations
  edges = [sources, destinations]
  return edges

#wrote a function to create a graph dataset since we wont be able to store all of it in memory at once
#create a subset, store it, and return.
#hard-coded to work with GPT at the moment. Can work with others in the future

def make_jack_graph_data(document):
  nodes, edges, features = create_dependency_graph(document)
  nodes_index = nodes.index([-1])
  nodes = torch.tensor(nodes[:nodes_index], dtype = torch.float)
  edges = torch.tensor(edge_dict_to_list(edges), dtype = torch.long)
  features = torch.tensor(edges, dtype = torch.long)
  GraphData = pyg.data.Data(x=nodes,edge_index =edges,edge_attr=features)
  return GraphData




class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
       
        self.conv1 = GCNConv(100, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

###############Abhinav##################

def preprocess_function(examples):
    return Abhinavtokenizer(examples, padding='max_length',max_length = 400, truncation=True,return_tensors="pt")








import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',type = str)
  parser.add_argument('destination',type = str)
  args = parser.parse_args()

  input_file = args.input_file
  output_dir = args.output_dir

  returned = return_json(input_file)

  Ben_Model_Path = '/app/550_Iterations_first_5000_wikipedia_False_Ben_GCN_model.pth'
  Jack_Model_Path = '/app/GCN2.pt'
  Abhinav_Model_Path = '/app/AbhinavModel'
  Meta_Classifier_Path = '/app/xgboost_model.pkl'

  original_keys = {}
  mixmatch = {}

  for i, (key, value) in enumerate(returned.items()):
      original_keys[i] = key
      mixmatch[i] = value


    
  wiki2vec_jack = Wikipedia2Vec.load(extracted_file_path)
  #nlp_jack = spacy.load('en_core_web_lg')

  #get our dataset. Currently only works with GPT wiki

  deps = {'root': 0, 'nummod': 1, 'advcl':2, 'oprd':3, 'intj':4,
          'auxpass':5, 'dep':6, 'parataxis':7, 'nmod':8,
          'aux':9, 'conj':10, 'amod':11, 'cc':12, 'nsubjpass':13,
          'csubj':14, 'neg':15, 'relcl':16, 'attr':17, 'npadvmod':18,
          'meta':19, 'preconj':20, 'advmod':21, 'csubjpass':22,
          'prt':23, 'compound':24, 'case':25, 'xcomp':26, 'nsubj':27,
          'det':28, 'acomp':29, 'dative':30, 'expl':31, 'pcomp':32,
          'dobj':33, 'predet':34, 'quantmod':35, 'pobj':36, 'acl':37,
          'ccomp':38, 'agent':39, 'poss':40, 'mark':41, 'punct':42,
          'appos':43, 'prep':44}



  # Load the configuration
  

  # Load the model with the configuration and weights
  Abhinavmodel = RobertaForSequenceClassification.from_pretrained(Abhinav_Model_Path)
  Abhinavtokenizer = RobertaTokenizer.from_pretrained(Abhinav_Model_Path)
  Abhinavmodel.batch_size = 1



  # Set the model to evaluation mode
  Abhinavmodel.eval()

  #---------------------Combined------------------------
  input_dim = 1024
  hidden_dim = 256
  output_dim = 1
  numv_convs = 5
  wiki2vec = False
  BenModel = MyGCN(input_dim, hidden_dim, output_dim,numv_convs,wiki2vec).to(device)
  BenModel.load_state_dict(torch.load(Ben_Model_Path))
  BenModel.eval()

  texts = {}
  CLS_tokens = {}
  document_list = mixmatch

  texts,CLS_tokens = process_document(texts,CLS_tokens,document_list)
  Initial_embeddings,entity_sentences,lengths,nsp_scores,adj_matrixs = Dataset_Curation(texts,similarity=0.9,wiki2vec = wiki2vec)

  JackModel = GCN(64).to(device)
  JackModel.load_state_dict(torch.load(Jack_Model_Path))
  JackModel.eval()

  DataFrame = pd.DataFrame(data = None,columns = ['Index','BenText1', 'BenText2','JackText1','JackText2','AbhinavText1','AbhinavText2'])


  keys = ['text1:','text2:']
  for i,key in enumerate(list(mixmatch.keys())):
    indy = i
    dater = CleanedGraphData(Initial_embeddings[indy],entity_sentences[indy],lengths[indy],nsp_scores[indy],adj_matrixs[indy],CLS_tokens[indy] )
    dater._indices = range(1)
    dater.transform = None # Add the transform attribute
    batch_size = 1
    train_data_loader = DataLoader(dater, batch_size=batch_size, shuffle=False)
    for batch in train_data_loader:
      try:
        BenText1 = BenModel(batch[0])
        BenText1 = float(BenText1.detach().cpu().numpy())
      except:
        BenText1 = -9
      
      try:
        BenText2 = BenModel(batch[1])
      
        BenText2 = float(BenText2.detach().cpu().numpy())
      except:
        BenText2 = -9

    


    ##Jack
    x_Jack_1 = make_jack_graph_data(mixmatch[i]['text1:'])
    x_Jack_2 = make_jack_graph_data(mixmatch[i]['text2:'])

    out_Jack_1 = JackModel(x = x_Jack_1.x.to(device), edge_index = x_Jack_1.edge_index.to(device),batch = torch.tensor([0]).to(device)).detach().cpu().numpy()[0][0]
    #out_Jack_1 = out_Jack_1.argmax(dim=1)

    out_Jack_2 = JackModel(x = x_Jack_2.x.to(device), edge_index = x_Jack_2.edge_index.to(device),batch = torch.tensor([0]).to(device)).detach().cpu().numpy()[0][0]
    #out_Jack_2 = out_Jack_2.argmax(dim=1)

    ##Abhinav
    inputs_1 = preprocess_function(mixmatch[i]['text1:'])
    inputs_2 = preprocess_function(mixmatch[i]['text2:'])
    Abhinavmodel.batch_size = 1

    # Set the model to evaluation mode
    Abhinavmodel.eval()
    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs_1 = Abhinavmodel(input_ids = inputs_1.input_ids, attention_mask=inputs_1.attention_mask)
        outputs_2 = Abhinavmodel(input_ids = inputs_2.input_ids, attention_mask=inputs_2.attention_mask)

    outputs_1 = float(torch.nn.functional.softmax(outputs_1[0][0])[0])
    outputs_2 = float(torch.nn.functional.softmax(outputs_2[0][0])[0])
    
    DataFrame.loc[len(DataFrame)] = [key,BenText1,BenText2,out_Jack_1,out_Jack_2,outputs_1,outputs_2]






# Save the trained model
with open(Meta_Classifier_Path, 'rb') as f:
  model = pickle.load(f)

predictions = model.predict(DataFrame[DataFrame.columns[1:]])

final_dict = {}
for i,key in enumerate(original_keys):
  final_dict[original_keys[key]] = predictions[i]

def dict_to_jsonl(input_dict, file_path):
  with open(file_path, 'w') as f:
      for key, value in input_dict.items():
          json_obj = json.dumps({"id": key, "is_human": float(value)})
          f.write(json_obj + '\n')

dict_to_jsonl(final_dict, output_dir+'/predictions.jsonl')