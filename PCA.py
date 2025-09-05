
import os
import torch
import math
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.decomposition import PCA

from Img_extractor import img_extractor

def encode_sent_emb(config, dataset, output_path):

    meta_sentences = []
    for i in range(1, dataset.n_items):
        meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

    if 'sentence-t5-base' in config['sent_emb_model']:
        sent_emb_model = SentenceTransformer(
            config['sent_emb_model']
        ).to(config['device'])

        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=config['device']
        )
    elif 'text-embedding-3' in config['sent_emb_model']:
        client = OpenAI(api_key = os.getenv("API_KEY"))
        sent_embs = []
        for i in tqdm(range(0, len(meta_sentences), config['sent_emb_batch_size']), desc='Encoding'):
            try:
                responses = client.embeddings.create(
                    input=meta_sentences[i: i + config['sent_emb_batch_size']],
                    model=config['sent_emb_model']
                )
            except:
                batch = meta_sentences[i: i + config['sent_emb_batch_size']]
                new_batch = []
                for sent in batch:
                    n_tokens = config['n_tokens']
                    if n_tokens < 8192:
                        new_batch.append(sent)
                    else:
                        n_chars = 8192 / n_tokens * len(sent) - 100
                        new_batch.append(sent[:int(n_chars)])
                responses = client.embeddings.create(
                    input=new_batch,
                    model=config['sent_emb_model']
                )

            for response in responses.data:
                sent_embs.append(response.embedding)
        sent_embs = np.array(sent_embs, dtype=np.float32)
    sent_embs.tofile(output_path) # XXXXX.sent_emb
    return sent_embs


# Encodes the image embeddings for the given dataset and saves them to the specified output path.
def encode_img_emb(config, model_path, dataset, img_path, output_path, img_dim):
    item_name_list = [] 
    for i in range(1, dataset.n_items):
        item_name_list.append(dataset.id_mapping['id2item'][i])
    
    img_embs = img_extractor(config['device'], img_path, item_name_list, img_dim, model_path)
    img_embs = np.array(img_embs, dtype=np.float32)
    img_embs.tofile(output_path) # XXXXX.img_emb

    return img_embs

def get_items_for_training(dataset):
    items_for_training = set()
    for item_seq in dataset.split_data['train']['item_seq']:
      for item in item_seq:
        items_for_training.add(item)
    mask = np.zeros(dataset.n_items - 1, dtype=bool)
    for item in items_for_training:
      mask[dataset.item2id[item] - 1] = True
    return mask


def PCA_emb(config, dataset_path, dataset):

    PCA_emb_path = os.path.join(
        dataset_path, 'processed/item_txt_img.PCA_emb'
    )

    if os.path.exists(PCA_emb_path):
        os.remove(PCA_emb_path) 

    sent_emb_path = os.path.join(
        dataset_path,
        f'processed/{os.path.basename(config["sent_emb_model"])}.sent_emb'
    )
    img_emb_path = os.path.join(
        dataset_path,
        f'processed/{os.path.basename(config["sent_emb_model"])}.img_emb'
    )
    if os.path.exists(sent_emb_path):
        sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, config['sent_emb_dim'])
    else:
        sent_embs = encode_sent_emb(config, dataset, sent_emb_path)

    if os.path.exists(img_emb_path):
        img_embs = np.fromfile(img_emb_path, dtype=np.float32).reshape(-1, config['img_emb_dim'])
    else:
        image_model_path = config['image_model_path']
        img_path = config['img_path']
        img_embs = encode_img_emb(config, image_model_path, dataset, img_path, img_emb_path, img_dim=config['img_emb_dim'])

    # PCA sent_emb
    pca = PCA(n_components=512, whiten=True)
    sent_embs = pca.fit_transform(sent_embs)

    # PCA img_emb
    pca = PCA(n_components=256, whiten=True) 
    img_embs = pca.fit_transform(img_embs)
    pca = PCA(n_components=64, whiten=True)

    sent_img_embs = np.concatenate((sent_embs, img_embs), axis=1)

    sent_img_embs.tofile(PCA_emb_path) # XXXXX.sent_emb

    return sent_img_embs
