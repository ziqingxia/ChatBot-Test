import os
import torch
from openai import OpenAI
import torch.nn.functional as F

class RAGKnowledgeBase():
    def __init__(self, config, root_path, database_names=None):
        if database_names is None:
            database_names = os.listdir(root_path)
        self.database = {}
        self.datanames = []
        self.config = config
        self.client = OpenAI()
        for name in database_names:
            database_path = os.path.join(root_path, name, 'contents_with_embed.pth')
            self.add_knowledge(name, database_path)
            self.datanames.append(name)
            print(f"==> Load processed file into database: {name}")        

    def add_knowledge(self, name, database_path):
        self.database[name] = torch.load(database_path, weights_only=False)
        self.database[name]['embedding'] = self.database[name]['embedding'].cuda()

    def remove_knowledge(self, name):
        self.database.pop(name)

    def get_embeddings(self, text):
        embeddings = self.client.embeddings.create(
                    model=self.config['MODEL_TYPES']['TEXT_EMBED_MODEL'],
                    input=text,
                    encoding_format="float"
                    )
        return embeddings.data[0].embedding

    def get_topk(self, input_embed, topk=5, threshold=0.1):
        selected_scores = []
        selected_metas = []
        selected_contents = []
        # local search file one by one
        for name in self.datanames:
            similarity = F.cosine_similarity(input_embed, self.database[name]['embedding'], dim=-1)
            values, indices = torch.topk(similarity, k=topk, largest=True)
            selected_scores.append(values)
            for index in indices.tolist():
                selected_metas.append(str(self.database[name]['meta'][index]))
                selected_contents.append(str(self.database[name]['content'][index]))

        # global combine all results and get final topk
        final_scores = []
        final_metas = []
        final_contents = []
        global_similarity = torch.cat(selected_scores, dim=0)
        values, indices = torch.topk(global_similarity, k=topk, largest=True)
        for index, score in zip(indices.tolist(), values.tolist()):
            if score > threshold:
                final_scores.append(score)
                final_metas.append(selected_metas[index])
                final_contents.append(selected_contents[index])
        return final_scores, final_metas, final_contents

        
    def search_knowledge(self, input, prefix="RAG", topk=5):
        # get input embedding
        input_embed = self.get_embeddings(input)
        input_embed = torch.FloatTensor(input_embed).unsqueeze(0).cuda()
        # get search parameters
        threshold = self.config['SEARCH']['THRESHOLD']
        display_length = self.config['SEARCH']['DISPLAY_LENGTH']
        # search
        scores, metas, contents = self.get_topk(input_embed, topk=topk, threshold=threshold)
        # generate content prompt and reference
        if len(contents) > 0:
            content_prompt = "\n".join(contents)
            print_reference = f"{prefix} REFERENCE:\n" + "\n".join([f"[{i}] ({s:.3f} score) {m}: {c[:display_length]}" for i, (s, m, c) in enumerate(zip(scores, metas, contents))])
            print(f"====="*10)
            print(print_reference)
        else:
            content_prompt = None
            print_reference = f"{prefix} REFERENCE:\nNo related results found!"
            print(f"====="*10)
            print(print_reference)
        return content_prompt





    