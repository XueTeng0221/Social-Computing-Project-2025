import torch
import hashlib
import re
import pandas as pd
import json
import numpy as np
from typing import Dict, List
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Loading tokenizer and model from {model_name} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        with open('risk_keywords.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.risk_keywords = data['risk_keywords']
            self.urgent_keywords = data['urgent_keywords']
            self.risk_domains = data['risk_domains']

    def clean_text(self, text) -> str:
        """
        清洗文本，处理非字符串输入
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # 仅保留中文、字母和数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        return text

    def extract_urls(self, text: str) -> List[str]:
        """提取URL并标准化域名"""
        if not isinstance(text, str):
            return []
            
        urls = re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            text
        )
        domains = []
        for url in urls:
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/:]+)', url)
            if domain_match:
                domains.append(domain_match.group(1))
        return list(set(domains))  # 去重

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取潜在的高风险实体：手机号, 群号, 加密货币地址"""
        if not isinstance(text, str):
            return {'phones': [], 'groups': [], 'crypto': []}

        phones = re.findall(r'1[3-9]\d{9}', text)
        group_ids = re.findall(r'(?:群号?|QQ|微信)[：:\s]*(\d{6,10})', text)
        crypto_addrs = re.findall(r'\b(?:[13][a-km-zA-HJ-NP-Z1-9]{25,34}|0x[a-fA-F0-9]{40}|T[A-Za-z1-9]{33})\b', text)

        return {
            'phones': list(set(phones)),
            'groups': list(set(group_ids)),
            'crypto': list(set(crypto_addrs))
        }

    def extract_media_hash(self, media_url: str) -> str:
        """为图片/视频生成哈希标识（实际项目中应使用感知哈希）"""
        if not isinstance(media_url, str):
            return "dummy_hash"
        return hashlib.md5(media_url.encode()).hexdigest()[:16]

    def get_text_embeddings_batch(self, texts: List[str], batch_size=32, max_len=64) -> torch.Tensor:
        """
        分批次获取文本的 [CLS] 向量
        返回: (num_samples, hidden_size) 的 Tensor
        """
        all_embeddings = []
        
        # 简单的空文本处理：如果全是空，返回零向量
        if not texts:
            return torch.empty((0, 768))

        # 使用 tqdm 显示进度
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            # 确保batch内没有非字符串
            batch_texts = [t if isinstance(t, str) and len(t) > 0 else "空" for t in batch_texts]
            
            encoded = self.tokenizer(
                batch_texts, 
                padding='max_length', 
                truncation=True, 
                max_length=max_len, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)

    def compute_text_similarity_edges(self, embeddings: torch.Tensor, threshold: float = 0.85, chunk_size: int = 1000):
        """
        计算文本相似度，用于构建 P-P 边
        注意：O(N^2) 复杂度，对于大量数据(N > 10000)需要优化
        """
        num_nodes = embeddings.shape[0]
        if num_nodes == 0:
            return torch.empty((2, 0), dtype=torch.long)
            
        # 归一化向量以便直接计算余弦相似度 (A . B) / (|A|*|B|) -> norm_A . norm_B
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        edge_index = []
        
        # 如果数据量较小，可以使用矩阵乘法一次性计算
        if num_nodes < 5000:
            sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
            mask = sim_matrix[triu_indices[0], triu_indices[1]] >= threshold
            sources = triu_indices[0][mask]
            targets = triu_indices[1][mask]
            edge_index = torch.stack([
                torch.cat([sources, targets]),
                torch.cat([targets, sources])
            ], dim=0)
            
        else:
            # 数据量大时，为了节省内存，可以使用分块计算或Faiss库（此处简化为分块循环）
            print(f"Warning: Large number of posts ({num_nodes}), computing similarity might be slow.")
            sources_list = []
            targets_list = []
            chunk_size = 1000
            for i in range(0, num_nodes, chunk_size):
                end_i = min(i + chunk_size, num_nodes)
                chunk_i = embeddings_norm[i:end_i]
                sim_chunk = torch.mm(chunk_i, embeddings_norm.t())
                rows, cols = torch.where(sim_chunk >= threshold)
                global_rows = rows + i
                valid_mask = global_rows < cols
                sources_list.append(global_rows[valid_mask])
                targets_list.append(cols[valid_mask])
            
            if sources_list:
                all_sources = torch.cat(sources_list)
                all_targets = torch.cat(targets_list)
                edge_index = torch.stack([
                    torch.cat([all_sources, all_targets]),
                    torch.cat([all_targets, all_sources])
                ], dim=0)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index

    def build_graph(self, df_posts: pd.DataFrame, df_users: pd.DataFrame, df_relations: pd.DataFrame,
                    df_media: pd.DataFrame = None) -> HeteroData:
        data = HeteroData()

        # ========== 1. 构建节点特征 ==========

        # ----- Post 节点 -----
        print("Processing Post texts...")
        post_texts_clean = df_posts['content'].apply(self.clean_text).tolist()
        post_embeddings = self.get_text_embeddings_batch(post_texts_clean, batch_size=64)
        data['post'].x = post_embeddings  # Shape: [num_posts, 768]
        data['post'].y = torch.tensor(df_posts['label'].values, dtype=torch.long)
        post_meta_features = []
        for text in df_posts['content']:
            if pd.isna(text) or not isinstance(text, str):
                text = ""
            keyword_count = sum([1 for kw in self.risk_keywords if kw in text])
            entities = self.extract_entities(text)
            entity_count = len(entities['phones']) + len(entities['groups']) + len(entities['crypto'])
            post_meta_features.append([
                float(keyword_count),
                float(entity_count),
                float(len(text)),
                1.0 if any(kw in text for kw in self.urgent_keywords) else 0.0
            ])
        data['post'].meta = torch.tensor(post_meta_features, dtype=torch.float)

        # ----- User 节点 -----
        print("Processing Users...")
        user_features = []
        user_labels = []
        df_users['user_name'] = df_users['user_name'].replace(r'^\s*$', np.nan, regex=True)
        subset_cols = ['reg_time', 'post_count', 'follower_count', 'following_count']
        df_sorted = df_users.sort_values(by=['user_name'], na_position='last')
        df_users_clean = df_sorted.drop_duplicates(subset=subset_cols, keep='first')

        for idx, row in df_users_clean.iterrows():
            reg_time = float(row.get('reg_time', 0.0)) if pd.notna(row.get('reg_time')) else 0.0
            post_freq = row.get('post_count', 0) / max(reg_time, 1)
            follower_ratio = row.get('follower_count', 0) / max(row.get('following_count', 1), 1)
            user_features.append([
                reg_time / 365.0,
                post_freq,
                follower_ratio,
                float(row.get('post_count', 0)),
                float(row.get('verified', 0)),
                float(row.get('has_avatar', 1)),
            ])

            user_posts = df_posts[df_posts['user_id'] == row['user_id']]
            is_risky = 1 if user_posts['label'].sum() > 0 else 0
            user_labels.append(is_risky)

        data['user'].x = torch.tensor(user_features, dtype=torch.float)
        data['user'].y = torch.tensor(user_labels, dtype=torch.long)

        # ----- Domain 节点 -----
        domain_to_id = {}
        domain_features = []
        domain_labels = []

        for text in df_posts['content']:
            if pd.isna(text) or not isinstance(text, str):
                continue
            domains = self.extract_urls(text)
            for domain in domains:
                if domain not in domain_to_id:
                    domain_to_id[domain] = len(domain_to_id)
                    is_blacklisted = 1 if domain in self.risk_domains else 0
                    domain_features.append([
                        float(is_blacklisted),
                        float(len(domain)),
                        1.0 if any(char.isdigit() for char in domain) else 0.0,
                        1.0 if domain.endswith('.tk') or domain.endswith('.ml') else 0.0
                    ])
                    domain_labels.append(is_blacklisted)

        if domain_features:
            data['domain'].x = torch.tensor(domain_features, dtype=torch.float)
            data['domain'].y = torch.tensor(domain_labels, dtype=torch.long)
        else:
            data['domain'].x = torch.zeros((1, 4), dtype=torch.float)
            data['domain'].y = torch.zeros(1, dtype=torch.long)

        # ----- Entity 节点 -----
        entity_to_id = {}
        entity_types = []

        for text in df_posts['content']:
            if pd.isna(text) or not isinstance(text, str):
                continue
            entities = self.extract_entities(text)
            all_raw_entities = [(e, 0) for e in entities['phones']] + \
                               [(e, 1) for e in entities['groups']] + \
                               [(e, 2) for e in entities['crypto']]
            
            for ent_str, ent_type in all_raw_entities:
                if ent_str not in entity_to_id:
                    entity_to_id[ent_str] = len(entity_to_id)
                    entity_types.append(ent_type)

        if entity_types:
            data['entity'].x = torch.nn.functional.one_hot(
                torch.tensor(entity_types, dtype=torch.long), num_classes=3
            ).float()
        else:
            data['entity'].x = torch.zeros((1, 3), dtype=torch.float)

        # ----- Media 节点 -----
        media_to_id = {}
        if df_media is not None and len(df_media) > 0:
            for idx, row in df_media.iterrows():
                media_hash = self.extract_media_hash(row['media_url'])
                if media_hash not in media_to_id:
                    media_to_id[media_hash] = len(media_to_id)

            num_media = len(media_to_id)
            data['media'].x = torch.eye(num_media) if num_media < 1000 else torch.randn(num_media, 64)
        else:
            data['media'].x = torch.zeros((1, 16), dtype=torch.float)
            media_to_id['dummy'] = 0

        # ========== 2. 构建边 ==========
        print("Building edges...")

        user_id_to_idx = {uid: idx for idx, uid in enumerate(df_users_clean['user_id'])}
        post_id_to_idx = {pid: idx for idx, pid in enumerate(df_posts['post_id'])}

        # (U -> P)
        publish_edges = []
        for idx, row in df_posts.iterrows():
            user_idx = user_id_to_idx.get(row['user_id'])
            if user_idx is not None:
                publish_edges.append([user_idx, idx])
        if publish_edges:
            data['user', 'publish', 'post'].edge_index = torch.tensor(publish_edges, dtype=torch.long).t()

        # (U -> P) Repost
        repost_edges = []
        for idx, row in df_posts.iterrows():
            if row.get('is_repost', False) and pd.notna(row.get('parent_post_id')):
                user_idx = user_id_to_idx.get(row['user_id'])
                parent_idx = post_id_to_idx.get(row['parent_post_id'])
                if user_idx is not None and parent_idx is not None:
                    repost_edges.append([user_idx, parent_idx])
        if repost_edges:
            data['user', 'repost', 'post'].edge_index = torch.tensor(repost_edges, dtype=torch.long).t()

        # (P -> D)
        cite_domain_edges = []
        for post_idx, text in enumerate(df_posts['content']):
            if pd.isna(text) or not isinstance(text, str):
                continue
            domains = self.extract_urls(text)
            for domain in domains:
                domain_idx = domain_to_id.get(domain)
                if domain_idx is not None:
                    cite_domain_edges.append([post_idx, domain_idx])
        if cite_domain_edges:
            data['post', 'cite', 'domain'].edge_index = torch.tensor(cite_domain_edges, dtype=torch.long).t()

        # (P -> E)
        contain_entity_edges = []
        for post_idx, text in enumerate(df_posts['content']):
            if pd.isna(text) or not isinstance(text, str):
                continue
            entities = self.extract_entities(text)
            all_ents = entities['phones'] + entities['groups'] + entities['crypto']
            for ent in all_ents:
                ent_idx = entity_to_id.get(ent)
                if ent_idx is not None:
                    contain_entity_edges.append([post_idx, ent_idx])
        if contain_entity_edges:
            data['post', 'contain', 'entity'].edge_index = torch.tensor(contain_entity_edges, dtype=torch.long).t()

        # (P -> M)
        if 'media_urls' in df_posts.columns:
            post_media_edges = []
            for post_idx, media_urls in enumerate(df_posts['media_urls']):
                if pd.isna(media_urls):
                    continue
                urls = str(media_urls).split(',')
                for url in urls:
                    media_hash = self.extract_media_hash(url.strip())
                    media_idx = media_to_id.get(media_hash)
                    if media_idx is not None:
                        post_media_edges.append([post_idx, media_idx])
            if post_media_edges:
                data['post', 'has_media', 'media'].edge_index = torch.tensor(post_media_edges, dtype=torch.long).t()

        # (P <-> P) 相似文本边 
        print("Computing P-P similarity edges...")
        similarity_edge_index = self.compute_text_similarity_edges(data['post'].x, threshold=0.85)
        if similarity_edge_index.size(1) > 0:
            data['post', 'similar', 'post'].edge_index = similarity_edge_index

        # (U <-> U)
        follow_edges = []
        interact_edges = []
        for idx, row in df_relations.iterrows():
            src_idx = user_id_to_idx.get(row['source_user_id'])
            tgt_idx = user_id_to_idx.get(row['target_user_id'])
            if src_idx is not None and tgt_idx is not None:
                if row['relation_type'] == 'follow':
                    follow_edges.append([src_idx, tgt_idx])
                elif row['relation_type'] == 'interact':
                    interact_edges.append([src_idx, tgt_idx])
        
        if follow_edges:
            data['user', 'follow', 'user'].edge_index = torch.tensor(follow_edges, dtype=torch.long).t()
        if interact_edges:
            data['user', 'interact', 'user'].edge_index = torch.tensor(interact_edges, dtype=torch.long).t()

        return data
