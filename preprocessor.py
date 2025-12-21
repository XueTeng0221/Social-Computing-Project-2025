# preprocessor.py

import torch
import hashlib
import re
import pandas as pd
from collections import defaultdict
from typing import Dict, List
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


class DataPreprocessor:
    def __init__(self, model_name: str = 'hfl/chinese-roberta-wwm-ext'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.risk_keywords = ['高收益', '带单', '内幕', '回本', '加群', 'usdt', '兼职刷单']
        self.risk_domains = ['fake-invest.com', 'scam-platform.net']

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        return text

    def extract_urls(self, text: str) -> List[str]:
        """提取URL并标准化域名"""
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
        return hashlib.md5(media_url.encode()).hexdigest()[:16]

    def get_text_embedding(self, texts, max_len=64):
        """预处理文本用于模型输入 (Token IDs & Attention Masks)"""
        encoded = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def compute_text_similarity(self, embeddings: torch.Tensor, threshold: float = 0.75):
        """计算文本相似度，用于构建 P-P 边"""
        sim_matrix = cosine_similarity(embeddings)
        edge_index = []
        for i in range(len(sim_matrix)):
            for j in range(i + 1, len(sim_matrix)):
                if sim_matrix[i, j] >= threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])  # 无向边
        return torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.empty((2, 0), dtype=torch.long)

    def build_graph(self, df_posts: pd.DataFrame, df_users: pd.DataFrame, df_relations: pd.DataFrame,
                    df_media: pd.DataFrame = None) -> HeteroData:
        """
        完整构建异构图

        参数:
        - df_posts: [post_id, content, user_id, label, is_repost, parent_post_id, media_urls]
        - df_users: [user_id, reg_time, post_count, follower_count, ...]
        - df_relations: [source_user_id, target_user_id, relation_type]
                       relation_type: 'follow'关注, 'interact'互动
        - df_media: [media_id, media_url, media_type] (可选)

        返回:
        - HeteroData 对象，包含所有节点、边和任务标签
        """
        data = HeteroData()

        # ========== 1. 构建节点特征 ==========

        # ----- Post 节点 -----
        post_texts = df_posts['content'].apply(self.clean_text).tolist()
        input_ids, att_masks = self.get_text_embedding(post_texts)
        data['post'].x = input_ids
        data['post'].mask = att_masks

        # Post 标签（节点分类任务1：帖子是否诈骗）
        data['post'].y = torch.tensor(df_posts['label'].values, dtype=torch.long)

        # Post 元特征（可用于增强表征）
        # 包含敏感词数量、实体数量、文本长度等
        post_meta_features = []
        for text in df_posts['content']:
            if not isinstance(text, str):
                text = ""
            keyword_count = sum([1 for kw in self.risk_keywords if kw in text])
            entities = self.extract_entities(text)
            entity_count = len(entities['phones']) + len(entities['groups']) + len(entities['crypto'])
            post_meta_features.append([
                keyword_count,
                entity_count,
                len(text),
                1 if any(kw in text for kw in ['urgent', '紧急', '限时']) else 0
            ])
        data['post'].meta = torch.tensor(post_meta_features, dtype=torch.float)

        # ----- User 节点 -----
        num_users = len(df_users)
        user_features = []
        user_labels = []  # 用户风险标签（节点分类任务2）

        for idx, row in df_users.iterrows():
            # 提取用户行为特征
            account_age_days = (pd.Timestamp.now() - pd.to_datetime(row.get('reg_time', '2020-01-01'))).days
            post_freq = row.get('post_count', 0) / max(account_age_days, 1)  # 发帖频率
            follower_ratio = row.get('follower_count', 0) / max(row.get('following_count', 1), 1)

            user_features.append([
                account_age_days / 365.0,  # 归一化账户年龄
                post_freq,
                follower_ratio,
                row.get('post_count', 0),
                row.get('verified', 0),  # 是否认证
                row.get('has_avatar', 1),  # 是否有头像
            ])

            # 用户风险标签（通过规则或已有标注）
            # 示例：如果用户发布了诈骗帖子 -> 高风险
            user_posts = df_posts[df_posts['user_id'] == row['user_id']]
            is_risky = 1 if user_posts['label'].sum() > 0 else 0
            user_labels.append(is_risky)

        data['user'].x = torch.tensor(user_features, dtype=torch.float)
        data['user'].y = torch.tensor(user_labels, dtype=torch.long)  # 任务2标签

        # ----- Domain 节点 -----
        domain_to_id = {}
        domain_features = []
        domain_labels = []  # 域名风险标签（节点分类任务3）

        for text in df_posts['content']:
            if not isinstance(text, str):
                continue
            domains = self.extract_urls(text)
            for domain in domains:
                if domain not in domain_to_id:
                    domain_to_id[domain] = len(domain_to_id)
                    # 域名特征：是否在黑名单、域名长度、是否包含数字等
                    is_blacklisted = 1 if domain in self.risk_domains else 0
                    domain_features.append([
                        is_blacklisted,
                        len(domain),
                        1 if any(char.isdigit() for char in domain) else 0,
                        1 if domain.endswith('.tk') or domain.endswith('.ml') else 0  # 免费域名
                    ])
                    domain_labels.append(is_blacklisted)  # 简单规则标注

        if domain_features:
            data['domain'].x = torch.tensor(domain_features, dtype=torch.float)
            data['domain'].y = torch.tensor(domain_labels, dtype=torch.long)
        else:
            data['domain'].x = torch.zeros((1, 4), dtype=torch.float)
            data['domain'].y = torch.zeros(1, dtype=torch.long)

        # ----- Entity 节点（群号/电话/加密地址） -----
        entity_to_id = {}
        entity_types = []  # 0: phone, 1: group, 2: crypto

        for text in df_posts['content']:
            if not isinstance(text, str):
                continue
            entities = self.extract_entities(text)
            for phone in entities['phones']:
                if phone not in entity_to_id:
                    entity_to_id[phone] = len(entity_to_id)
                    entity_types.append(0)
            for group in entities['groups']:
                if group not in entity_to_id:
                    entity_to_id[group] = len(entity_to_id)
                    entity_types.append(1)
            for crypto in entities['crypto']:
                if crypto not in entity_to_id:
                    entity_to_id[crypto] = len(entity_to_id)
                    entity_types.append(2)

        num_entities = len(entity_to_id)
        if num_entities > 0:
            # Entity 特征：One-hot 类型编码 + learnable embedding
            entity_type_onehot = torch.nn.functional.one_hot(
                torch.tensor(entity_types, dtype=torch.long), num_classes=3
            ).float()
            data['entity'].x = entity_type_onehot
        else:
            data['entity'].x = torch.zeros((1, 3), dtype=torch.float)

        # ----- Media 节点（图片/视频） -----
        media_to_id = {}
        if df_media is not None and len(df_media) > 0:
            for idx, row in df_media.iterrows():
                media_hash = self.extract_media_hash(row['media_url'])
                if media_hash not in media_to_id:
                    media_to_id[media_hash] = len(media_to_id)

            num_media = len(media_to_id)
            # Media 特征：使用 learnable embedding（实际应用中可用预训练的图像特征）
            data['media'].x = torch.eye(num_media) if num_media < 1000 else torch.randn(num_media, 128)
        else:
            data['media'].x = torch.zeros((1, 16), dtype=torch.float)
            media_to_id['dummy'] = 0

        # ========== 2. 构建边（Edge Index） ==========

        # 创建 user_id 到索引的映射
        user_id_to_idx = {uid: idx for idx, uid in enumerate(df_users['user_id'])}
        post_id_to_idx = {pid: idx for idx, pid in enumerate(df_posts['post_id'])}

        # ----- (U -> P) 发布边 -----
        publish_edges = []
        for idx, row in df_posts.iterrows():
            user_idx = user_id_to_idx.get(row['user_id'])
            if user_idx is not None:
                publish_edges.append([user_idx, idx])

        if publish_edges:
            data['user', 'publish', 'post'].edge_index = torch.tensor(publish_edges, dtype=torch.long).t()

        # ----- (U -> P) 转发/评论边 -----
        repost_edges = []
        for idx, row in df_posts.iterrows():
            if row.get('is_repost', False) and pd.notna(row.get('parent_post_id')):
                user_idx = user_id_to_idx.get(row['user_id'])
                parent_idx = post_id_to_idx.get(row['parent_post_id'])
                if user_idx is not None and parent_idx is not None:
                    repost_edges.append([user_idx, parent_idx])

        if repost_edges:
            data['user', 'repost', 'post'].edge_index = torch.tensor(repost_edges, dtype=torch.long).t()

        # ----- (P -> D) 引用域名边 -----
        cite_domain_edges = []
        for post_idx, text in enumerate(df_posts['content']):
            if not isinstance(text, str):
                continue
            domains = self.extract_urls(text)
            for domain in domains:
                domain_idx = domain_to_id.get(domain)
                if domain_idx is not None:
                    cite_domain_edges.append([post_idx, domain_idx])

        if cite_domain_edges:
            data['post', 'cite', 'domain'].edge_index = torch.tensor(cite_domain_edges, dtype=torch.long).t()

        # ----- (P -> E) 包含实体边 -----
        contain_entity_edges = []
        for post_idx, text in enumerate(df_posts['content']):
            if not isinstance(text, str):
                continue
            entities = self.extract_entities(text)
            all_entities = entities['phones'] + entities['groups'] + entities['crypto']
            for entity in all_entities:
                entity_idx = entity_to_id.get(entity)
                if entity_idx is not None:
                    contain_entity_edges.append([post_idx, entity_idx])

        if contain_entity_edges:
            data['post', 'contain', 'entity'].edge_index = torch.tensor(contain_entity_edges, dtype=torch.long).t()

        # ----- (P -> M) 共同媒体边 -----
        if 'media_urls' in df_posts.columns:
            post_media_edges = []
            for post_idx, media_urls in enumerate(df_posts['media_urls']):
                if pd.isna(media_urls):
                    continue
                # 假设 media_urls 是以逗号分隔的字符串
                urls = str(media_urls).split(',')
                for url in urls:
                    media_hash = self.extract_media_hash(url.strip())
                    media_idx = media_to_id.get(media_hash)
                    if media_idx is not None:
                        post_media_edges.append([post_idx, media_idx])

            if post_media_edges:
                data['post', 'has_media', 'media'].edge_index = torch.tensor(post_media_edges, dtype=torch.long).t()

        # ----- (P <-> P) 相似文本边 -----
        # 需要先获取文本向量（这里使用简化的 TF-IDF 或预计算的 embeddings）
        # 在实际项目中，应该使用 RoBERTa 的 [CLS] token 向量
        # 这里用随机向量模拟（实际应该在外部预计算）
        # post_embeddings = torch.randn(len(df_posts), 768)  # 模拟
        # similar_edges = self.compute_text_similarity(post_embeddings.numpy(), threshold=0.75)
        # data['post', 'similar', 'post'].edge_index = similar_edges

        # ----- (U <-> U) 关注/互动边 -----
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

        # ========== 3. 边/子图异常检测任务准备 ==========

        # 标记"协同团伙"边（示例：多个用户短时间内互相转发同一批诈骗帖）
        # 这需要时序信息，这里提供框架
        data['edge_labels'] = {}  # 存储边级别的标签

        # 示例：检测异常密集的 User-User 互动
        if interact_edges:
            # 统计每条边的权重（互动频次）
            edge_weights = defaultdict(int)
            for edge in interact_edges:
                edge_weights[tuple(edge)] += 1

            # 标记高频异常边（简单阈值法）
            suspicious_edges = []
            for edge, weight in edge_weights.items():
                if weight > 5:  # 超过5次互动视为可疑
                    suspicious_edges.append(1)
                else:
                    suspicious_edges.append(0)

            data['edge_labels']['user_interact_anomaly'] = torch.tensor(suspicious_edges, dtype=torch.long)

        # 子图级特征（用于团伙检测）
        # 可以预计算每个连通分量的统计特征，存储在 data.subgraph_features 中

        return data
    