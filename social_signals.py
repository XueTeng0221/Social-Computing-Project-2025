# social_signals.py

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict, deque
import networkx as nx

class SocialSignalExtractor:
    """
    提取三类社会信号：
    1. 关系信号 (Relational)
    2. 传播信号 (Diffusion)
    3. 行为与注意力信号 (Behavioral)
    """
    
    def __init__(self):
        self.cascade_cache = {}
        
    # ========== 3.2 传播信号 (Diffusion Signals) ==========
    
    def extract_cascade_features(self, df_posts: pd.DataFrame, 
                                 post_id: str, 
                                 time_limit: int = 60) -> Dict[str, float]:
        """
        提取单个帖子的级联传播特征
        
        参数:
        - df_posts: 帖子数据框
        - post_id: 目标帖子ID
        - time_limit: 早期预警时间窗口(分钟)
        
        返回:
        - cascade_depth: 转发树深度
        - cascade_width: 最大宽度(单层最多转发数)
        - early_growth_rate: 早期增长率(前time_limit分钟)
        - structural_entropy: 树结构熵
        """
        
        # 构建转发树
        G = nx.DiGraph()
        root_post = df_posts[df_posts['post_id'] == post_id]
        
        if root_post.empty:
            return {
                'cascade_depth': 0.0,
                'cascade_width': 0.0,
                'early_growth_rate': 0.0,
                'structural_entropy': 0.0
            }
        
        root_time = pd.to_datetime(root_post.iloc[0]['timestamp'])
        G.add_node(post_id, time=root_time)
        
        # BFS构建转发树
        queue = deque([post_id])
        level_nodes = defaultdict(list)  # 记录每层节点
        level_nodes[0] = [post_id]
        
        while queue:
            current_id = queue.popleft()
            current_node = df_posts[df_posts['post_id'] == current_id].iloc[0]
            
            # 查找所有转发当前帖子的帖子
            reposts = df_posts[
                (df_posts['parent_post_id'] == current_id) & 
                (df_posts['is_repost'] == 1)
            ]
            
            for _, repost in reposts.iterrows():
                repost_id = repost['post_id']
                repost_time = pd.to_datetime(repost['timestamp'])
                
                G.add_node(repost_id, time=repost_time)
                G.add_edge(current_id, repost_id)
                
                queue.append(repost_id)
                
                # 记录层级
                parent_level = max([lvl for lvl, nodes in level_nodes.items() 
                                   if current_id in nodes], default=0)
                level_nodes[parent_level + 1].append(repost_id)
        
        # 计算特征
        features = {}
        
        # 1. 深度 (Depth)
        if len(level_nodes) > 0:
            features['cascade_depth'] = float(max(level_nodes.keys()))
        else:
            features['cascade_depth'] = 0.0
        
        # 2. 宽度 (Width)
        if level_nodes:
            features['cascade_width'] = float(max(len(nodes) for nodes in level_nodes.values()))
        else:
            features['cascade_width'] = 0.0
        
        # 3. 早期增长率 (Early Growth Rate)
        early_nodes = [n for n in G.nodes() 
                      if (G.nodes[n]['time'] - root_time).total_seconds() <= time_limit * 60]
        features['early_growth_rate'] = len(early_nodes) / max(time_limit, 1)
        
        # 4. 结构熵 (Structural Entropy)
        features['structural_entropy'] = self._compute_tree_entropy(G, level_nodes)
        
        return features
    
    def _compute_tree_entropy(self, G: nx.DiGraph, level_nodes: Dict[int, List]) -> float:
        """
        计算转发树的结构熵
        H = -Σ (p_i * log(p_i))，其中 p_i 是每层节点数的比例
        """
        if not level_nodes or len(G.nodes()) == 0:
            return 0.0
        
        total_nodes = len(G.nodes())
        entropy = 0.0
        
        for level, nodes in level_nodes.items():
            if len(nodes) > 0:
                p_i = len(nodes) / total_nodes
                entropy -= p_i * np.log2(p_i + 1e-10)  # 避免log(0)
        
        return float(entropy)
    
    def batch_extract_cascade_features(self, df_posts: pd.DataFrame, 
                                       time_limit: int = 60) -> pd.DataFrame:
        """
        批量提取所有原始帖子的级联特征
        """
        # 只对原始帖子(非转发)提取级联特征
        original_posts = df_posts[df_posts['is_repost'] == 0]['post_id'].unique()
        
        cascade_features = []
        
        for post_id in original_posts:
            features = self.extract_cascade_features(df_posts, post_id, time_limit)
            features['post_id'] = post_id
            cascade_features.append(features)
        
        # 转发帖子继承原始帖子的级联特征
        cascade_df = pd.DataFrame(cascade_features)
        
        # 为转发帖子填充特征
        result = df_posts[['post_id', 'parent_post_id', 'is_repost']].copy()
        result = result.merge(cascade_df, on='post_id', how='left')
        
        # 转发帖子使用其根帖子的特征
        for idx, row in result.iterrows():
            if row['is_repost'] == 1 and pd.notna(row['parent_post_id']):
                parent_features = cascade_df[cascade_df['post_id'] == row['parent_post_id']]
                if not parent_features.empty:
                    for col in ['cascade_depth', 'cascade_width', 'early_growth_rate', 'structural_entropy']:
                        result.at[idx, col] = parent_features.iloc[0][col]
        
        # 填充缺失值
        result = result.fillna(0.0)
        
        return result[['post_id', 'cascade_depth', 'cascade_width', 
                       'early_growth_rate', 'structural_entropy']]
    
    # ========== 3.3 行为与注意力信号 (Behavioral Signals) ==========
    
    def compute_user_behavioral_features(self, df_posts: pd.DataFrame, 
                                         df_users: pd.DataFrame) -> pd.DataFrame:
        """
        计算用户行为特征
        
        特征:
        - ignore_nudge_rate: 用户忽略警告后仍分享的比例 (模拟)
        - avg_share_delay: 平均分享延迟(小时)
        - risky_content_ratio: 发布风险内容的比例
        """
        user_features = []
        
        for user_id in df_users['user_id'].unique():
            user_posts = df_posts[df_posts['user_id'] == user_id]
            
            if len(user_posts) == 0:
                user_features.append({
                    'user_id': user_id,
                    'ignore_nudge_rate': 0.0,
                    'avg_share_delay': 0.0,
                    'risky_content_ratio': 0.0
                })
                continue
            
            # 1. 风险内容比例
            risky_ratio = user_posts['label'].sum() / len(user_posts)
            
            # 2. 平均分享延迟 (模拟：转发帖子与原帖的时间差)
            repost_delays = []
            for _, post in user_posts[user_posts['is_repost'] == 1].iterrows():
                if pd.notna(post['parent_post_id']):
                    parent = df_posts[df_posts['post_id'] == post['parent_post_id']]
                    if not parent.empty:
                        delay = (pd.to_datetime(post['timestamp']) - 
                                pd.to_datetime(parent.iloc[0]['timestamp'])).total_seconds() / 3600
                        repost_delays.append(max(delay, 0))
            
            avg_delay = np.mean(repost_delays) if repost_delays else 0.0
            
            # 3. 忽略提醒率 (模拟：如果用户发布风险内容后继续发布，视为忽略)
            sorted_posts = user_posts.sort_values('timestamp')
            ignore_count = 0
            total_risky = 0
            
            for i in range(len(sorted_posts) - 1):
                if sorted_posts.iloc[i]['label'] == 1:
                    total_risky += 1
                    # 如果之后还发布了帖子，视为忽略
                    if i < len(sorted_posts) - 1:
                        ignore_count += 1
            
            ignore_rate = ignore_count / max(total_risky, 1)
            
            user_features.append({
                'user_id': user_id,
                'ignore_nudge_rate': ignore_rate,
                'avg_share_delay': avg_delay,
                'risky_content_ratio': risky_ratio
            })
        
        return pd.DataFrame(user_features)
    
    # ========== 3.1 关系信号辅助函数 ==========
    
    def compute_neighbor_risk_score(self, user_id: str, df_relations: pd.DataFrame,
                                    user_risk_dict: Dict[str, float]) -> float:
        """
        计算用户邻居的平均风险分数
        用于构建关系信号特征
        """
        neighbors = df_relations[
            (df_relations['source_user_id'] == user_id) |
            (df_relations['target_user_id'] == user_id)
        ]
        
        neighbor_ids = set()
        for _, row in neighbors.iterrows():
            if row['source_user_id'] == user_id:
                neighbor_ids.add(row['target_user_id'])
            else:
                neighbor_ids.add(row['source_user_id'])
        
        if not neighbor_ids:
            return 0.0
        
        risk_scores = [user_risk_dict.get(nid, 0.0) for nid in neighbor_ids]
        return np.mean(risk_scores)
    
    def extract_interaction_homogeneity(self, df_relations: pd.DataFrame) -> Dict[str, float]:
        """
        计算用户互动同质性
        返回每个用户与其邻居的互动重叠度
        """
        G = nx.Graph()
        for _, row in df_relations.iterrows():
            if row['relation_type'] == 'interact':
                G.add_edge(row['source_user_id'], row['target_user_id'])
        
        homogeneity = {}
        for user in G.nodes():
            neighbors = set(G.neighbors(user))
            if len(neighbors) < 2:
                homogeneity[user] = 0.0
                continue
            
            # 计算邻居之间的连接密度
            subgraph = G.subgraph(neighbors)
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = subgraph.number_of_edges()
            
            homogeneity[user] = actual_edges / max(possible_edges, 1)
        
        return homogeneity
