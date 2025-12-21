import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
from datetime import datetime
from collections import defaultdict
from urllib.parse import quote
import json
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Optional, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TiebaFetcher:
    """
    百度贴吧数据爬取器
    
    功能:
        1. 爬取指定贴吧的帖子列表
        2. 爬取帖子详情页的所有楼层（含回复）
        3. 爬取用户个人主页信息
        4. 构建用户关系网络
    
    注意事项:
        - 需要设置合适的延时避免封IP
        - 建议使用代理池或Cookie池
        - 部分字段可能需要登录才能获取（如关注/粉丝数）
    """

    def __init__(self, tieba_name: str, max_pages: int = 5, delay_range: Tuple[int, int] = (3, 7)):
        """
        参数:
            tieba_name: 贴吧名称（如 "python"）
            max_pages: 爬取帖子列表的最大页数
            delay_range: 请求间隔随机延时范围（秒）- 增加延迟
        """
        self.tieba_name = tieba_name
        self.max_pages = max_pages
        self.delay_range = delay_range
        
        # 百度贴吧 URL 模板
        self.base_url = "https://tieba.baidu.com"
        self.list_url_template = f"{self.base_url}/f?kw={quote(tieba_name)}&pn={{page}}"
        self.thread_url_template = f"{self.base_url}/p/{{tid}}"
        self.user_url_template = f"{self.base_url}/home/main?un={{username}}&fr=pb"
        
        # 更完整的请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://tieba.baidu.com/',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # 创建 Session 对象以保持连接
        self.session = self._create_session()
        
        # 数据存储
        self.posts_data = []
        self.users_data = {}
        self.relations_data = []
        self.seen_posts = set()
        self.seen_users = set()

    def _create_session(self):
        """创建配置好的 requests Session"""
        session = requests.Session()
        
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 重试间隔：1, 2, 4 秒
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _request_with_retry(self, url: str, max_retries: int = 3):
        """发送HTTP请求，带重试机制 - 优化版"""
        for attempt in range(max_retries):
            try:
                # 随机延时（反爬虫）
                if attempt > 0:
                    wait_time = random.uniform(5, 10)  # 重试时等待更久
                    logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    time.sleep(random.uniform(*self.delay_range))
                
                # 动态更新 Cookie（你需要手动从浏览器获取最新的）
                cookies = {
                    'BDUSS': 'DhLMXJEbHJXNmw5M3M5aDhIZ2gwOWpJcmZERlQzbzYzS0MtYWhwWE5pcXR2MmRwSVFBQUFBJCQAAAAAAAAAAAEAAACVpW4fMzgxODAzNjgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK0yQGmtMkBpR',  # ⚠️ 需要更新
                    'STOKEN': 'faa8a44959da177afbc0b78534296fa00cfee756410f30073831aebfe833f02d',  # ⚠️ 需要更新
                    'BAIDUID': '5C50BAEC366C84488F2E13C4B2F42881:FG=1',  # 可选，从浏览器复制
                    'TIEBA_SID': 'H4sIAAAAAAAAA9MFAPiz3ZcBAAAA'
                }
                
                logger.info(f"正在请求 (尝试 {attempt+1}/{max_retries}): {url}")
                
                # 增加超时时间
                response = self.session.get(
                    url, 
                    headers=self.headers, 
                    cookies=cookies, 
                    timeout=(10, 30),  # (连接超时, 读取超时)
                    allow_redirects=True
                )
                
                response.raise_for_status()
                response.encoding = 'utf-8'
                
                logger.info(f"✓ 请求成功: {url}")
                return response
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"⏱ 超时 (尝试 {attempt+1}/{max_retries}): {e}")
                
            except requests.exceptions.HTTPError as e:
                logger.warning(f"❌ HTTP错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if e.response.status_code == 403:
                    logger.error("被服务器拒绝访问，可能需要更新Cookie或添加验证码处理")
                    return None
                    
            except Exception as e:
                logger.warning(f"⚠ 其他错误 (尝试 {attempt+1}/{max_retries}): {e}")
            
            if attempt == max_retries - 1:
                logger.error(f"💥 最终失败: {url}")
                return None
        
        return None

    
    def fetch_thread_list(self):
        """使用 Selenium 爬取（适用于动态加载页面）"""
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        thread_list = []
        
        try:
            for page in range(self.max_pages):
                pn = page * 50
                url = self.list_url_template.format(page=pn)
                logger.info(f"正在爬取第 {page+1} 页: {url}")
                
                driver.get(url)
                
                # 🔧 增加等待时间 + 多策略检查
                try:
                    # 策略1：等待帖子列表容器
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.ID, "thread_list"))
                    )
                except:
                    # 策略2：等待任意帖子链接
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/p/']"))
                    )
                
                # 🔧 滚动页面以触发懒加载
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3)
                
                # 检查是否需要验证码
                if "验证" in driver.page_source or "captcha" in driver.current_url:
                    logger.error("⚠️ 遇到验证码，需要手动处理！")
                    input("请在浏览器中完成验证后按回车继续...")
                
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
                # 🔧 多种选择器策略
                title_links = soup.select('a.j_th_tit')
                title_links += soup.select('li.j_thread_list.clearfix.thread_item_box')
            
                for i, thread in enumerate(title_links):
                    logger.info(f"正在处理第 {i+1} 条帖子...")
                    logger.info(f"帖子标题: {thread.text.strip()}")
                    try:
                        # 提取帖子ID（从data-field属性）
                        data_field = thread.get('data-field')
                        if data_field:
                            logger.info(f"数据字段: {data_field}")
                            thread_info = json.loads(data_field)
                            tid = thread_info.get('id')
                            logger.info(f"提取到帖子ID: {tid}")
                        else:
                            # 备用方案：从链接提取
                            link = thread.find('a', class_='j_th_tit')
                            logger.info(f"链接: {link}")
                            if not link:
                                continue
                            tid = re.search(r'/p/(\d+)', link['href'])
                            tid = tid.group(1) if tid else None
                            logger.info(f"提取到帖子ID: {tid}")
                        
                        if not tid or tid in self.seen_posts:
                            continue
                        
                        logger.info(f"帖子ID: {tid}")
                        
                        # 提取标题
                        title_tag = thread.find('a', class_='j_th_tit')
                        title = title_tag.text.strip() if title_tag else "无标题"
                        
                        # 提取作者
                        author_tag = thread.find('span', class_='tb_icon_author')
                        if not author_tag:
                            author_tag = thread.find('a', class_='frs-author-name')
                        author = author_tag.text.strip() if author_tag else "匿名"
                        
                        # 提取回复数
                        reply_tag = thread.find('span', class_='threadlist_rep_num')
                        reply_count = int(reply_tag.text.strip()) if reply_tag else 0
                        
                        thread_list.append({
                            'tid': tid,
                            'title': title,
                            'author': author,
                            'reply_count': reply_count,
                            'url': f"{self.base_url}/p/{tid}"
                        })
                        
                        self.seen_posts.add(tid)
                        
                    except Exception as e:
                        logger.warning(f"解析帖子失败: {e}")
                        continue
            
        finally:
            driver.quit()
        
        logger.info(f"✅ 共获取 {len(thread_list)} 个帖子")
        return thread_list
    
    def fetch_thread_detail(self, tid: str, max_floors: int = 50):
        """
        使用 Selenium 爬取帖子详情（主楼 + 楼层回复）
        
        参数:
            tid: 帖子ID
            max_floors: 最大爬取楼层数
        
        返回: List[dict] - 所有楼层的内容
        """
        url = self.thread_url_template.format(tid=tid)
        
        # 配置 Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        floors = []
        
        try:
            logger.info(f"正在爬取帖子详情: {url}")
            driver.get(url)
            
            # 等待楼层容器加载
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "l_post"))
                )
            except:
                logger.warning(f"帖子 {tid} 加载超时")
                return []
            
            # 滚动页面加载所有楼层
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 5
            
            while scroll_attempts < max_scroll_attempts:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                
                last_height = new_height
                scroll_attempts += 1
            
            # 检查是否需要验证码
            if "验证" in driver.page_source or "captcha" in driver.current_url:
                logger.error("⚠️ 遇到验证码！")
                input("请在浏览器中完成验证后按回车继续...")
            
            # 获取页面源码
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # 查找所有楼层容器
            floor_divs = soup.find_all('div', class_='l_post')
            logger.info(f"找到 {len(floor_divs)} 个楼层")
            
            for idx, floor_div in enumerate(floor_divs[:max_floors]):
                try:
                    # 解析楼层的 data-field
                    data_field = floor_div.get('data-field')
                    if not data_field:
                        continue
                    
                    floor_info = json.loads(data_field)
                    author_id = floor_info['author']['user_id']
                    author_name = floor_info['author']['user_name']
                    post_id = floor_info['content']['post_id']
                    floor_num = floor_info['content']['post_no']
                    
                    # 提取楼层内容
                    content_div = floor_div.find('div', class_='d_post_content')
                    if content_div:
                        # 移除图片和换行标签
                        for tag in content_div.find_all(['img', 'br']):
                            tag.decompose()
                        content = content_div.get_text(strip=True)
                    else:
                        content = ""
                    
                    # 提取媒体链接
                    media_urls = []
                    img_tags = floor_div.find_all('img', class_='BDE_Image')
                    for img in img_tags:
                        img_url = img.get('src') or img.get('data-original')
                        if img_url:
                            media_urls.append(img_url)
                    
                    # 判断是否是转发
                    is_repost = any(kw in content for kw in ['转发', '分享', 'RT @'])
                    
                    # 提取父帖子ID
                    parent_post_id = None
                    quote_div = floor_div.find('div', class_='post-tail-wrap')
                    if quote_div:
                        quote_link = quote_div.find('a', href=re.compile(r'pid=(\d+)'))
                        if quote_link:
                            parent_post_id = re.search(r'pid=(\d+)', quote_link['href']).group(1)
                    
                    floors.append({
                        'post_id': post_id,
                        'content': content,
                        'user_id': author_id,
                        'user_name': author_name,
                        'floor_num': floor_num,
                        'is_repost': is_repost,
                        'parent_post_id': parent_post_id,
                        'media_urls': ','.join(media_urls) if media_urls else None,
                        'thread_id': tid
                    })
                    
                    # 记录用户
                    if author_id not in self.seen_users:
                        self.seen_users.add(author_id)
                        self.users_data[author_id] = {
                            'user_id': author_id,
                            'user_name': author_name
                        }
                    
                except Exception as e:
                    logger.warning(f"解析楼层 {idx+1} 失败: {e}")
                    continue
            
            logger.info(f"✓ 成功解析 {len(floors)} 个楼层")
            
        except Exception as e:
            logger.error(f"爬取帖子 {tid} 失败: {e}")
        
        finally:
            driver.quit()
            # 随机延时
            time.sleep(random.uniform(*self.delay_range))
        
        return floors


    def fetch_user_info(self, username: str):
        """
        使用 Selenium 爬取用户个人主页信息
        
        参数:
            username: 用户名
        
        返回: dict - 用户信息
        """
        url = self.user_url_template.format(username=quote(username))
        
        # 配置 Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # 可选：添加 Cookie（如果有登录信息）
        # options.add_argument('--user-data-dir=C:/Users/YourName/AppData/Local/Google/Chrome/User Data')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        user_info = {}
        
        try:
            logger.info(f"正在爬取用户主页: {username}")
            driver.get(url)
            
            # 等待页面加载
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "userinfo_head"))
                )
            except:
                logger.warning(f"用户 {username} 主页加载超时")
                return None
            
            time.sleep(2)  # 额外等待动态内容
            
            # 检查验证码
            if "验证" in driver.page_source or "captcha" in driver.current_url:
                logger.error("⚠️ 遇到验证码！")
                input("请在浏览器中完成验证后按回车继续...")
            
            # 获取页面源码
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # 提取注册时间
            reg_time_tag = soup.find('span', class_='userinfo_title', string=re.compile('注册时间'))
            if reg_time_tag:
                reg_time = reg_time_tag.find_next('span', class_='userinfo_cont').text.strip()
                user_info['reg_time'] = reg_time
            else:
                user_info['reg_time'] = None
            
            # 提取发帖数
            post_count_tag = soup.find('span', class_='concern_num', attrs={'id': re.compile('post')})
            if not post_count_tag:
                # 备用方案：从其他位置提取
                post_count_tag = soup.find('span', string=re.compile('发贴'))
                if post_count_tag:
                    post_count_text = post_count_tag.find_next('span').text.strip()
                    user_info['post_count'] = int(re.sub(r'\D', '', post_count_text))
                else:
                    user_info['post_count'] = 0
            else:
                user_info['post_count'] = int(post_count_tag.text.strip())
            
            # 提取粉丝数和关注数
            fans_tag = soup.find('span', class_='concern_num', attrs={'id': re.compile('fans')})
            follow_tag = soup.find('span', class_='concern_num', attrs={'id': re.compile('concern')})
            
            user_info['follower_count'] = int(fans_tag.text.strip()) if fans_tag else 0
            user_info['following_count'] = int(follow_tag.text.strip()) if follow_tag else 0
            
            # 提取是否认证
            verified_tag = soup.find('img', class_='userinfo_auth')
            user_info['verified'] = bool(verified_tag)
            
            # 提取是否有头像
            avatar_tag = soup.find('img', class_='userinfo_head')
            has_avatar = False
            if avatar_tag and avatar_tag.get('src'):
                has_avatar = 'default' not in avatar_tag['src'].lower()
            user_info['has_avatar'] = has_avatar
            
            logger.info(f"✓ 成功解析用户 {username}")
            
        except Exception as e:
            logger.error(f"爬取用户 {username} 失败: {e}")
            return None
        
        finally:
            driver.quit()
            # 随机延时
            time.sleep(random.uniform(*self.delay_range))
        
        return user_info
    
    def build_user_relations(self, posts_df: pd.DataFrame):
        """
        基于互动行为构建用户关系网络
        
        规则:
            1. 回复关系 -> interact
            2. 在同一帖子多次互动 -> interact（强化）
            3. （可扩展）基于共同关注的吧、相似文本推断潜在关系
        
        参数:
            posts_df: DataFrame - 包含 user_id, parent_post_id 等字段
        
        返回: List[dict] - 关系列表
        """
        relations = []
        interaction_count = defaultdict(int)  # {(user1, user2): count}
        
        # 构建 post_id -> user_id 的映射
        post_user_map = dict(zip(posts_df['post_id'], posts_df['user_id']))
        
        for _, row in posts_df.iterrows():
            if pd.notna(row['parent_post_id']) and row['parent_post_id'] in post_user_map:
                # 当前用户回复了某个帖子 -> 建立 interact 关系
                source_user = row['user_id']
                target_user = post_user_map[row['parent_post_id']]
                
                if source_user != target_user:  # 避免自环
                    interaction_count[(source_user, target_user)] += 1
        
        # 转换为关系列表（只保留互动次数 >= 1 的关系）
        for (src, tgt), count in interaction_count.items():
            relations.append({
                'source_user_id': src,
                'target_user_id': tgt,
                'relation_type': 'interact',
                'interaction_count': count  # 可用于边权重
            })
        
        # （可选）添加 follow 关系
        # 这需要爬取用户的关注列表，百度贴吧需要登录，此处省略
        
        return relations
    
    def run(self):
        """
        执行完整的爬取流程
        
        返回: (df_posts, df_users, df_relations)
        """
        logger.info(f"🚀 开始爬取贴吧: {self.tieba_name}")
        
        # 1. 爬取帖子列表
        thread_list = self.fetch_thread_list()
        
        # 2. 爬取每个帖子的详情
        all_posts = []
        for thread in thread_list[:10]:  # 限制爬取数量，避免过长时间
            logger.info(f"正在爬取帖子: {thread['title']} (ID: {thread['tid']})")
            floors = self.fetch_thread_detail(thread['tid'])
            all_posts.extend(floors)
        
        # 3. 构建 DataFrame
        df_posts = pd.DataFrame(all_posts)
        
        # 添加伪标签（实际项目中需要人工标注）
        # 这里用简单规则：包含高风险关键词的标记为 1
        risk_keywords = ['包卡', '带单', '加群', 'QQ', '微信', 'usdt']
        df_posts['label'] = df_posts['content'].apply(
            lambda x: 1 if any(kw in str(x) for kw in risk_keywords) else 0
        )
        
        # 4. 补充用户信息（爬取个人主页）
        logger.info(f"开始补充 {len(self.users_data)} 个用户的信息...")
        for user_id, user_base_info in list(self.users_data.items())[:20]:  # 限制数量
            username = user_base_info['user_name']
            logger.info(f"爬取用户: {username}")
            user_detail = self.fetch_user_info(username)
            
            if user_detail:
                self.users_data[user_id].update(user_detail)
            else:
                # 填充默认值
                self.users_data[user_id].update({
                    'reg_time': '2020-01-01',
                    'post_count': 0,
                    'follower_count': 0,
                    'following_count': 0,
                    'verified': False,
                    'has_avatar': True
                })
        
        df_users = pd.DataFrame(list(self.users_data.values()))
        
        # 5. 构建用户关系网络
        relations = self.build_user_relations(df_posts)
        df_relations = pd.DataFrame(relations)
        
        logger.info(f"✅ 爬取完成!")
        logger.info(f"  - 帖子数: {len(df_posts)}")
        logger.info(f"  - 用户数: {len(df_users)}")
        logger.info(f"  - 关系数: {len(df_relations)}")
        
        return df_posts, df_users, df_relations
    
    def save_to_csv(self, df_posts: pd.DataFrame, df_users: pd.DataFrame, df_relations: pd.DataFrame, output_dir: str = 'data/raw/'):
        """保存为CSV文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        df_posts.to_csv(f'{output_dir}/posts.csv', index=False, encoding='utf-8-sig')
        df_users.to_csv(f'{output_dir}/users.csv', index=False, encoding='utf-8-sig')
        df_relations.to_csv(f'{output_dir}/relations.csv', index=False, encoding='utf-8-sig')
        
        logger.info(f"💾 数据已保存到 {output_dir}")


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例：爬取 "python" 贴吧
    fetcher = TiebaFetcher(
        tieba_name='三角洲行动',
        max_pages=5,
        delay_range=(2, 4)
    )
    
    df_posts, df_users, df_relations = fetcher.run()
    
    # 查看数据
    print("\n===== Posts Sample =====")
    print(df_posts.head())
    print("\n===== Users Sample =====")
    print(df_users.head())
    print("\n===== Relations Sample =====")
    print(df_relations.head())
    
    # 保存到文件
    fetcher.save_to_csv(df_posts, df_users, df_relations)
