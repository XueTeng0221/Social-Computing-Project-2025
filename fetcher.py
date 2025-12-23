import asyncio
import random
import re
import os
import argparse
import json
import logging
import pandas as pd
from urllib.parse import quote
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='fetcher.log', filemode='w',
                    encoding='utf-8')
logger = logging.getLogger(__name__)

argp = argparse.ArgumentParser()
argp.add_argument('--tieba', type=str, default='三角洲行动陪玩', help='贴吧名称')
argp.add_argument('--max-pages', type=int, default=2, help='最大页数')
argp.add_argument('--max-scrolls', type=int, default=10, help='每个帖子最大滚动次数')
argp.add_argument('--max-floor', type=int, default=50, help='每个帖子最大爬取楼层数')
argp.add_argument('--output', type=str, default='data', help='输出目录')
argp.add_argument('--concurrency', type=int, default=8, help='并发数')
argp.add_argument('--headless', type=bool, default=True, help='是否无头模式运行')
args = argp.parse_args()


class AsyncTiebaFetcher:
    def __init__(self, tieba_name, max_pages=5, delay_range=(4, 8), concurrency=3):
        self.tieba_name = tieba_name
        self.max_pages = max_pages
        self.delay_range = delay_range
        self.base_url = "https://tieba.baidu.com"
        self.list_url_template = f"{self.base_url}/f?kw={quote(tieba_name)}&pn={{page}}"
        self.thread_url_template = f"{self.base_url}/p/{{tid}}"
        self.user_url_template = f"{self.base_url}/home/main?un={{username}}&fr=pb"
        self.users_data = {}
        self.seen_posts = set()
        self.seen_users = set()
        self.sem = asyncio.Semaphore(concurrency)
        self.browser = None
        self.context = None

    async def init_browser(self):
        """初始化 Playwright 浏览器"""
        logger.info("正在启动 Playwright...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=args.headless,
            args=['--no-sandbox', '--disable-setuid-sandbox',
                  '--disable-blink-features=AutomationControlled', '--disable-dev-shm-usage', '--disable-quic']
        )
        
        if os.path.exists('auth.json'):
            logger.info("正在加载本地 Cookie (auth.json)...")
            self.context = await self.browser.new_context(
                storage_state='auth.json',
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
        else:
            logger.warning("⚠️ 未找到 auth.json，将使用无痕模式（极易被重定向验证）")
            self.context = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )

        await self.context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")


    async def close(self):
        """关闭资源"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("Playwright 已关闭")

    async def _random_sleep(self):
        await asyncio.sleep(random.uniform(*self.delay_range))

    async def fetch_thread_list(self):
        """爬取帖子列表"""
        thread_list = []
        page = await self.context.new_page()

        try:
            for idx in range(self.max_pages):
                pn_page = idx + 2
                pn = pn_page * 50
                url = self.list_url_template.format(page=pn)
                logger.info(f"正在爬取列表第 {pn_page} 页: {url}")

                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                except Exception as e:
                    logger.warning(f"加载列表页失败: {e}")
                    continue

                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')
                title_links = soup.select(
                    'li.j_thread_list.clearfix.thread_item_box')

                for thread in title_links:
                    try:
                        data_field = thread.get('data-field')
                        thread_info = json.loads(data_field)
                        tid = thread_info.get('id')

                        if not tid or tid in self.seen_posts:
                            continue

                        title_tag = thread.find('a', class_='j_th_tit')
                        title = title_tag.text.strip() if title_tag else "无标题"
                        author_tag = thread.find('span', class_='tb_icon_author') or thread.find(
                            'a', class_='frs-author-name')
                        author = author_tag.text.strip() if author_tag else "匿名"

                        thread_list.append({
                            'tid': tid,
                            'title': title,
                            'author': author,
                            'url': f"{self.base_url}/p/{tid}"
                        })
                        self.seen_posts.add(tid)
                    except Exception as e:
                        continue
        finally:
            await page.close()

        logger.info(f"✅ 共获取 {len(thread_list)} 个帖子")
        return thread_list

    async def fetch_thread_detail(self, tid, max_floors=50, max_scrolls=5):
        """并发爬取单个帖子详情"""
        async with self.sem:  # 限制并发数
            logger.info(f"开始处理帖子 ID: {tid}")
            floors = []
            page = await self.context.new_page()
            await page.route("**/*.{png,jpg,jpeg,gif,webp,svg}", lambda route: route.abort())
            await page.route("**/*.{mp4,avi,flv}", lambda route: route.abort())
            try:
                url = self.thread_url_template.format(tid=tid)
                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=20000)
                except PlaywrightTimeoutError:
                    logger.warning(f"帖子 {tid} 加载超时，尝试解析已加载内容")
                    await page.evaluate("window.stop()")

                # 模拟滚动加载
                last_height = await page.evaluate("document.body.scrollHeight")
                for _ in range(max_scrolls):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(1500)
                    new_height = await page.evaluate("document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height

                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                floor_divs = soup.find_all('div', class_='l_post')
                for idx, floor_div in enumerate(floor_divs[:max_floors]):
                    try:
                        data_field = floor_div.get('data-field')
                        if not data_field:
                            continue

                        floor_info = json.loads(data_field.replace("'", '"'))
                        author_info = floor_info.get('author', {})
                        content_info = floor_info.get('content', {})

                        author_id = author_info.get(
                            'portrait', '').split('?')[0]
                        author_name = author_info.get('user_name', '')

                        # 收集用户信息用于后续爬取
                        if author_id and author_id not in self.seen_users:
                            self.seen_users.add(author_id)
                            self.users_data[author_id] = {
                                'user_id': author_id, 'user_name': author_name}

                        # 主楼层内容
                        content_div = floor_div.find(
                            'div', class_='d_post_content')
                        clean_content = content_div.get_text(
                            strip=True) if content_div else ""
                        media_urls = []
                        img_tags = floor_div.find_all('img')
                        for img in img_tags:
                            img_url = img.get('src') or img.get('data-original')
                            if img_url:
                                media_urls.append(img_url.split('/')[-1])
                        
                        post_info = {
                            'post_id': content_info.get('post_id'),
                            'content': clean_content,
                            'user_id': author_id,
                            'user_name': author_name,
                            'floor_num': content_info.get('post_no'),
                            'is_repost': 0,
                            'parent_post_id': None,
                            'media_urls': ','.join(media_urls) if media_urls else None,
                            'thread_id': tid
                        }
                        floors.append(post_info)

                        # 解析楼中楼
                        lzl_replies = floor_div.select('li.lzl_single_post')
                        for lzl in lzl_replies:
                            try:
                                lzl_data = json.loads(
                                    lzl.get('data-field', '{}').replace("'", '"'))
                                lzl_author_id = lzl_data.get('portrait')
                                lzl_name = lzl_data.get('user_name')

                                if lzl_author_id and lzl_author_id not in self.seen_users:
                                    self.seen_users.add(lzl_author_id)
                                    self.users_data[lzl_author_id] = {
                                        'user_id': lzl_author_id, 'user_name': lzl_name}

                                lzl_content_span = lzl.find(
                                    'span', class_='lzl_content_main')
                                floors.append({
                                    'post_id': lzl_data.get('spid') or content_info.get('post_id'),
                                    'content': lzl_content_span.get_text(strip=True) if lzl_content_span else "",
                                    'user_id': lzl_author_id,
                                    'user_name': lzl_name,
                                    'floor_num': f"{content_info.get('post_no')}.sub",
                                    'is_repost': 1,
                                    'parent_post_id': lzl_data.get('pid'),
                                    'media_urls': None,
                                    'thread_id': tid
                                })
                            except:
                                pass

                    except Exception as e:
                        continue

            except Exception as e:
                logger.error(f"处理帖子 {tid} 异常: {e}")
            finally:
                await page.close()
                await self._random_sleep()

            return floors

    async def fetch_user_info(self, username):
        """并发爬取用户信息"""
        async with self.sem:
            def safe_int(element, pattern=None):
                """安全提取整数，支持正则匹配"""
                try:
                    if element is None:
                        return 0
                    text = element.get_text(strip=True)
                    if pattern:
                        match = re.search(pattern, text)
                        return int(match.group(1)) if match else 0
                    return int(re.sub(r'\D', '', text)) if text else 0
                except (ValueError, AttributeError):
                    return 0
                
            if not username:
                return None
            logger.info(f"爬取用户: {username}")
            page = await self.context.new_page()
            user_info = {}

            try:
                url = self.user_url_template.format(username=quote(username))
                try:
                    await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                except:
                    pass

                content = await page.content()
                soup = BeautifulSoup(content, 'html.parser')
                userdata_div = soup.find('div', class_='userinfo_userdata')
                if userdata_div:
                    text_spans = [span.get_text(strip=True) for span in userdata_div.find_all('span') 
                                if 'userinfo_split' not in span.get('class', [])]
                    
                    logger.info(f"提取到的用户数据: {text_spans}")
                    
                    for span_text in text_spans:
                        if '吧龄' in span_text:
                            age_match = re.search(r'吧龄:([\d.]+)年?', span_text)
                            user_info['reg_time'] = float(age_match.group(1))
                            
                        if '发贴' in span_text or '发帖' in span_text:
                            post_match = re.search(r'发[贴帖]:(\d+)', span_text)
                            user_info['post_count'] = int(post_match.group(1))
                            
                        user_info['verified'] = '会员天数' in span_text
                
                # 粉丝关注
                concern_nums = soup.find_all('span', class_='concern_num')
                logger.info(f"找到 {len(concern_nums)} 个关注数据标签")
                if len(concern_nums) >= 2:
                    fans_link = concern_nums[1].find('a')
                    user_info['follower_count'] = safe_int(fans_link, r'(\d+)')
                    follow_link = concern_nums[0].find('a')
                    user_info['following_count'] = safe_int(follow_link, r'(\d+)')
                
                avatar_tag = soup.find('a', class_='userinfo_head')
                has_avatar = bool(avatar_tag and avatar_tag.find('img'))
                user_info['has_avatar'] = has_avatar
                
                logger.info(f"✓ 成功解析用户 {username}: {user_info}")
                
            except Exception as e:
                logger.warning(f"用户 {username} 爬取失败: {e}")
            finally:
                await page.close()
                await self._random_sleep()

            return user_info

    def build_relations(self, posts_df):
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
        if posts_df.empty:
            return []

        df_clean = posts_df[['post_id', 'parent_post_id', 'user_id']].copy()
        df_clean['post_id'] = (
            df_clean['post_id']
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
        )

        df_clean['parent_post_id'] = (
            df_clean['parent_post_id']
            .fillna('')
            .astype(str)
            .str.replace(r'\.0$', '', regex=True)
        )

        sources = df_clean[
            (df_clean['parent_post_id'] != '') &
            (df_clean['parent_post_id'] != 'nan')
        ][['user_id', 'parent_post_id']].rename(columns={'user_id': 'source_user_id'})

        targets = df_clean[['post_id', 'user_id']].rename(
            columns={'user_id': 'target_user_id'})

        merged_df = pd.merge(
            sources,
            targets,
            left_on='parent_post_id',
            right_on='post_id',
            how='inner'  # 仅保留能找到父帖子的互动
        )
        merged_df = merged_df[merged_df['source_user_id']
                              != merged_df['target_user_id']]
        if merged_df.empty:
            return []

        relations_df = (
            merged_df
            .groupby(['source_user_id', 'target_user_id'])
            .size()
            .reset_index(name='weight')  # 统计互动次数作为权重
        )
        relations_df['relation_type'] = 'interact'
        relations = relations_df.to_dict(orient='records')
        for r in relations:
            r['interaction_count'] = r.pop('weight')

        return relations

    async def run(self):
        await self.init_browser()
        try:
            # 1. 获取帖子列表
            thread_list = await self.fetch_thread_list()

            # 2. 并发获取帖子详情
            tasks = []
            for thread in thread_list:
                if thread['tid'] == 1:
                    continue  # 跳过置顶
                tasks.append(self.fetch_thread_detail(
                    thread['tid'],
                    max_floors=args.max_floor,
                    max_scrolls=args.max_scrolls
                ))

            # 使用 asyncio.gather 并发执行所有任务
            results = await asyncio.gather(*tasks)
            all_posts = [item for sublist in results for item in sublist]

            df_posts = pd.DataFrame(all_posts)

            # 3. 风险标签处理 (模拟)
            if not df_posts.empty:
                try:
                    with open('risk_keywords.json', 'r', encoding='utf-8') as f:
                        keywords = json.load(f).get('risk_keywords', [])
                    df_posts['label'] = df_posts['content'].apply(
                        lambda x: 1 if any(k in str(x)
                                           for k in keywords) else 0
                    )
                except FileNotFoundError:
                    df_posts['label'] = 0

            # 4. 并发补充用户信息
            logger.info(f"需要爬取 {len(self.users_data)} 个用户详情")
            user_tasks = []
            for uid, info in self.users_data.items():
                user_tasks.append(
                    (uid, self.fetch_user_info(info['user_name'])))

            # 由于 gather 无法直接绑定 ID，这里稍微处理一下
            user_results_raw = await asyncio.gather(*[t[1] for t in user_tasks])

            for i, res in enumerate(user_results_raw):
                uid = user_tasks[i][0]
                if res:
                    self.users_data[uid].update(res)
                else:
                    self.users_data[uid].update(
                        {'reg_time': 0, 'post_count': 0})

            df_users = pd.DataFrame(list(self.users_data.values()))

            # 5. 构建关系
            relations = self.build_relations(df_posts)
            df_relations = pd.DataFrame(relations)

            return df_posts, df_users, df_relations

        finally:
            await self.close()

    def save_data(self, df_posts, df_users, df_relations, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        df_posts.to_csv(f'{output_dir}/posts.csv', index=False, encoding='utf-8-sig')
        df_users.to_csv(f'{output_dir}/users.csv', index=False, encoding='utf-8-sig')
        df_relations.to_csv(f'{output_dir}/relations.csv', index=False, encoding='utf-8-sig')
        logger.info(f"数据保存至 {output_dir}")


async def main():
    fetcher = AsyncTiebaFetcher(
        tieba_name=args.tieba,
        max_pages=args.max_pages,
        concurrency=5  # 设置并发数为5
    )
    df_posts, df_users, df_relations = await fetcher.run()
    fetcher.save_data(df_posts, df_users, df_relations,
                      f'{args.output}/raw')

if __name__ == "__main__":
    # Windows下需要设置 Loop 策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
