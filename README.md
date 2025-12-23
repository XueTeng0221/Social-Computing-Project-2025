# Social-Computing-Project-2025

社会计算结课项目（2025秋武大网安）

## ScamHeteroFusion: 异构社会图融合的金融诈骗检测框架

### Quick Start

```aiignore
cd Social-Computing-Project-2025
mkdir data
mkdir data/raw data/processed
mkdir results
```

```aiignore
# Linux
python3 -m venv .venv/
source .venv/bin/activate
pip3 install -r requirements.txt; playwright install # 加载爬虫 Driver
python3 -u get_cookies.py
python3 -u fetcher.py --output=./data --tieba=三角洲行动陪玩 --max-pages=100  --max-floor=50 --concurrency=5 --headless=True # 启动爬虫
python3 -u main.py --alpha=0.7 --gamma=2 --epochs=50 --save-dir=./results --force-rebuild # 启动训练
```

```aiignore
# Windows
python -m venv .venv/
.\.venv\Scripts\activate
pip install -r requirements.txt; playwright install # 加载爬虫 Driver
python -u get_cookies.py
python -u fetcher.py --output=./data --tieba=三角洲行动陪玩 --max-pages=100  --max-floor=50 --concurrency=5 --headless=True # 启动爬虫
python -u main.py --alpha=0.7 --gamma=2 --epochs=50 --save-dir=./results --force-rebuild # 启动训练
```
