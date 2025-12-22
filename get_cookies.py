import asyncio
from playwright.async_api import async_playwright

async def save_cookies():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        print("正在打开百度贴吧，请在浏览器中手动登录或通过验证...")
        await page.goto("https://tieba.baidu.com/")
        input("✅ 请在浏览器中完成登录或验证，看到正常帖子列表后，在此处按回车键保存 Cookie >> ")
        await context.storage_state(path="auth.json")
        print("🎉 Cookie 已保存至 auth.json，请运行 fetcher.py 开始爬虫！")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(save_cookies())
