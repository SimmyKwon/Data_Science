import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from collections import Counter

async def extract_features_with_playwright(url):
    async with async_playwright() as p:
        # 브라우저 실행 (headless=True로 하면 창이 뜨지 않음)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            # 페이지 접속 및 로딩 대기
            await page.goto(url, wait_until="networkidle", timeout=30000)
            current_domain = urlparse(url).netloc

            # 1. 모든 하이퍼링크(<a> 태그) 추출
            # eval_on_selector_all를 사용하면 브라우저 컨텍스트 내에서 JS로 빠르게 가져옵니다.
            hrefs = await page.eval_on_selector_all("a", "elements => elements.map(e => e.href)")
            
            # --- PctNullSelfRedirectHyperlinks 계산 ---
            total_links = len(hrefs)
            null_self_link_count = 0
            
            # Playwright는 상대 경로를 절대 경로로 자동 변환해서 가져오므로 처리가 더 정확합니다.
            for href in hrefs:
                parsed_href = urlparse(href)
                # 빈 링크, JS 실행 링크, 혹은 현재 URL과 동일한 경우
                if href in [url, url + "/", "#"] or "javascript:" in href:
                    null_self_link_count += 1
                # 앵커(Anchor) 링크인 경우 (현재 페이지 내 이동)
                elif parsed_href.fragment and href.split('#')[0] == url.split('#')[0]:
                    null_self_link_count += 1
            
            pct_null_self_redirect = null_self_link_count / total_links if total_links > 0 else 0

            # --- FrequentDomainNameMismatch 계산 ---
            # a(href), img(src), link(href), script(src) 모두 조사
            resource_urls = await page.evaluate("""() => {
                const urls = [];
                document.querySelectorAll('a, img, link, script').forEach(el => {
                    const src = el.href || el.src;
                    if (src) urls.push(src);
                });
                return urls;
            }""")

            domain_list = [urlparse(u).netloc for u in resource_urls if urlparse(u).netloc]
            
            mismatch_flag = 0
            if domain_list:
                most_frequent_domain = Counter(domain_list).most_common(1)[0][0]
                if most_frequent_domain != current_domain:
                    mismatch_flag = 1

            return {
                "URL": url,
                "PctNullSelfRedirectHyperlinks": pct_null_self_redirect,
                "FrequentDomainNameMismatch": mismatch_flag
            }

        except Exception as e:
            print(f"Error: {e}")
            return None
        finally:
            await browser.close()

# 실행 예시
if __name__ == "__main__":
    target_url = "https://www.naver.com" # 테스트 URL
    result = asyncio.run(extract_features_with_playwright(target_url))
    print(result)