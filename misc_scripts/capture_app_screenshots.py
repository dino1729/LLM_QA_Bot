import asyncio
import os
import time
import subprocess
import requests
from playwright.async_api import async_playwright

import sys

async def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url)
            return True
        except requests.exceptions.ConnectionError:
            await asyncio.sleep(1)
    return False

async def capture_screenshots():
    # Start the server
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "gradio_ui_full.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        url = "http://0.0.0.0:7860"
        print(f"Waiting for server at {url}...")
        if not await wait_for_server(url.replace("0.0.0.0", "localhost")):
            print("Server failed to start")
            return

        print("Server started. Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()

            # 1. Home / Document Q&A
            print("Navigating to Home...")
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            # Wait for React hydration
            await page.wait_for_selector(".header") 
            await page.screenshot(path="screenshots/01_docqa_tab.png", full_page=True)
            print("Captured 01_docqa_tab.png")

            # 2. AI Assistant
            print("Clicking AI Assistant tab...")
            await page.click("button:has-text('AI Assistant')")
            await page.wait_for_timeout(500) # Animation
            await page.screenshot(path="screenshots/02_ai_assistant_tab.png", full_page=True)
            print("Captured 02_ai_assistant_tab.png")

            # 3. Fun Tools
            print("Clicking Fun Tools tab...")
            await page.click("button:has-text('Fun Tools')")
            await page.wait_for_timeout(500)
            await page.screenshot(path="screenshots/03_fun_tools_tab.png", full_page=True)
            print("Captured 03_fun_tools_tab.png")

            # 4. Image Studio
            print("Clicking Image Studio tab...")
            await page.click("button:has-text('Image Studio')")
            await page.wait_for_timeout(500)
            await page.screenshot(path="screenshots/04_image_studio_tab.png", full_page=True)
            print("Captured 04_image_studio_tab.png")

            await browser.close()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    asyncio.run(capture_screenshots())

