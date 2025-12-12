import asyncio
import os
import time
import sys
import subprocess
import requests
from playwright.async_api import async_playwright

async def wait_for_server(url, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(url)
            return True
        except requests.exceptions.ConnectionError:
            await asyncio.sleep(1)
    return False

async def comprehensive_ui_testing():
    """
    Navigate through the entire app and capture detailed screenshots
    to inspect design quality and identify issues
    """
    # Check if server is already running
    try:
        requests.get("http://localhost:7860")
        print("Server already running on port 7860")
        server_process = None
    except:
        print("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "gradio_ui_full.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        url = "http://0.0.0.0:7860"
        print(f"Waiting for server at {url}...")
        if not await wait_for_server(url.replace("0.0.0.0", "localhost")):
            print("Server failed to start")
            if server_process:
                server_process.terminate()
            return
    
    try:
        print("Server ready. Launching browser...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # headless=False to see the browser
            
            # Desktop viewport
            context = await browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2
            )
            page = await context.new_page()

            print("\n" + "="*80)
            print("COMPREHENSIVE UI INSPECTION")
            print("="*80)

            # 1. DOCUMENT Q&A TAB - Initial Load
            print("\n1. Loading homepage (Document Q&A)...")
            await page.goto("http://localhost:7860")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_selector(".header", timeout=10000)
            await asyncio.sleep(1)  # Let animations complete
            await page.screenshot(path="screenshots/flow_01_home_initial.png", full_page=True)
            print("   ✓ Captured: flow_01_home_initial.png")

            # 2. Test sub-tab switching in Document Q&A
            print("\n2. Testing Article sub-tab...")
            await page.click("button:has-text('Article')")
            await asyncio.sleep(0.5)
            await page.screenshot(path="screenshots/flow_02_article_subtab.png", full_page=True)
            print("   ✓ Captured: flow_02_article_subtab.png")
            
            print("\n3. Testing File sub-tab...")
            await page.click("button:has-text('File')")
            await asyncio.sleep(0.5)
            await page.screenshot(path="screenshots/flow_03_file_subtab.png", full_page=True)
            print("   ✓ Captured: flow_03_file_subtab.png")

            # 3. AI ASSISTANT TAB
            print("\n4. Navigating to AI Assistant...")
            await page.click("button:has-text('AI Assistant')")
            await asyncio.sleep(0.8)
            await page.screenshot(path="screenshots/flow_04_ai_assistant.png", full_page=True)
            print("   ✓ Captured: flow_04_ai_assistant.png")
            
            # Test slider interactions
            print("\n5. Testing slider interactions...")
            # Hover over slider to check styling
            await page.hover("input[type=range]")
            await asyncio.sleep(0.3)
            await page.screenshot(path="screenshots/flow_05_sliders_hover.png", full_page=True)
            print("   ✓ Captured: flow_05_sliders_hover.png")

            # 4. FUN TOOLS TAB
            print("\n6. Navigating to Fun Tools...")
            await page.click("button:has-text('Fun Tools')")
            await asyncio.sleep(0.8)
            await page.screenshot(path="screenshots/flow_06_fun_tools_city.png", full_page=True)
            print("   ✓ Captured: flow_06_fun_tools_city.png")
            
            print("\n7. Switching to Cravings sub-tab...")
            await page.click("button:has-text('Cravings')")
            await asyncio.sleep(0.5)
            await page.screenshot(path="screenshots/flow_07_fun_tools_cravings.png", full_page=True)
            print("   ✓ Captured: flow_07_fun_tools_cravings.png")

            # 5. IMAGE STUDIO TAB
            print("\n8. Navigating to Image Studio...")
            await page.click("button:has-text('Image Studio')")
            await asyncio.sleep(0.8)
            await page.screenshot(path="screenshots/flow_08_image_studio_generate.png", full_page=True)
            print("   ✓ Captured: flow_08_image_studio_generate.png")
            
            print("\n9. Switching to Edit mode...")
            await page.click("button:has-text('Edit')")
            await asyncio.sleep(0.5)
            await page.screenshot(path="screenshots/flow_09_image_studio_edit.png", full_page=True)
            print("   ✓ Captured: flow_09_image_studio_edit.png")

            # 6. MOBILE VIEWPORT TEST
            print("\n10. Testing mobile viewport (375px)...")
            await context.close()
            mobile_context = await browser.new_context(
                viewport={"width": 375, "height": 812},
                device_scale_factor=2,
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
            )
            mobile_page = await mobile_context.new_page()
            await mobile_page.goto("http://localhost:7860")
            await mobile_page.wait_for_load_state("networkidle")
            await asyncio.sleep(1)
            await mobile_page.screenshot(path="screenshots/flow_10_mobile_docqa.png", full_page=True)
            print("   ✓ Captured: flow_10_mobile_docqa.png")
            
            print("\n11. Mobile - AI Assistant...")
            await mobile_page.click("button:has-text('AI Assistant')")
            await asyncio.sleep(0.8)
            await mobile_page.screenshot(path="screenshots/flow_11_mobile_ai.png", full_page=True)
            print("   ✓ Captured: flow_11_mobile_ai.png")

            await mobile_context.close()
            await browser.close()

            print("\n" + "="*80)
            print("SCREENSHOT CAPTURE COMPLETE")
            print("="*80)
            print("\nAll screenshots saved to: screenshots/")
            print("\nInspection Points:")
            print("  1. Typography hierarchy (Playfair Display gold titles)")
            print("  2. Spacing consistency (8px scale)")
            print("  3. Glassmorphism effect on cards")
            print("  4. Button hover states and gold gradients")
            print("  5. Input focus states with gold border")
            print("  6. Tab navigation and active states")
            print("  7. Mobile responsiveness")
            print("  8. Chat bubble styling")

    except Exception as e:
        print(f"Error during UI testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server_process:
            print("\nStopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    asyncio.run(comprehensive_ui_testing())

