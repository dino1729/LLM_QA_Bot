#!/usr/bin/env python3
"""
Generate screenshots of the newsletter HTML files at multiple viewport sizes.

Reads HTML files from: newsletter_research_data/
Saves screenshots to: screenshots/
"""
import os
import sys
from pathlib import Path


# Project root is parent of misc_scripts/
PROJECT_ROOT = Path(__file__).parent.parent
HTML_INPUT_DIR = PROJECT_ROOT / "newsletter_research_data"
SCREENSHOTS_OUTPUT_DIR = PROJECT_ROOT / "screenshots"


def try_html2image_screenshots():
    """Try taking screenshots with html2image"""
    try:
        from html2image import Html2Image
        import time

        SCREENSHOTS_OUTPUT_DIR.mkdir(exist_ok=True)

        # Viewport configurations
        viewports = [
            {"name": "mobile_320", "width": 320, "height": 800},
            {"name": "mobile_375", "width": 375, "height": 812},
            {"name": "mobile_414", "width": 414, "height": 896},
            {"name": "tablet_768", "width": 768, "height": 1024},
            {"name": "desktop_1024", "width": 1024, "height": 900},
        ]

        html_files = [
            "test_newsletter_progress.html",
            "test_newsletter_news.html",
        ]

        # Initialize Html2Image
        hti = Html2Image(
            output_path=str(SCREENSHOTS_OUTPUT_DIR),
            custom_flags=[
                '--no-sandbox',
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--hide-scrollbars',
            ]
        )

        for html_file in html_files:
            file_path = HTML_INPUT_DIR / html_file
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue

            # Read HTML content
            html_content = file_path.read_text()
            base_name = html_file.replace(".html", "")

            for viewport in viewports:
                screenshot_name = f"{base_name}_{viewport['name']}.png"
                print(f"Taking screenshot: {screenshot_name}")

                try:
                    hti.screenshot(
                        html_str=html_content,
                        save_as=screenshot_name,
                        size=(viewport["width"], viewport["height"])
                    )
                    print(f"  Saved: {SCREENSHOTS_OUTPUT_DIR / screenshot_name}")
                except Exception as e:
                    print(f"  Failed: {e}")

        return True
    except Exception as e:
        print(f"html2image failed: {e}")
        return False


def try_playwright_screenshots():
    """Try taking screenshots with Playwright"""
    try:
        import asyncio
        from playwright.async_api import async_playwright

        async def take_screenshots():
            viewports = [
                {"name": "mobile_320", "width": 320, "height": 800},
                {"name": "mobile_375", "width": 375, "height": 812},
                {"name": "mobile_414", "width": 414, "height": 896},
                {"name": "tablet_768", "width": 768, "height": 1024},
                {"name": "desktop_1024", "width": 1024, "height": 900},
            ]

            html_files = [
                "test_newsletter_progress.html",
                "test_newsletter_news.html",
            ]

            SCREENSHOTS_OUTPUT_DIR.mkdir(exist_ok=True)

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)

                for html_file in html_files:
                    file_path = HTML_INPUT_DIR / html_file
                    if not file_path.exists():
                        print(f"File not found: {file_path}")
                        continue

                    file_url = f"file://{file_path.absolute()}"
                    base_name = html_file.replace(".html", "")

                    for viewport in viewports:
                        # Generate both themes explicitly so you can validate dark/light rendering.
                        theme_variants = [
                            {"suffix": "_dark", "color_scheme": "dark"},
                            {"suffix": "_light", "color_scheme": "light"},
                        ]

                        for theme in theme_variants:
                            screenshot_name = f"{base_name}_{viewport['name']}{theme['suffix']}.png"
                            print(f"Taking screenshot: {screenshot_name}")

                            context = await browser.new_context(
                                viewport={"width": viewport["width"], "height": viewport["height"]},
                                device_scale_factor=2,
                                color_scheme=theme["color_scheme"],
                            )
                            page = await context.new_page()
                            await page.goto(file_url)
                            await page.wait_for_load_state("networkidle")
                            await asyncio.sleep(0.5)

                            screenshot_path = SCREENSHOTS_OUTPUT_DIR / screenshot_name
                            await page.screenshot(path=str(screenshot_path), full_page=True)
                            print(f"  Saved: {screenshot_path}")
                            await context.close()

                await browser.close()

        asyncio.run(take_screenshots())
        return True
    except Exception as e:
        print(f"Playwright failed: {e}")
        return False


def generate_html_preview():
    """Generate an HTML preview file that shows all viewports side by side"""
    preview_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Newsletter UI Preview</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #fff;
            padding: 20px;
        }
        h1 { text-align: center; margin-bottom: 30px; color: #D4AF37; }
        h2 { margin: 30px 0 20px; color: #aaa; font-size: 14px; text-transform: uppercase; letter-spacing: 2px; }
        .preview-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .preview-frame {
            background: #2a2a2a;
            border-radius: 12px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .frame-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        iframe {
            border: 1px solid #333;
            border-radius: 8px;
            background: #000;
        }
        .instructions {
            max-width: 800px;
            margin: 0 auto 40px;
            padding: 20px;
            background: #2a2a2a;
            border-radius: 12px;
            text-align: center;
        }
        .instructions p {
            color: #aaa;
            line-height: 1.6;
        }
        .instructions code {
            background: #333;
            padding: 2px 6px;
            border-radius: 4px;
            color: #D4AF37;
        }
    </style>
</head>
<body>
    <h1>Newsletter UI Preview</h1>

    <div class="instructions">
        <p>Open this file in a browser to preview the newsletter at different viewport sizes.</p>
        <p>Toggle your system's dark/light mode to see theme switching in action.</p>
        <p>Use browser DevTools (<code>F12</code>) to test additional viewport sizes.</p>
    </div>

    <h2>Daily Briefing Newsletter</h2>
    <div class="preview-grid">
        <div class="preview-frame">
            <span class="frame-label">Mobile 320px</span>
            <iframe src="test_newsletter_news.html" width="320" height="600" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Mobile 375px</span>
            <iframe src="test_newsletter_news.html" width="375" height="667" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Mobile 414px</span>
            <iframe src="test_newsletter_news.html" width="414" height="736" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Tablet 768px</span>
            <iframe src="test_newsletter_news.html" width="768" height="600" loading="lazy"></iframe>
        </div>
    </div>

    <h2>Year Progress Report</h2>
    <div class="preview-grid">
        <div class="preview-frame">
            <span class="frame-label">Mobile 320px</span>
            <iframe src="test_newsletter_progress.html" width="320" height="600" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Mobile 375px</span>
            <iframe src="test_newsletter_progress.html" width="375" height="667" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Mobile 414px</span>
            <iframe src="test_newsletter_progress.html" width="414" height="736" loading="lazy"></iframe>
        </div>
        <div class="preview-frame">
            <span class="frame-label">Tablet 768px</span>
            <iframe src="test_newsletter_progress.html" width="768" height="600" loading="lazy"></iframe>
        </div>
    </div>
</body>
</html>
"""

    HTML_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = HTML_INPUT_DIR / "ui_preview.html"
    preview_path.write_text(preview_html)
    print(f"Created UI preview: {preview_path}")
    return preview_path


if __name__ == "__main__":
    print("=" * 60)
    print("Newsletter UI Screenshot Generator")
    print("=" * 60)
    print(f"\nHTML input dir: {HTML_INPUT_DIR}")
    print(f"Screenshots output dir: {SCREENSHOTS_OUTPUT_DIR}")

    # Always generate the HTML preview (works without any dependencies)
    preview_path = generate_html_preview()
    print(f"\n✓ HTML preview generated: {preview_path}")

    # Try different screenshot methods
    print("\nAttempting to generate PNG screenshots...")

    success = False

    # Try html2image first
    print("\n1. Trying html2image...")
    if try_html2image_screenshots():
        success = True

    # Try Playwright as fallback
    if not success:
        print("\n2. Trying Playwright...")
        if try_playwright_screenshots():
            success = True

    if success:
        print(f"\n✓ Screenshots saved to {SCREENSHOTS_OUTPUT_DIR}")
    else:
        print("\n⚠ Could not generate PNG screenshots.")
        print("  A browser (Chrome/Chromium) must be installed on the system.")
        print(f"\n  Use the HTML preview file instead: {preview_path}")
