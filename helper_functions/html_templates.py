from datetime import datetime
from typing import Any, Dict, List

from helper_functions.weather_utils import get_weather


def get_luxurious_css():
    """
    Ultra-premium, mobile-first CSS for the newsletter.
    Follows the Architecture Spec: 8px spacing scale, 44px tap targets,
    dark mode support, and fluid responsive design.
    """
    return """
        <style>
            /* Google Fonts with system fallbacks */
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

            /* ============================================
               DESIGN TOKENS - 8px Spacing Scale
               ============================================ */
            :root {
                /* Spacing Scale (8px base) */
                --space-xs: 8px;
                --space-sm: 16px;
                --space-md: 24px;
                --space-lg: 32px;
                --space-xl: 48px;
                --space-2xl: 64px;

                /* Colors - Dark Theme */
                --bg-primary: #0A0B0D;
                --bg-secondary: #12141A;
                --bg-card: rgba(22, 25, 32, 0.85);
                --bg-card-hover: rgba(28, 32, 42, 0.9);
                --border-subtle: rgba(255, 255, 255, 0.06);
                --border-medium: rgba(255, 255, 255, 0.1);

                /* Text Colors */
                --text-primary: #F5F5F7;
                --text-secondary: #A1A1AA;
                --text-muted: #6B7280;

                /* Accent Colors */
                --accent-gold: #D4AF37;
                --accent-gold-light: #E8C959;
                --accent-gold-dark: #B8962E;
                --accent-glow: rgba(212, 175, 55, 0.15);
                --accent-glow-strong: rgba(212, 175, 55, 0.3);

                /* Typography */
                --font-display: 'Playfair Display', Georgia, 'Times New Roman', serif;
                --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

                /* Sizing */
                --tap-target-min: 44px;
                --border-radius-sm: 8px;
                --border-radius-md: 12px;
                --border-radius-lg: 16px;
                --border-radius-xl: 24px;

                /* Shadows */
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.25);
                --shadow-glow: 0 0 20px var(--accent-glow);

                /* Transitions */
                --transition-fast: 150ms ease;
                --transition-base: 250ms ease;
                --transition-slow: 400ms ease;
            }

            /* ============================================
               RESET & BASE STYLES (Mobile-First)
               ============================================ */
            *, *::before, *::after {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                padding: 0;
                background: var(--bg-primary);
                background-image:
                    radial-gradient(ellipse at top, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                background-attachment: fixed;
                color: var(--text-primary);
                font-family: var(--font-body);
                font-size: 16px;
                line-height: 1.6;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
                min-height: 100vh;
            }

            /* ============================================
               CONTAINER - Fluid with Max Width
               ============================================ */
            .container {
                width: 100%;
                max-width: 640px;
                margin: 0 auto;
                padding: var(--space-sm);
            }

            /* ============================================
               HEADER
               ============================================ */
            .header {
                text-align: center;
                padding: var(--space-lg) 0 var(--space-md);
                margin-bottom: var(--space-md);
                position: relative;
            }

            .header::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 60px;
                height: 2px;
                background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
            }

            h1 {
                font-family: var(--font-display);
                font-size: 28px;
                font-weight: 700;
                color: var(--accent-gold);
                margin: 0 0 var(--space-xs) 0;
                letter-spacing: 0.5px;
                text-shadow: 0 2px 20px var(--accent-glow);
            }

            .subtitle {
                color: var(--text-secondary);
                font-size: 14px;
                font-family: var(--font-body);
                font-weight: 400;
            }

            .date-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: var(--tap-target-min);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: var(--text-secondary);
                border: 1px solid var(--border-subtle);
                padding: var(--space-xs) var(--space-md);
                border-radius: 100px;
                background: rgba(255, 255, 255, 0.02);
                margin-top: var(--space-sm);
                transition: all var(--transition-base);
            }

            /* ============================================
               THEME TOGGLE (Browser Preview Only)
               NOTE: Most email clients strip <script> and/or block buttons.
               This is safe to include (no-op in email), and works in browsers.
               ============================================ */
            .theme-toggle {
                position: absolute;
                top: var(--space-sm);
                right: var(--space-sm);
                width: var(--tap-target-min);
                height: var(--tap-target-min);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                border: 1px solid var(--border-medium);
                background: rgba(255, 255, 255, 0.03);
                color: var(--text-primary);
                cursor: pointer;
                font-size: 16px;
                line-height: 1;
                user-select: none;
                -webkit-tap-highlight-color: transparent;
                transition: transform var(--transition-fast), background var(--transition-fast);
            }

            .theme-toggle:hover {
                background: rgba(255, 255, 255, 0.06);
            }

            .theme-toggle:active {
                transform: scale(0.98);
            }

            /* ============================================
               CARDS - Glass Morphism Effect
               ============================================ */
            .card {
                background: var(--bg-card);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid var(--border-subtle);
                border-radius: var(--border-radius-lg);
                padding: var(--space-md);
                margin-bottom: var(--space-md);
                box-shadow: var(--shadow-md);
                transition: all var(--transition-base);
                animation: fadeInUp 0.5s ease forwards;
                opacity: 0;
            }

            .card:nth-child(1) { animation-delay: 0.1s; }
            .card:nth-child(2) { animation-delay: 0.2s; }
            .card:nth-child(3) { animation-delay: 0.3s; }
            .card:nth-child(4) { animation-delay: 0.4s; }
            .card:nth-child(5) { animation-delay: 0.5s; }

            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: var(--space-xs);
                margin-bottom: var(--space-sm);
                padding-bottom: var(--space-sm);
                border-bottom: 1px solid var(--border-subtle);
            }

            .card-title {
                font-family: var(--font-display);
                font-size: 20px;
                font-weight: 600;
                color: var(--text-primary);
                margin: 0;
                display: flex;
                align-items: center;
                gap: var(--space-xs);
            }

            .card-icon {
                font-size: 20px;
            }

            .card-meta {
                font-size: 12px;
                color: var(--accent-gold);
                font-weight: 500;
            }

            /* ============================================
               STATS GRID - Mobile Stack, Desktop Row
               ============================================ */
            .stats-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: var(--space-sm);
                margin-bottom: var(--space-md);
            }

            .stat-card {
                background: var(--bg-card);
                border: 1px solid var(--border-subtle);
                border-radius: var(--border-radius-md);
                padding: var(--space-sm);
                text-align: center;
                transition: all var(--transition-base);
            }

            .stat-card:active {
                transform: scale(0.98);
            }

            .stat-icon {
                font-size: 28px;
                margin-bottom: var(--space-xs);
                display: block;
            }

            .stat-value {
                font-size: 14px;
                color: var(--text-secondary);
                line-height: 1.4;
            }

            /* ============================================
               PROGRESS BARS - Animated
               ============================================ */
            .progress-item {
                margin-bottom: var(--space-md);
            }

            .progress-item:last-child {
                margin-bottom: 0;
            }

            .progress-label {
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: var(--space-xs);
                font-size: 14px;
                color: var(--text-secondary);
                margin-bottom: var(--space-xs);
            }

            .progress-label strong {
                color: var(--text-primary);
                font-weight: 500;
            }

            .progress-track {
                height: 8px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 100px;
                overflow: hidden;
                position: relative;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--accent-gold-dark), var(--accent-gold), var(--accent-gold-light));
                border-radius: 100px;
                position: relative;
                animation: progressGrow 1.5s ease forwards;
                transform-origin: left;
            }

            @keyframes progressGrow {
                from { transform: scaleX(0); }
                to { transform: scaleX(1); }
            }

            .progress-fill::after {
                content: '';
                position: absolute;
                right: 0;
                top: 0;
                bottom: 0;
                width: 8px;
                background: radial-gradient(circle at center, rgba(255,255,255,0.4), transparent 60%);
                opacity: 0.8;
            }

            .progress-meta {
                margin-top: var(--space-xs);
                font-size: 12px;
                color: var(--text-secondary);
                display: flex;
                gap: var(--space-xs);
                align-items: center;
            }

            .highlight {
                color: var(--accent-gold);
                font-weight: 600;
            }

            /* ============================================
               QUOTE BLOCK
               ============================================ */
            .quote-card {
                position: relative;
                overflow: hidden;
            }

            .quote-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
            }

            blockquote {
                margin: 0;
                padding: var(--space-md);
                background: rgba(255, 255, 255, 0.02);
                border-radius: var(--border-radius-md);
                border: 1px solid var(--border-subtle);
                box-shadow: var(--shadow-sm);
            }

            .quote-icon {
                font-size: 24px;
                color: var(--accent-gold);
                display: inline-block;
                margin-bottom: var(--space-sm);
                text-shadow: 0 2px 10px var(--accent-glow);
            }

            .quote-text {
                font-size: 18px;
                line-height: 1.7;
                color: var(--text-primary);
                margin: 0 0 var(--space-sm) 0;
            }

            .quote-author {
                display: block;
                font-size: 13px;
                color: var(--text-secondary);
                letter-spacing: 0.5px;
            }

            /* ============================================
               DAILY WISDOM SECTION
               ============================================ */
            .wisdom-grid {
                display: grid;
                gap: var(--space-sm);
            }

            .wisdom-section {
                padding: var(--space-sm);
                border-radius: var(--border-radius-md);
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--border-subtle);
                box-shadow: var(--shadow-sm);
            }

            .wisdom-label {
                font-size: 11px;
                letter-spacing: 1px;
                color: var(--accent-gold);
                margin-bottom: var(--space-xs);
                display: inline-block;
                padding: 2px 10px;
                border-radius: 100px;
                background: rgba(212, 175, 55, 0.08);
                border: 1px solid rgba(212, 175, 55, 0.3);
            }

            .wisdom-text {
                color: var(--text-secondary);
                font-size: 15px;
                line-height: 1.7;
                margin: 0;
            }

            .wisdom-insight {
                background: linear-gradient(135deg, rgba(212, 175, 55, 0.06), rgba(10, 11, 13, 0.95));
                border-color: rgba(212, 175, 55, 0.2);
            }

            /* ============================================
               NEWS SECTION
               ============================================ */
            .news-grid {
                display: grid;
                gap: var(--space-sm);
            }

            .news-item {
                padding: var(--space-sm);
                border-radius: var(--border-radius-md);
                border: 1px solid var(--border-subtle);
                background: rgba(255, 255, 255, 0.02);
                transition: all var(--transition-fast);
            }

            .news-item:active {
                transform: translateY(1px);
            }

            .news-source {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 12px;
                color: var(--accent-gold);
                letter-spacing: 0.3px;
                text-transform: uppercase;
            }

            .news-headline {
                margin: var(--space-xs) 0;
                font-size: 16px;
                color: var(--text-primary);
                line-height: 1.5;
            }

            .news-headline a {
                color: var(--text-primary);
                text-decoration: none;
                transition: color var(--transition-fast);
            }

            .news-headline a:hover,
            .news-headline a:focus {
                color: var(--accent-gold);
            }

            .news-commentary {
                font-size: 14px;
                color: var(--text-secondary);
                line-height: 1.7;
                background: rgba(255, 255, 255, 0.02);
                padding: var(--space-sm);
                border-radius: var(--border-radius-sm);
                border-left: 3px solid var(--accent-gold);
                margin-top: var(--space-sm);
            }

            /* Bulletproof CTA Button - Compact size */
            .news-cta {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                height: 32px;
                padding: 0 var(--space-sm);
                margin-top: var(--space-xs);
                font-family: var(--font-body);
                font-size: 12px;
                font-weight: 500;
                letter-spacing: 0.3px;
                color: var(--bg-primary);
                background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-gold-light) 100%);
                border: none;
                border-radius: 6px;
                text-decoration: none;
                cursor: pointer;
                transition: all var(--transition-base);
                box-shadow: 0 2px 8px var(--accent-glow);
            }

            .news-cta:hover,
            .news-cta:focus {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px var(--accent-glow-strong);
            }

            .news-cta:active {
                transform: translateY(0);
            }

            /* ============================================
               FOOTER
               ============================================ */
            footer {
                text-align: center;
                padding: var(--space-lg) var(--space-sm);
                color: var(--text-muted);
                font-size: 12px;
                border-top: 1px solid var(--border-subtle);
                margin-top: var(--space-lg);
            }

            footer a {
                color: var(--accent-gold);
                text-decoration: none;
            }

            /* ============================================
               RESPONSIVE - Tablet & Desktop
               ============================================ */
            @media screen and (min-width: 480px) {
                .container {
                    padding: var(--space-md);
                }

                h1 {
                    font-size: 32px;
                }

                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }

                .card {
                    padding: var(--space-lg);
                }

                .quote-text {
                    font-size: 22px;
                }

                .wisdom-insight .wisdom-text {
                    font-size: 18px;
                }

                .news-headline {
                    font-size: 18px;
                }
            }

            @media screen and (min-width: 768px) {
                .container {
                    padding: var(--space-lg);
                }

                h1 {
                    font-size: 36px;
                }

                .header {
                    padding: var(--space-xl) 0 var(--space-lg);
                }

                .quote-text {
                    font-size: 24px;
                    padding: 0 var(--space-md);
                }

                .card {
                    border-radius: var(--border-radius-xl);
                }
            }

            /* ============================================
               DARK MODE SUPPORT (System Preference)
               ============================================ */
            @media (prefers-color-scheme: light) {
                :root {
                    --bg-primary: #FAFAFA;
                    --bg-secondary: #F5F5F5;
                    --bg-card: rgba(255, 255, 255, 0.9);
                    --bg-card-hover: rgba(255, 255, 255, 0.95);
                    --border-subtle: rgba(0, 0, 0, 0.06);
                    --border-medium: rgba(0, 0, 0, 0.1);
                    --text-primary: #1A1A1A;
                    --text-secondary: #4A4A4A;
                    --text-muted: #8A8A8A;
                    --accent-gold: #B8962E;
                    --accent-gold-light: #D4AF37;
                    --accent-gold-dark: #8B7023;
                    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
                    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
                    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
                }

                body {
                    background-image:
                        radial-gradient(ellipse at top, rgba(184, 150, 46, 0.05) 0%, transparent 50%),
                        linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                }

                .news-cta {
                    color: #FFFFFF;
                }
            }

            /* ============================================
               MANUAL THEME OVERRIDE (data-theme)
               data-theme always wins over system preference.
               This enables a browser toggle, while remaining safe for email.
               ============================================ */
            :root[data-theme="dark"] {
                --bg-primary: #0A0B0D;
                --bg-secondary: #12141A;
                --bg-card: rgba(22, 25, 32, 0.85);
                --bg-card-hover: rgba(28, 32, 42, 0.9);
                --border-subtle: rgba(255, 255, 255, 0.06);
                --border-medium: rgba(255, 255, 255, 0.1);
                --text-primary: #F5F5F7;
                --text-secondary: #A1A1AA;
                --text-muted: #6B7280;
                --accent-gold: #D4AF37;
                --accent-gold-light: #E8C959;
                --accent-gold-dark: #B8962E;
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.15);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.2);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.25);
            }

            :root[data-theme="dark"] body {
                background-image:
                    radial-gradient(ellipse at top, rgba(212, 175, 55, 0.03) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            }

            :root[data-theme="light"] {
                --bg-primary: #FAFAFA;
                --bg-secondary: #F5F5F5;
                --bg-card: rgba(255, 255, 255, 0.9);
                --bg-card-hover: rgba(255, 255, 255, 0.95);
                --border-subtle: rgba(0, 0, 0, 0.06);
                --border-medium: rgba(0, 0, 0, 0.1);
                --text-primary: #1A1A1A;
                --text-secondary: #4A4A4A;
                --text-muted: #8A8A8A;
                --accent-gold: #B8962E;
                --accent-gold-light: #D4AF37;
                --accent-gold-dark: #8B7023;
                --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.08);
                --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.1);
                --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.12);
            }

            :root[data-theme="light"] body {
                background-image:
                    radial-gradient(ellipse at top, rgba(184, 150, 46, 0.05) 0%, transparent 50%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            }

            :root[data-theme="light"] .news-cta {
                color: #FFFFFF;
            }

            /* ============================================
               ACCESSIBILITY & FOCUS STATES
               ============================================ */
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
            }

            a:focus-visible,
            button:focus-visible,
            .news-cta:focus-visible {
                outline: 2px solid var(--accent-gold);
                outline-offset: 2px;
            }

            @media (prefers-contrast: high) {
                :root {
                    --border-subtle: rgba(255, 255, 255, 0.2);
                    --border-medium: rgba(255, 255, 255, 0.3);
                }
            }

            .lesson-content {
                font-size: 15px;
                line-height: 1.7;
                color: var(--text-secondary);
            }

            .lesson-paragraph {
                margin-bottom: var(--space-sm);
            }

            .historical-note {
                border-left: 3px solid var(--accent-gold);
                padding: var(--space-sm);
                margin: var(--space-sm) 0;
                color: var(--text-primary);
                background: var(--accent-glow);
                border-radius: 0 var(--border-radius-sm) var(--border-radius-sm) 0;
            }
        </style>
    """


def get_theme_toggle_script():
    """
    Returns a tiny, dependency-free theme toggle script.
    """
    return """
        <script>
            (function () {
                var STORAGE_KEY = "edith_newsletter_theme";
                var root = document.documentElement;
                var btn = document.getElementById("theme-toggle");
                if (!root || !btn) return;

                function systemTheme() {
                    try {
                        return window.matchMedia &&
                            window.matchMedia("(prefers-color-scheme: light)").matches
                            ? "light"
                            : "dark";
                    } catch (e) {
                        return "dark";
                    }
                }

                function currentTheme() {
                    return root.getAttribute("data-theme") || systemTheme();
                }

                function updateButton(theme) {
                    var isDark = theme === "dark";
                    btn.textContent = isDark ? "☾" : "☀";
                    btn.setAttribute("aria-pressed", isDark ? "true" : "false");
                    var title = isDark ? "Switch to light mode" : "Switch to dark mode";
                    btn.setAttribute("title", title);
                    btn.setAttribute("aria-label", title);
                }

                function setTheme(theme, persist) {
                    if (theme !== "light" && theme !== "dark") return;
                    root.setAttribute("data-theme", theme);
                    updateButton(theme);
                    if (persist) {
                        try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) {}
                    }
                }

                function clearThemeOverride() {
                    root.removeAttribute("data-theme");
                    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
                    updateButton(systemTheme());
                }

                var stored = null;
                try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) {}

                if (stored === "light" || stored === "dark") {
                    setTheme(stored, false);
                } else {
                    updateButton(systemTheme());
                }

                btn.addEventListener("click", function (e) {
                    if (e && (e.shiftKey || e.altKey)) {
                        clearThemeOverride();
                        return;
                    }
                    var next = currentTheme() === "dark" ? "light" : "dark";
                    setTheme(next, true);
                });
            })();
        </script>
    """


def generate_html_progress_message(
    days_completed: int,
    weeks_completed: float,
    days_left: int,
    weeks_left: float,
    percent_days_left: float,
) -> str:
    temp, status = get_weather()
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y")
    current_year = now.year

    earnings_dates = [
        datetime(current_year, 1, 23),
        datetime(current_year, 4, 25),
        datetime(current_year, 7, 29),
        datetime(current_year, 10, 24),
        datetime(current_year + 1, 1, 23),
    ]

    current_quarter = None
    for i in range(len(earnings_dates) - 1):
        if earnings_dates[i] <= now < earnings_dates[i + 1]:
            current_quarter = i + 1
            start_of_quarter = earnings_dates[i]
            end_of_quarter = earnings_dates[i + 1]
            break

    if current_quarter is None:
        current_quarter = 4
        start_of_quarter = earnings_dates[3]
        end_of_quarter = earnings_dates[4]

    days_in_quarter = (end_of_quarter - start_of_quarter).days
    days_completed_in_quarter = (now - start_of_quarter).days
    if days_in_quarter == 0:
        days_in_quarter = 1
    percent_days_left_in_quarter = (
        (days_in_quarter - days_completed_in_quarter) / days_in_quarter
    ) * 100
    q_progress = 100 - percent_days_left_in_quarter
    year_progress = 100 - percent_days_left

    total_days_in_year = (
        366
        if (current_year % 4 == 0 and current_year % 100 != 0)
        or (current_year % 400 == 0)
        else 365
    )

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Year Progress Report</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Year Progress</h1>
                <p class="subtitle">Your daily temporal briefing</p>
                <div class="date-badge">{date_time}</div>
            </header>

            <section class="stats-grid" aria-label="Quick statistics">
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">🌤️</span>
                    <div class="stat-value">
                        <strong>{temp}°C</strong><br>
                        {status}
                    </div>
                </div>
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">📅</span>
                    <div class="stat-value">
                        <strong>Day {days_completed}</strong><br>
                        of {total_days_in_year}
                    </div>
                </div>
            </section>

            <article class="card" aria-label="Year and quarter progress">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">⏱️</span>
                        Temporal Status
                    </h2>
                    <span class="card-meta">{year_progress:.1f}% Complete</span>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Year {current_year}</strong>
                        <span>{days_completed} / {total_days_in_year} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{year_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {year_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{weeks_left:.1f}</span> weeks remaining
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Q{current_quarter}</strong>
                        <span>{days_completed_in_quarter} / {days_in_quarter} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{q_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {q_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{days_in_quarter - days_completed_in_quarter}</span> days until Q{current_quarter + 1 if current_quarter < 4 else 1}
                    </div>
                </div>
            </article>

            <article class="card quote-card" id="quote-card" aria-label="Quote of the day">
                <div class="quote-icon" aria-hidden="true">❝</div>
                <blockquote>
                    <p class="quote-text" id="quote-text"></p>
                    <cite class="quote-author" id="quote-author"></cite>
                </blockquote>
            </article>

            <article class="card" id="lesson-card" aria-label="Daily wisdom">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💡</span>
                        Daily Wisdom
                    </h2>
                </div>
                <div class="lesson-content" id="lesson-content"></div>
            </article>

            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template


def generate_html_news_template(news_content: str) -> str:
    """Generate premium mobile-first HTML newsletter for daily news briefing."""
    now = datetime.now()
    date_formatted = now.strftime("%B %d, %Y")
    day_of_week = now.strftime("%A")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Daily Briefing - {date_formatted}</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Daily Briefing</h1>
                <p class="subtitle">{day_of_week}'s Essential Updates</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <article class="card" aria-label="Technology news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💻</span>
                        Technology
                    </h2>
                    <span class="card-meta">Tech & AI</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "Tech News Update")}
                </div>
            </article>

            <article class="card" aria-label="Financial markets news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">📈</span>
                        Markets
                    </h2>
                    <span class="card-meta">Finance</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "Financial Markets News Update")}
                </div>
            </article>

            <article class="card" aria-label="India news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true" style="display: inline-flex; align-items: center;">
                            <svg width="20" height="14" viewBox="0 0 20 14" style="border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <rect width="20" height="4.67" fill="#FF9933"/>
                                <rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/>
                                <rect y="9.33" width="20" height="4.67" fill="#138808"/>
                                <circle cx="10" cy="7" r="1.8" fill="#000080"/>
                            </svg>
                        </span>
                        India
                    </h2>
                    <span class="card-meta">Regional</span>
                </div>
                <div class="card-content">
                    {format_news_section(news_content, "India News Update")}
                </div>
            </article>

            <footer>
                <p>Generated by <strong>EDITH</strong> • {now.year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template


def format_news_section(content: str, section_title: str) -> str:
    """
    Format news section for HTML display using the premium mobile-first design.
    """
    import html as html_module

    sections = content.split("##")
    section_content = ""

    for section in sections:
        if section_title.lower() in section.lower():
            section_content = section
            break

    formatted_items = []

    if section_content:
        lines = [line.strip() for line in section_content.split("\n") if line.strip()]

        points = []
        for line in lines:
            if line.startswith("- "):
                points.append(line[2:])
            elif not line.lower().endswith(("update:", "update")):
                if len(line) > 20 and not line.startswith("#"):
                    points.append(line)

        for point in points[:5]:
            news_text = point
            url = ""
            news_date = ""
            citation = ""
            commentary = ""

            if " | " in point:
                components = point.split(" | ")
                news_text = components[0]

                for component in components[1:]:
                    component = component.strip()
                    if component.startswith("[") and component.endswith("]"):
                        citation = component.strip("[]")
                    elif component.startswith("http"):
                        url = component
                    elif component.startswith("Date:"):
                        news_date = component.replace("Date:", "").strip()
                    elif component.startswith("Commentary:"):
                        commentary = component.replace("Commentary:", "").strip()

            if news_text.startswith("[") and "]" in news_text:
                end_bracket = news_text.index("]")
                citation = news_text[1:end_bracket]
                news_text = news_text[end_bracket + 1 :].strip()

            safe_news_text = html_module.escape(news_text) if news_text else ""
            safe_citation = html_module.escape(citation) if citation else "News Update"
            safe_commentary = html_module.escape(commentary) if commentary else ""

            if url:
                headline_html = f'''
                    <h3 class="news-headline">
                        <a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer">
                            {safe_news_text}
                        </a>
                    </h3>'''
            else:
                headline_html = f'<h3 class="news-headline">{safe_news_text}</h3>'

            date_html = ""
            if news_date:
                date_html = f'<time class="news-date" style="display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{html_module.escape(news_date)}</time>'

            commentary_html = ""
            if safe_commentary:
                commentary_html = f'<div class="news-commentary">{safe_commentary}</div>'

            cta_html = ""
            if url:
                cta_html = f'''
                    <a href="{html_module.escape(url)}"
                       target="_blank"
                       rel="noopener noreferrer"
                       class="news-cta"
                       aria-label="Read full article about {safe_news_text[:50]}...">
                        Read More →
                    </a>'''

            item_html = f'''
                <article class="news-item">
                    <span class="news-source">{safe_citation}</span>
                    {date_html}
                    {headline_html}
                    {commentary_html}
                    {cta_html}
                </article>
            '''
            formatted_items.append(item_html)

    if not formatted_items:
        return '''
            <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
                <p style="margin: 0;">No updates available for this section.</p>
                <p style="margin-top: 8px; font-size: 12px;">Check back later for the latest news.</p>
            </div>
        '''

    return "\n".join(formatted_items)


def format_news_items_html(items: List[Dict[str, Any]]) -> str:
    """
    Format a list of structured news items to HTML.
    Each item should have: source, headline, date_mmddyyyy, url (optional), commentary
    """
    import html as html_module

    if not items:
        return '''
            <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
                <p style="margin: 0;">No updates available for this section.</p>
                <p style="margin-top: 8px; font-size: 12px;">Check back later for the latest news.</p>
            </div>
        '''

    formatted_items = []

    for item in items[:5]:
        source = html_module.escape(item.get("source", "Update"))
        headline = html_module.escape(item.get("headline", ""))
        date_str = html_module.escape(item.get("date_mmddyyyy", ""))
        url = item.get("url", "")
        commentary = html_module.escape(item.get("commentary", ""))

        if not headline:
            continue

        if url:
            headline_html = f'''
                <h3 class="news-headline">
                    <a href="{html_module.escape(url)}" target="_blank" rel="noopener noreferrer">
                        {headline}
                    </a>
                </h3>'''
        else:
            headline_html = f'<h3 class="news-headline">{headline}</h3>'

        date_html = ""
        if date_str:
            date_html = f'<time class="news-date" style="display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{date_str}</time>'

        commentary_html = ""
        if commentary:
            commentary_html = f'<div class="news-commentary">{commentary}</div>'

        cta_html = ""
        if url:
            cta_html = f'''
                <a href="{html_module.escape(url)}"
                   target="_blank"
                   rel="noopener noreferrer"
                   class="news-cta"
                   aria-label="Read full article about {headline[:50]}...">
                    Read More →
                </a>'''

        item_html = f'''
            <article class="news-item">
                <span class="news-source">{source}</span>
                {date_html}
                {headline_html}
                {commentary_html}
                {cta_html}
            </article>
        '''
        formatted_items.append(item_html)

    return "\n".join(formatted_items) if formatted_items else '''
        <div style="padding: var(--space-md); text-align: center; color: var(--text-muted);">
            <p style="margin: 0;">No updates available for this section.</p>
        </div>
    '''


def render_lesson_html(lesson: Dict[str, Any]) -> str:
    """Render lesson content from structured dict to HTML."""
    import html as html_module

    html_parts = ['<div class="wisdom-grid">']

    key_insight = lesson.get("key_insight", "")
    if key_insight:
        safe_insight = html_module.escape(key_insight)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-insight">
            <div class="wisdom-label">KEY INSIGHT</div>
            <div class="wisdom-text">{safe_insight}</div>
        </div>''')

    historical = lesson.get("historical", "")
    if historical:
        safe_historical = html_module.escape(historical)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-historical">
            <div class="wisdom-label">HISTORICAL CONTEXT</div>
            <div class="wisdom-text">{safe_historical}</div>
        </div>''')

    application = lesson.get("application", "")
    if application:
        safe_application = html_module.escape(application)
        html_parts.append(f'''
        <div class="wisdom-section wisdom-application">
            <div class="wisdom-label">APPLICATION</div>
            <div class="wisdom-text">{safe_application}</div>
        </div>''')

    html_parts.append("</div>")
    return "\n".join(html_parts)


def render_year_progress_html_from_bundle(bundle: Dict[str, Any]) -> str:
    """Render the year progress HTML from bundle data."""
    import html as html_module

    meta = bundle["meta"]
    progress = bundle["progress"]
    time_data = progress["time"]
    quarter_data = progress["quarter"]
    weather = progress["weather"]
    quote = progress["quote"]
    lesson = progress["lesson"]

    date_formatted = meta["date_formatted"]
    current_year = time_data["year"]
    days_completed = time_data["days_completed"]
    total_days_in_year = time_data["total_days_in_year"]
    year_progress = time_data["percent_complete"]
    weeks_left = time_data["weeks_left"]

    current_quarter = quarter_data["current_quarter"]
    days_in_quarter = quarter_data["days_in_quarter"]
    days_completed_in_quarter = quarter_data["days_completed_in_quarter"]
    days_left_in_quarter = quarter_data["days_left_in_quarter"]
    q_progress = quarter_data["percent_complete"]

    temp = weather.get("temp_c", "N/A")
    status = html_module.escape(weather.get("status", "Unknown"))

    quote_text = html_module.escape(quote.get("text", ""))
    quote_author = html_module.escape(quote.get("author", ""))

    lesson_html = render_lesson_html(lesson)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Year Progress Report</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Year Progress</h1>
                <p class="subtitle">Your daily temporal briefing</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <section class="stats-grid" aria-label="Quick statistics">
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">🌤️</span>
                    <div class="stat-value">
                        <strong>{temp}°C</strong><br>
                        {status}
                    </div>
                </div>
                <div class="stat-card">
                    <span class="stat-icon" aria-hidden="true">📅</span>
                    <div class="stat-value">
                        <strong>Day {days_completed}</strong><br>
                        of {total_days_in_year}
                    </div>
                </div>
            </section>

            <article class="card" aria-label="Year and quarter progress">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">⏱️</span>
                        Temporal Status
                    </h2>
                    <span class="card-meta">{year_progress:.1f}% Complete</span>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Year {current_year}</strong>
                        <span>{days_completed} / {total_days_in_year} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{year_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {year_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{weeks_left:.1f}</span> weeks remaining
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-label">
                        <strong>Q{current_quarter}</strong>
                        <span>{days_completed_in_quarter} / {days_in_quarter} Days</span>
                    </div>
                    <div class="progress-track" role="progressbar" aria-valuenow="{q_progress:.0f}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-fill" style="width: {q_progress}%"></div>
                    </div>
                    <div class="progress-meta">
                        <span class="highlight">{days_left_in_quarter}</span> days until Q{current_quarter + 1 if current_quarter < 4 else 1}
                    </div>
                </div>
            </article>

            <article class="card quote-card" aria-label="Quote of the day">
                <div class="quote-icon" aria-hidden="true">❝</div>
                <blockquote>
                    <p class="quote-text">{quote_text}</p>
                    <cite class="quote-author">— {quote_author}</cite>
                </blockquote>
            </article>

            <article class="card" aria-label="Daily wisdom">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💡</span>
                        Daily Wisdom
                    </h2>
                </div>
                <div class="lesson-content">{lesson_html}</div>
            </article>

            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template


def render_newsletter_html_from_bundle(bundle: Dict[str, Any]) -> str:
    """
    Render the newsletter HTML from bundle data.
    Uses structured news sections directly - no text parsing.
    """
    meta = bundle["meta"]
    news = bundle["news"]
    newsletter = news["newsletter"]
    sections = newsletter["sections"]

    date_formatted = meta["date_formatted"]
    day_of_week = meta["day_of_week"]
    current_year = datetime.now().year

    tech_html = format_news_items_html(sections.get("tech", []))
    financial_html = format_news_items_html(sections.get("financial", []))
    india_html = format_news_items_html(sections.get("india", []))

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0">
        <meta name="theme-color" content="#0A0B0D">
        <meta name="color-scheme" content="dark light">
        <title>Daily Briefing - {date_formatted}</title>
        {get_luxurious_css()}
    </head>
    <body>
        <div class="container">
            <header class="header">
                <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle theme" aria-pressed="false" title="Toggle theme">☾</button>
                <h1>Daily Briefing</h1>
                <p class="subtitle">{day_of_week}'s Essential Updates</p>
                <div class="date-badge">{date_formatted}</div>
            </header>

            <article class="card" aria-label="Technology news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">💻</span>
                        Technology
                    </h2>
                    <span class="card-meta">Tech & AI</span>
                </div>
                <div class="card-content">
                    {tech_html}
                </div>
            </article>

            <article class="card" aria-label="Financial markets news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true">📈</span>
                        Markets
                    </h2>
                    <span class="card-meta">Finance</span>
                </div>
                <div class="card-content">
                    {financial_html}
                </div>
            </article>

            <article class="card" aria-label="India news">
                <div class="card-header">
                    <h2 class="card-title">
                        <span class="card-icon" aria-hidden="true" style="display: inline-flex; align-items: center;">
                            <svg width="20" height="14" viewBox="0 0 20 14" style="border-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                <rect width="20" height="4.67" fill="#FF9933"/>
                                <rect y="4.67" width="20" height="4.67" fill="#FFFFFF"/>
                                <rect y="9.33" width="20" height="4.67" fill="#138808"/>
                                <circle cx="10" cy="7" r="1.8" fill="#000080"/>
                            </svg>
                        </span>
                        India
                    </h2>
                    <span class="card-meta">Regional</span>
                </div>
                <div class="card-content">
                    {india_html}
                </div>
            </article>

            <footer>
                <p>Generated by <strong>EDITH</strong> • {current_year}</p>
                <p style="margin-top: 8px; font-size: 11px;">Even Dead, I'm The Hero</p>
            </footer>
        </div>
        {get_theme_toggle_script()}
    </body>
    </html>
    """
    return html_template
