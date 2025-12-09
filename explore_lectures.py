import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import plotly.express as px
import time
import urllib.parse  
from transformers import pipeline
import torch
from typing import List, Dict, Tuple  # noqa: F401


def show_page():
    
    st.set_page_config(page_title=" Explore your Favourite Lectures", layout="wide")
    # -------------------------
    # Helper functions
    # -------------------------
    def create_chrome_driver(headless: bool = True, timeout: int = 90) -> webdriver.Chrome:
        """Create a Chrome webdriver instance with sensible options."""
        options = webdriver.ChromeOptions()
        # recommended flags
        if headless:
            # newer chrome may accept "--headless=new"
            try:
                options.add_argument("--headless=new")
            except Exception:
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument(
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                )
        # avoid logging messages
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        svc = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=svc, options=options)
        driver.set_page_load_timeout(timeout)
        driver.set_script_timeout(90)
        return driver


    def get_top_video_links(driver: webdriver.Chrome, query: str, max_videos: int = 20) -> List[Tuple[str, str]]:
        """
        Search YouTube and return a list of tuples (video_title, video_url) for top results.
        Uses: https://www.youtube.com/results?search_query=...
        """
        base = "https://www.youtube.com/results?search_query="
        q = urllib.parse.quote_plus(query)
        url = base + q
        driver.get(url)
        time.sleep(3)  

        scroll_tries = 3
        for _ in range(scroll_tries):
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(1)

        # collect video anchors
        links = []
        # prefer anchors with id 'video-title'
        anchors = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
        for a in anchors:
            href = a.get_attribute("href")
            title = a.get_attribute("title") or a.text
            if href and "watch" in href and title:
                links.append((title.strip(), href.strip()))
            if len(links) >= max_videos:
                break

        # fallback: sometimes results appear under ytd-video-renderer
        if len(links) < max_videos:
            items = driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer a#video-title")
            for a in items:
                href = a.get_attribute("href")
                title = a.get_attribute("title") or a.text
                if href and "watch" in href and title:
                    tup = (title.strip(), href.strip())
                    if tup not in links:
                        links.append(tup)
                if len(links) >= max_videos:
                    break

        # dedupe and limit
        seen = set()
        final = []
        for t, h in links:
            if h not in seen:
                final.append((t, h))
                seen.add(h)
            if len(final) >= max_videos:
                break

        return final
    def get_youtube_comments_for_video(driver: webdriver.Chrome, video_url: str, max_comments: int = 20, scroll_pause: float = 2.0) -> List[str]:
        """
        Open a video page and scrape comments text up to max_comments.
        """
        driver.get(video_url)
        time.sleep(4) 
        
        try:
            driver.execute_script("window.scrollTo(0, 600);")
            time.sleep(2)
        except Exception:
            pass

        # attempt to scroll until we have enough comments or no change
        last_count = 0
        same_count_tries = 0
        max_same_tries = 6
        total_scrolls = 0
        while True:
            comment_els = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
            if len(comment_els) >= max_comments:
                break
            driver.execute_script("window.scrollBy(0, 1000);")
            time.sleep(scroll_pause)
            new_count = len(driver.find_elements(By.XPATH, '//*[@id="content-text"]'))
            total_scrolls += 1
            if new_count == last_count:
                same_count_tries += 1
            else:
                same_count_tries = 0
            last_count = new_count
            # break conditions to avoid infinite loops
            if same_count_tries >= max_same_tries or total_scrolls > 30:
                break

        comment_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
        comments = [el.text for el in comment_elements[:max_comments] if el.text.strip()]
        return comments
    @st.cache_resource(show_spinner=False)
    def load_sentiment_pipeline():
        """Load transformers pipeline and cache it (so model is downloaded only once)."""
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
        return pipe

    def analyze_comments_sentiment(pipe, comments: List[str], batch_size: int = 16) -> pd.DataFrame:
        """Run sentiment pipeline on list of comments and return DataFrame with Label and Score."""
        if not comments:
            return pd.DataFrame(columns=["Comment", "Label", "Score"])

        # truncate overly long comments to 512 tokens (approx. ~1024 characters for safety)
        truncated = [c if len(c) < 1000 else c[:1000] for c in comments]
        results = pipe(truncated, batch_size=batch_size, truncation=True)
        df = pd.DataFrame({
            "Comment": truncated,
            "Label": [r.get("label", "") for r in results],
            "Score": [r.get("score", 0.0) for r in results]
        })
        return df


    def compute_positive_rate(df_sent: pd.DataFrame) -> float:
        """Return positive rate between 0 and 1 given sentiment df with 'Label' column."""
        if df_sent.empty:
            return 0.0
        positives = (df_sent["Label"] == "POSITIVE").sum()
        total = len(df_sent)
        return float(positives) / float(total)
    
    # -------------------------
    # Streamlit UI
    # -------------------------
    st.title("📚 Explore Your Favourite Lectures")
    st.markdown(
        "Enter a search keyword. The app will search YouTube, scrape top 20 videos, "
        "fetch up to 20 comments per video, run sentiment analysis and rank videos by positive-rate."
    )

    st.header("Settings")
    max_videos = st.number_input("How many top videos to consider (max 20)", min_value=11, max_value=20, value=20, step=1)
    max_comments_per_video = st.number_input("How many comments per video (max 50)", min_value=10, max_value=50, value=20, step=1)
    headless = st.checkbox("Run Chrome headless", value=True)
    show_comments = st.checkbox("Show scraped comments for selected video", value=True)
    query = st.text_input("Search keyword (e.g. Machine Learning)", value="Machine Learning")
    run_button = st.button("Run Search & Analyze")

    if run_button and query.strip():
        placeholder = st.empty()
        status = placeholder.container()
        status.info("**Starting...**")
        progress = st.progress(0)

        try:
            # create driver once for the entire run
            status.info("Launching Chrome driver...")
            driver = create_chrome_driver(headless=headless)
            time.sleep(1)
            status.info("Searching YouTube...")
            vids = get_top_video_links(driver, query, max_videos)
            status.info(f"Found {len(vids)} videos. Beginning comment scraping & sentiment analysis...")
            results = []  # will store dicts: title, url, positive_rate, comments_df

            # load model once
            status.info("Loading sentiment model (may take a moment first run)...")
            pipe = load_sentiment_pipeline()

            total_videos = len(vids)
            for i, (title, url) in enumerate(vids, start=1):
                status.markdown(f"Processing video {i}/{total_videos}: **{title}**")
                progress.progress(int((i-1)/total_videos * 100))

                # get comments
                try:
                    comments = get_youtube_comments_for_video(driver, url, max_comments=max_comments_per_video)
                except Exception as e:
                    st.warning(f"Failed to scrape comments for {title[:60]}... continuing. Error: {e}")
                    comments = []

                # sentiment
                df_sent = analyze_comments_sentiment(pipe, comments)
                pos_rate = compute_positive_rate(df_sent)

                results.append({
                    "title": title,
                    "url": url,
                    "positive_rate": pos_rate,
                    "num_comments": len(df_sent),
                    "comments_df": df_sent
                })

            progress.progress(100)
            status.markdown("All videos processed. Sorting results...")

            # build results table
            summary = pd.DataFrame([{
                "Title": r["title"],
                "URL": r["url"],
                "PositiveRate": r["positive_rate"],
                "NumComments": r["num_comments"]
            } for r in results])

            # sort by positive rate descending
            summary_sorted = summary.sort_values("PositiveRate", ascending=False).reset_index(drop=True)
            top_n = min(10, len(summary_sorted))
            top10 = summary_sorted.head(top_n)

            # Display results
            st.subheader(f"Top {top_n} videos by Positive Rate")
            # show as clickable links in DataFrame: construct markdown
            def make_link(x):
                return f"[Open]({x})"

            display_df = top10.copy()
            display_df["Open"] = display_df["URL"].apply(make_link)
            display_df["PositiveRate"] = (display_df["PositiveRate"] * 100).round(1).astype(str) + " %"
            display_df = display_df[["Title", "PositiveRate", "NumComments", "Open"]]

            st.table(display_df)

            # Let user pick one of top videos to inspect
            choice = st.selectbox("Inspect one of the top videos", options=top10["URL"].tolist(), format_func=lambda u: next((r["title"] for r in results if r["url"]==u), u))
            if choice:
                sel = next((r for r in results if r["url"] == choice), None)
                if sel is not None:
                    st.markdown(f"### {sel['title']}")
                    st.markdown(f"[Open on YouTube]({sel['url']})")
                    st.metric("Positive Rate", f"{sel['positive_rate']*100:.1f}%")
                    st.write(f"Comments analyzed: {sel['num_comments']}")
                    if show_comments:
                        df_display = sel["comments_df"].copy()
                        if not df_display.empty:
                            st.dataframe(df_display)
                            # plot distribution
                            counts = df_display['Label'].value_counts().reindex(["POSITIVE", "NEGATIVE"]).fillna(0)
                            fig = px.pie(names=counts.index, values=counts.values, title="Sentiment Distribution (selected video)")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No comments to display for this video.")
            # download CSV of top10 summary
            csv = top10.to_csv(index=False)
            st.download_button("Download top list as CSV", data=csv, file_name="top_videos_by_positive_rate.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
            placeholder.empty()

    else:
        st.info("Enter a keyword and click **Run Search & Analyze** to begin.")

    # Footer / tips
    st.markdown("---")
    st.markdown(
        "**Tips:**\n"
        "- Run the app on a machine with enough RAM. The sentiment model downloads on first run.\n"
        "- If Chrome can't start in your environment, consider using a server with Chrome installed or adjust ChromeOptions.\n"
        "- If YouTube structure changes, update CSS/XPath selectors for video links and comments."
    )