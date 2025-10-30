from dotenv import load_dotenv
from brightdata import bdclient
from openai import OpenAI
from sendgrid import SendGridAPIClient
from pydantic import BaseModel, Field
from typing import List
import json
from sendgrid.helpers.mail import Mail

# Load environment variables from the .env file
load_dotenv()

# Initialize the Bright Data SDK client
brightdata_client = bdclient()
# Initialize the OpenAI SDK client
openai_client = OpenAI()
# Initialize the SendGrid SDK client
sendgrid_client = SendGridAPIClient()

# Pydantic models
class Config(BaseModel):
    search_queries: List[str] = Field(..., min_items=1)
    num_news: int = Field(..., gt=0)
    sender: str = Field(..., min_length=1)
    recipients: List[str] = Field(..., min_items=1)

class URLList(BaseModel):
    urls: List[str]

class NewsAnalysis(BaseModel):
    title: str
    url: str
    summary: str
    sentiment_analysis: str
    insights: List[str]

def get_google_news_page_urls(search_queries):
    # Retrieve SERPs for the given search queries
    serp_results = brightdata_client.search(
        search_queries,
        search_engine="google",
        parse=True, # To get the SERP result as a parsed JSON string
    )

    news_page_urls = []
    for serp_result in serp_results:
        # Loading the JSON string to a dictionary
        serp_data = json.loads(serp_result)
        # Extract the Google News URL from each parsed SERP
        if serp_data.get("navigation"):
            for item in serp_data["navigation"]:
                if item["title"] == "News":
                    news_url = item["href"]
                    news_page_urls.append(news_url)

    return news_page_urls

def scrape_news_pages(news_page_urls):
    # Scrape each news page in parallel and return their content in Markdown
    return brightdata_client.scrape(
        url=news_page_urls,
        data_format="markdown",
    )

def get_best_news_urls(news_pages, num_news):
    # Use GPT to extract the most relevant news URLs
    response = openai_client.responses.parse(
        model="gpt-5-mini",
        input=[
            {
                "role": "system",
                "content": f"Extract the {num_news} most relevant news for brand reputation monitoring from the text and return them as a list of URL strings.",
            },
            {
                "role": "user",
                "content": "\n\n---------------\n\n".join(news_pages),
            },
        ],
        text_format=URLList,
    )

    return response.output_parsed.urls

def scrape_news_articles(news_urls):
    # Scrape each news URL and return a list of dicts with URL and content
    news_content_list = brightdata_client.scrape(
        url=news_urls,
        data_format="markdown",
    )

    news_list = []
    for url, content in zip(news_urls, news_content_list):
        news_list.append({
            "url": url,
            "content": content,
        })

    return news_list

def process_news_list(news_list):
    # Where to store the analyzed news articles
    news_analysis_list = []

    # Analyze each news article with GPT for brand reputation monitoring insights
    for news in news_list:
        response = openai_client.responses.parse(
            model="gpt-5-mini",
            input=[
                {
                    "role": "system",
                    "content": f"""
                        Given the news content:
                        1. Extract the title.
                        2. Extract the URL.
                        3. Write a summary in no more than 30 words.
                        4. Extract the sentiment of the news as one of the following: "positive", "negative", or "neutral".
                        5. Extract the top 3 to 5 actionable, short insights (no more than 10/12 words) about brand reputation from the news, presenting them in clear, concise, straightforward language.
                    """
                },
                {
                    "role": "user",
                    "content": f"NEWS URL: {news["url"]}\n\nNEWS CONTENT:{news["content"]}"
                },
            ],
            text_format=NewsAnalysis,
        )

        # Get the output analyzed news object and append it to the list
        news_analysis = response.output_parsed
        news_analysis_list.append(news_analysis)

    return news_analysis_list

def create_html_email_body(news_analysis_list):
    # Generate a structured HTML email body from analyzed news
    response = openai_client.responses.create(
        model="gpt-5-mini",
        input=f"""
        Given the content below, generate a structured HTML email body that is well-formatted, responsive, and ready to send.
        Ensure proper use of headings, paragraphs, colored labels, and links where appropriate.
        Do not include a header or footer section, and include only this information—nothing else.

        CONTENT:
        {[json.dumps([item.model_dump() for item in news_analysis_list], indent=2)]}
        """
    )
    return response.output_text

def send_email(sender, recipients, html_body):
    # Send the HTML email using SendGrid
    message = Mail(
        from_email=sender,
        to_emails=recipients,
        subject="Brand Monitoring Weekly Report",
        html_content=html_body
    )
    sendgrid_client.send(message)

def main():
    # Read the config file and validate it
    with open("config.json", "r", encoding="utf-8") as f:
        raw_config = json.load(f)
        config = Config.model_validate(raw_config)

    search_queries = config.search_queries
    print(f"Retrieving Google News page URLs for the following search queries: {", ".join(search_queries)}")
    google_news_page_urls = get_google_news_page_urls(search_queries)
    print(f"{len(google_news_page_urls)} Google News page URL(s) retrieved!\n")

    print("Scraping content from each Google News page...")
    scraped_news_pages = scrape_news_pages(google_news_page_urls)
    print("Google News pages scraped!\n")

    print("Extracting the most relevant news URLs...")
    news_urls = get_best_news_urls(scraped_news_pages, config.num_news)
    print(f"{len(news_urls)} news articles found:\n" + "\n".join(f"- {news}" for news in news_urls) + "\n")

    print("Scraping the selected news articles...")
    news_list = scrape_news_articles(news_urls)
    print(f"{len(news_urls)} news articles scraped!")

    print("Analyzing each news for brand reputation monitoring...")
    news_analysis_list = process_news_list(news_list)
    print("News analysis complete!\n")

    print("Generating HTML email body...")
    html = create_html_email_body(news_analysis_list)
    print("HTML email body generated!\n")

    print("Sending the email with the brand reputation monitoring HTML report...")
    send_email(config.sender, config.recipients, html)
    print("Email sent!")

# Run the main function
if __name__ == "__main__":
    main()
