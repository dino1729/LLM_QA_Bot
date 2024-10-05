import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from openai import AzureOpenAI as OpenAIAzure
from pyowm import OWM
from config import config
from helper_functions.chat_generation_with_internet import internet_connected_chatbot
from helper_functions.audio_processors import text_to_speech

azure_api_key = config.azure_api_key
azure_api_base = config.azure_api_base
azure_chatapi_version = config.azure_chatapi_version
azure_chatapi_version = config.azure_chatapi_version
azure_gpt4omini_deploymentid = config.azure_gpt4omini_deploymentid

yahoo_id = config.yahoo_id
yahoo_app_password = config.yahoo_app_password
pyowm_api_key = config.pyowm_api_key

temperature = config.temperature
max_tokens = config.max_tokens
model_name = config.model_name

def generate_gpt_response(user_message):
    client = OpenAIAzure(
        api_key=azure_api_key,
        azure_endpoint=azure_api_base,
        api_version=azure_chatapi_version,
    )
    syspromptmessage = f"""You are Edith, a world-class AI assistant that helps Dinesh to summarize reports. Analyze the report and rewrite it in an engaging way. Note that a python script runs in the background daily to generate the report and your final response will be wrapped and sent to Dinesh as an email update. For year progress reports, include an inspirational quote at the end of the report to get the day started on a positive note. For news updates, extract key information and provide a 5 bullet-point summary for each news category. Do not mix both of them."""
    system_prompt = [{
        "role": "system",
        "content": syspromptmessage
    }]
    conversation = system_prompt.copy()
    conversation.append({"role": "user", "content": str(user_message)})
    response = client.chat.completions.create(
        model=azure_gpt4omini_deploymentid,
        messages=conversation,
        max_tokens=1024,
        temperature=0.3,
    )
    message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": str(message)})

    return message

def get_weather():
    owm = OWM(pyowm_api_key)
    mgr = owm.weather_manager()
    weather = mgr.weather_at_id(5743413).weather  # North Plains, OR
    temp = weather.temperature('celsius')['temp']
    status = weather.detailed_status
    return temp, status

def generate_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left):
    # Weather setup
    temp, status = get_weather()

    # Date and time
    now = datetime.now()
    date_time = now.strftime("%B %d, %Y %H:%M:%S")

    # Determine the season
    month = now.month
    if month in (12, 1, 2):
        season = "Winter 🥶"
    elif month in (3, 4, 5):
        season = "Spring 🌸"
    elif month in (6, 7, 8):
        season = "Summer 🌞"
    else:
        season = "Autumn 🍂"

    progress_bar_full = '█'
    progress_bar_empty = '░'
    progress_bar_length = 20
    progress_filled_length = int(progress_bar_length * (100 - percent_days_left) / 100)
    progress_bar = progress_bar_full * progress_filled_length + progress_bar_empty * (progress_bar_length - progress_filled_length)

    return f"""
    Good Morning Dinesh! 🌞

    Year Progress Report 🌟

    Today's Date and Time: {date_time}
    Weather in North Plains, OR: {temp}°C, {status}

    Current Season: {season}

    Days completed in the year: {days_completed} 📆
    Weeks completed in the year: {weeks_completed:.2f} 🗓️

    Days left in the year: {days_left} 📆
    Weeks left in the year: {weeks_left:.2f} 🗓️
    Percentage of the year left: {percent_days_left:.2f}% 📉

    Year Progress: [{progress_bar}] {100 - percent_days_left:.2f}% completed
    """

def send_email(subject, message):
    sender_email = yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = yahoo_app_password

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    email_message.attach(MIMEText(message, "plain"))

    try:
        server = smtplib.SMTP('smtp.mail.yahoo.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print(f"Email sent successfully with subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def time_left_in_year():
    today = datetime.now()
    end_of_year = datetime(today.year, 12, 31)
    days_completed = today.timetuple().tm_yday
    weeks_completed = days_completed / 7
    delta = end_of_year - today
    days_left = delta.days + 1
    weeks_left = days_left / 7
    total_days_in_year = 366 if (today.year % 4 == 0 and today.year % 100 != 0) or (today.year % 400 == 0) else 365
    percent_days_left = (days_left / total_days_in_year) * 100

    return days_completed, weeks_completed, days_left, weeks_left, percent_days_left

if __name__ == "__main__":
    days_completed, weeks_completed, days_left, weeks_left, percent_days_left = time_left_in_year()
    year_progress_message = generate_progress_message(days_completed, weeks_completed, days_left, weeks_left, percent_days_left)
    # print(year_progress_message)

    year_progress_subject = "Year Progress Report 📅"
    gpt_message = generate_gpt_response(year_progress_message)
    # print(f"\nGPT Response:\n {gpt_message}")
    send_email(year_progress_subject, gpt_message)

    # Convert the year progress report to speech
    yearprogress_tts_output_path = "year_progress_report.mp3"
    text_to_speech(gpt_message, yearprogress_tts_output_path)

    # News Updates
    news_update_subject = "News Updates 📰"
    technews = "Latest news in technology"
    news_update_tech = internet_connected_chatbot(technews, [], model_name, max_tokens, temperature)
    usanews = "Latest news in the USA"
    news_update_usa = internet_connected_chatbot(usanews, [], model_name, max_tokens, temperature)
    india_news = "Latest news in India"
    news_update_india = internet_connected_chatbot(india_news, [], model_name, max_tokens, temperature)

    # Collate all news updates and send them in an email after processing them with gpt
    news_updates = f"""
    Good Morning Dinesh! 🌞

    Tech News Update:
    {news_update_tech}

    USA News Update:
    {news_update_usa}

    India News Update:
    {news_update_india}
    """

    # print(f"\nDetailed News Updates:\n {news_updates}")

    news_gpt_response = generate_gpt_response(news_updates)
    # print(f"\nSummarized News Update:\n {news_gpt_response}")
    
    send_email(news_update_subject, news_gpt_response)

    # Convert the news updates to speech
    news_tts_output_path = "news_update_report.mp3"
    text_to_speech(news_gpt_response, news_tts_output_path)



