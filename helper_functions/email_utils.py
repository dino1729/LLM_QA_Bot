import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import config

logger = logging.getLogger(__name__)


def send_email(subject, message, is_html=False, skip_send=False) -> bool:
    """
    Send email via Yahoo SMTP.
    Returns True if email was sent successfully, False otherwise.
    """
    if skip_send:
        logger.info("[SKIP] Would send email: %s", subject)
        print(f"[SKIP] Email skipped (testing mode): {subject}")
        return True

    sender_email = config.yahoo_id
    receiver_email = "katam.dinesh@hotmail.com"
    password = config.yahoo_app_password

    if not sender_email or not password:
        logger.error("Email credentials not configured in config.yml")
        print("ERROR: Email credentials missing - check yahoo_id and yahoo_app_password in config.yml")
        return False

    email_message = MIMEMultipart()
    email_message["From"] = sender_email
    email_message["To"] = receiver_email
    email_message["Subject"] = subject

    if is_html:
        email_message.attach(MIMEText(message, "html", "utf-8"))
    else:
        email_message.attach(MIMEText(message, "plain", "utf-8"))

    try:
        logger.info("Connecting to Yahoo SMTP server...")
        server = smtplib.SMTP("smtp.mail.yahoo.com", 587, timeout=30)
        server.set_debuglevel(0)

        logger.info("Starting TLS...")
        server.starttls()

        logger.info("Logging in as %s...", sender_email)
        server.login(sender_email, password)

        logger.info("Sending email to %s...", receiver_email)
        text = email_message.as_string()
        server.sendmail(sender_email, receiver_email, text)

        server.quit()
        logger.info("Email sent successfully: %s", subject)
        print(f"✓ Email sent: {subject} → {receiver_email}")
        return True

    except smtplib.SMTPAuthenticationError as e:
        logger.error("SMTP Authentication failed: %s", e)
        print("ERROR: Email authentication failed - check Yahoo app password")
        print("  Hint: Generate an app password at https://login.yahoo.com/account/security")
        return False
    except smtplib.SMTPConnectError as e:
        logger.error("SMTP Connection failed: %s", e)
        print("ERROR: Could not connect to Yahoo SMTP server")
        return False
    except smtplib.SMTPException as e:
        logger.error("SMTP Error: %s", e)
        print(f"ERROR: SMTP error - {e}")
        return False
    except Exception as e:
        logger.error("Failed to send email: %s", e, exc_info=True)
        print(f"ERROR: Failed to send email - {e}")
        return False
