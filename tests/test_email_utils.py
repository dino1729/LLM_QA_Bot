"""
Tests for email_utils.py - Email sending functionality
"""
import pytest
import smtplib
from unittest.mock import Mock, patch, MagicMock, call
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from helper_functions import email_utils


class TestSendEmail:
    """Tests for send_email() function"""
    
    def test_send_email_skip_mode(self):
        """Test email sending in skip mode (testing)"""
        result = email_utils.send_email(
            subject="Test Subject",
            message="Test message",
            skip_send=True
        )
        
        assert result is True
    
    def test_send_email_skip_mode_with_html(self):
        """Test HTML email sending in skip mode"""
        result = email_utils.send_email(
            subject="HTML Test",
            message="<h1>Test</h1>",
            is_html=True,
            skip_send=True
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.config')
    def test_send_email_missing_credentials(self, mock_config):
        """Test email sending with missing credentials"""
        mock_config.yahoo_id = None
        mock_config.yahoo_app_password = None
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.config')
    def test_send_email_missing_sender(self, mock_config):
        """Test email sending with missing sender email"""
        mock_config.yahoo_id = None
        mock_config.yahoo_app_password = "test_password"
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.config')
    def test_send_email_missing_password(self, mock_config):
        """Test email sending with missing password"""
        mock_config.yahoo_id = "test@yahoo.com"
        mock_config.yahoo_app_password = None
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_success_plain_text(self, mock_config, mock_smtp):
        """Test successful plain text email sending"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password_123"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="Test Subject",
            message="Plain text message body",
            is_html=False
        )
        
        assert result is True
        mock_smtp.assert_called_once_with("smtp.mail.yahoo.com", 587, timeout=30)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@yahoo.com", "app_password_123")
        mock_server.sendmail.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_success_html(self, mock_config, mock_smtp):
        """Test successful HTML email sending"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password_123"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="HTML Email",
            message="<html><body><h1>Hello</h1></body></html>",
            is_html=True
        )
        
        assert result is True
        mock_server.sendmail.assert_called_once()
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_authentication_error(self, mock_config, mock_smtp):
        """Test email sending with authentication error"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "wrong_password"
        
        mock_server = Mock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, "Authentication failed")
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_connection_error(self, mock_config, mock_smtp):
        """Test email sending with connection error"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_smtp.side_effect = smtplib.SMTPConnectError(421, "Cannot connect")
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_smtp_exception(self, mock_config, mock_smtp):
        """Test email sending with generic SMTP exception"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_server.sendmail.side_effect = smtplib.SMTPException("SMTP error occurred")
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_general_exception(self, mock_config, mock_smtp):
        """Test email sending with general exception"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_server.starttls.side_effect = Exception("Unexpected error")
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_unicode_content(self, mock_config, mock_smtp):
        """Test email sending with Unicode content"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="测试邮件 Test Email 🎉",
            message="Hello 世界! This is a test with émojis 🚀 and spëcial çhars",
            is_html=False
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_long_subject(self, mock_config, mock_smtp):
        """Test email sending with very long subject"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        long_subject = "A" * 500  # Very long subject
        result = email_utils.send_email(
            subject=long_subject,
            message="Short message",
            is_html=False
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_long_message(self, mock_config, mock_smtp):
        """Test email sending with very long message"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        long_message = "Lorem ipsum " * 10000  # Very long message
        result = email_utils.send_email(
            subject="Long message test",
            message=long_message,
            is_html=False
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_html_with_images(self, mock_config, mock_smtp):
        """Test HTML email with embedded image tags"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        html_with_images = """
        <html>
            <body>
                <h1>Newsletter</h1>
                <img src="https://example.com/image.png" alt="Test" />
                <p>Content here</p>
            </body>
        </html>
        """
        
        result = email_utils.send_email(
            subject="Newsletter with images",
            message=html_with_images,
            is_html=True
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_empty_message(self, mock_config, mock_smtp):
        """Test email sending with empty message"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        result = email_utils.send_email(
            subject="Empty message test",
            message="",
            is_html=False
        )
        
        assert result is True
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_server_timeout(self, mock_config, mock_smtp):
        """Test email sending with server timeout"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_smtp.side_effect = Exception("Timeout during connection")
        
        result = email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        assert result is False
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_receiver_is_hardcoded(self, mock_config, mock_smtp):
        """Test that receiver email is hardcoded (design verification)"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        # Verify sendmail is called with the hardcoded receiver
        args = mock_server.sendmail.call_args[0]
        assert args[0] == "sender@yahoo.com"  # sender
        assert args[1] == "katam.dinesh@hotmail.com"  # hardcoded receiver
    
    @patch('helper_functions.email_utils.smtplib.SMTP')
    @patch('helper_functions.email_utils.config')
    def test_send_email_debug_level_set(self, mock_config, mock_smtp):
        """Test that SMTP debug level is set to 0"""
        mock_config.yahoo_id = "sender@yahoo.com"
        mock_config.yahoo_app_password = "app_password"
        
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        email_utils.send_email(
            subject="Test",
            message="Test message"
        )
        
        mock_server.set_debuglevel.assert_called_once_with(0)

