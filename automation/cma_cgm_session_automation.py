#!/usr/bin/env python3
"""
CMA CGM Session-Based Quote Automation
Uses user-provided session for authenticated quote retrieval
"""

import os
import sys
import time
import json
import logging
import traceback
import requests
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

# Enhanced import handling
SELENIUM_AVAILABLE = False
UNDETECTED_CHROME_AVAILABLE = False
WEBDRIVER_MANAGER_AVAILABLE = False
SUPABASE_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
    print("✅ Selenium available")
except ImportError as e:
    print(f"❌ Selenium not available: {e}")

try:
    import undetected_chromedriver as uc
    UNDETECTED_CHROME_AVAILABLE = True
    print("✅ Undetected Chrome available")
except ImportError as e:
    print(f"❌ Undetected Chrome not available: {e}")

try:
    from webdriver_manager.chrome import ChromeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
    print("✅ WebDriver Manager available")
except ImportError as e:
    print(f"❌ WebDriver Manager not available: {e}")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase available")
except ImportError as e:
    print(f"❌ Supabase not available: {e}")

class SessionBasedLogger:
    """Enhanced logging for session-based automation"""
    
    def __init__(self, run_id=None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.step_count = 0
        self.setup_logging()
        
    def setup_logging(self):
        """Set up comprehensive logging"""
        os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger('session_automation')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        file_handler = logging.FileHandler(f'logs/session_automation_{self.run_id}.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(funcName)20s:%(lineno)4d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.log_event("SESSION_AUTOMATION_START", {
            "run_id": self.run_id,
            "approach": "session_based_quote_automation",
            "max_execution_time": "120_seconds"
        })
        
    def log_event(self, event_type, data=None, level="INFO"):
        """Log structured event"""
        self.step_count += 1
        
        message = f"[{self.step_count:03d}] {event_type}"
        if data:
            message += f" | {json.dumps(data, default=str)}"
            
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
            
    def log_error(self, error_type, error_message):
        """Log error with context"""
        self.log_event("ERROR", {
            "error_type": error_type,
            "error_message": str(error_message),
            "traceback": traceback.format_exc()
        }, "ERROR")

class CMASessionAPIClient:
    """API client that uses user's session for authenticated requests"""
    
    def __init__(self, session_data, logger):
        self.session_data = session_data
        self.logger = logger
        self.session = requests.Session()
        self.base_url = "https://www.cma-cgm.com"
        self.setup_session()
        
    def setup_session(self):
        """Setup requests session with user's session data"""
        try:
            self.logger.log_event("SESSION_SETUP_START", "Setting up authenticated session")
            
            # Set realistic headers
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            })
            
            # Add session token if provided
            if 'sessionToken' in self.session_data:
                # Try different ways to use the session token
                session_token = self.session_data['sessionToken']
                
                # Method 1: As Authorization header
                self.session.headers['Authorization'] = f'Bearer {session_token}'
                
                # Method 2: As custom session header
                self.session.headers['X-Session-Token'] = session_token
                
                # Method 3: As cookie
                self.session.cookies.set('session', session_token, domain='.cma-cgm.com')
                self.session.cookies.set('JSESSIONID', session_token, domain='.cma-cgm.com')
                
                self.logger.log_event("SESSION_TOKEN_APPLIED", {"token_length": len(session_token)})
            
            # Add cookies if provided
            if 'cookies' in self.session_data:
                for cookie in self.session_data['cookies']:
                    self.session.cookies.set(
                        cookie.get('name', ''),
                        cookie.get('value', ''),
                        domain=cookie.get('domain', '.cma-cgm.com')
                    )
                self.logger.log_event("SESSION_COOKIES_APPLIED", {"cookie_count": len(self.session_data['cookies'])})
            
            self.logger.log_event("SESSION_SETUP_COMPLETE", "Authenticated session ready")
            
        except Exception as e:
            self.logger.log_error("SESSION_SETUP_FAILED", f"Failed to setup session: {e}")
    
    def validate_session(self):
        """Validate that the session is still active"""
        try:
            self.logger.log_event("SESSION_VALIDATION_START", "Checking session validity")
            
            # Try to access a protected page
            response = self.session.get(f"{self.base_url}/ebusiness/pricing/instant-Quoting", timeout=15)
            
            if response.status_code == 200:
                # Check for login indicators
                page_content = response.text.lower()
                
                # Look for logged-in indicators
                logged_in_indicators = [
                    'welcome',
                    'logout',
                    'my account',
                    'user-menu',
                    'dashboard'
                ]
                
                # Look for login required indicators
                login_required_indicators = [
                    'please log in',
                    'login required',
                    'authentication required',
                    'sign in'
                ]
                
                has_logged_in_indicator = any(indicator in page_content for indicator in logged_in_indicators)
                has_login_required = any(indicator in page_content for indicator in login_required_indicators)
                
                if has_logged_in_indicator and not has_login_required:
                    self.logger.log_event("SESSION_VALID", "Session is active and authenticated")
                    return True
                else:
                    self.logger.log_event("SESSION_INVALID", "Session appears to be expired or invalid")
                    return False
            else:
                self.logger.log_event("SESSION_VALIDATION_FAILED", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.log_error("SESSION_VALIDATION_ERROR", f"Session validation error: {e}")
            return False
    
    def search_quotes_with_session(self, origin, destination, container_type):
        """Search for quotes using the authenticated session"""
        try:
            self.logger.log_event("QUOTE_SEARCH_START", {
                "origin": origin,
                "destination": destination,
                "container_type": container_type
            })
            
            # First validate session
            if not self.validate_session():
                raise Exception("Session is invalid or expired")
            
            # Try API endpoints first
            quotes = self.try_api_endpoints(origin, destination, container_type)
            if quotes:
                return quotes
            
            # Fallback to browser automation with session
            quotes = self.try_browser_automation(origin, destination, container_type)
            return quotes
            
        except Exception as e:
            self.logger.log_error("QUOTE_SEARCH_FAILED", f"Quote search failed: {e}")
            return []
    
    def try_api_endpoints(self, origin, destination, container_type):
        """Try various API endpoints for quote retrieval"""
        try:
            self.logger.log_event("API_ENDPOINTS_START", "Trying authenticated API endpoints")
            
            # Common API endpoints for quote search
            api_endpoints = [
                '/api/quotes/instant',
                '/api/quotes/search',
                '/ebusiness/api/quotes/instant',
                '/ebusiness/api/quotes/search',
                '/ajax/quotes/search',
                '/ajax/quotes/instant',
                '/services/quotes/search'
            ]
            
            # Prepare search data
            search_data = {
                'origin': origin,
                'destination': destination,
                'containerType': container_type,
                'container_type': container_type,
                'equipmentType': f"{container_type}GP",
                'commodity': 'FREIGHT ALL KINDS',
                'weight': 1000,
                'currency': 'USD'
            }
            
            for endpoint in api_endpoints:
                try:
                    url = urljoin(self.base_url, endpoint)
                    self.logger.log_event("API_ENDPOINT_TRY", f"Trying: {endpoint}")
                    
                    # Set appropriate headers for this request
                    headers = {
                        'Content-Type': 'application/json',
                        'X-Requested-With': 'XMLHttpRequest',
                        'Referer': f"{self.base_url}/ebusiness/pricing/instant-Quoting",
                        'Origin': self.base_url
                    }
                    
                    # Try POST request
                    response = self.session.post(url, json=search_data, headers=headers, timeout=20)
                    
                    if response.status_code in [200, 201]:
                        try:
                            result = response.json()
                            if self.has_quote_data(result):
                                quotes = self.parse_api_quotes(result, origin, destination, container_type)
                                if quotes:
                                    self.logger.log_event("API_QUOTES_FOUND", f"Found {len(quotes)} quotes via {endpoint}")
                                    return quotes
                        except json.JSONDecodeError:
                            # Check for quote data in HTML response
                            if self.has_quote_data_in_html(response.text):
                                quotes = self.parse_html_quotes(response.text, origin, destination, container_type)
                                if quotes:
                                    self.logger.log_event("HTML_QUOTES_FOUND", f"Found {len(quotes)} quotes via {endpoint}")
                                    return quotes
                    
                    # Try GET request with parameters
                    params = {
                        'origin': origin,
                        'destination': destination,
                        'containerType': container_type
                    }
                    response = self.session.get(url, params=params, headers=headers, timeout=20)
                    
                    if response.status_code in [200, 201]:
                        try:
                            result = response.json()
                            if self.has_quote_data(result):
                                quotes = self.parse_api_quotes(result, origin, destination, container_type)
                                if quotes:
                                    self.logger.log_event("API_QUOTES_FOUND", f"Found {len(quotes)} quotes via {endpoint}")
                                    return quotes
                        except json.JSONDecodeError:
                            if self.has_quote_data_in_html(response.text):
                                quotes = self.parse_html_quotes(response.text, origin, destination, container_type)
                                if quotes:
                                    self.logger.log_event("HTML_QUOTES_FOUND", f"Found {len(quotes)} quotes via {endpoint}")
                                    return quotes
                    
                except requests.exceptions.RequestException as e:
                    self.logger.log_event("API_ENDPOINT_FAILED", f"Endpoint {endpoint} failed: {e}")
                    continue
            
            self.logger.log_event("API_ENDPOINTS_FAILED", "No quotes found via API endpoints")
            return []
            
        except Exception as e:
            self.logger.log_error("API_ENDPOINTS_ERROR", f"API endpoints error: {e}")
            return []
    
    def has_quote_data(self, data):
        """Check if response contains quote data"""
        if not isinstance(data, dict):
            return False
        
        quote_keys = ['quotes', 'rates', 'results', 'prices', 'offers']
        return any(key in data for key in quote_keys)
    
    def has_quote_data_in_html(self, html):
        """Check if HTML contains quote data"""
        html_lower = html.lower()
        quote_indicators = ['rate', 'price', 'usd', 'eur', 'cost', 'quote', 'freight']
        return any(indicator in html_lower for indicator in quote_indicators)
    
    def parse_api_quotes(self, data, origin, destination, container_type):
        """Parse quotes from API response"""
        quotes = []
        try:
            # Extract quote data from various possible structures
            quote_data = None
            for key in ['quotes', 'rates', 'results', 'prices', 'offers']:
                if key in data:
                    quote_data = data[key]
                    break
            
            if not quote_data:
                return []
            
            if not isinstance(quote_data, list):
                quote_data = [quote_data]
            
            for item in quote_data:
                if isinstance(item, dict):
                    quote = self.extract_quote_from_item(item, origin, destination, container_type)
                    if quote:
                        quotes.append(quote)
            
            return quotes
            
        except Exception as e:
            self.logger.log_error("API_QUOTE_PARSE_FAILED", f"Failed to parse API quotes: {e}")
            return []
    
    def parse_html_quotes(self, html, origin, destination, container_type):
        """Parse quotes from HTML response"""
        quotes = []
        try:
            # Look for price patterns in HTML
            price_patterns = [
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*USD',
                r'USD\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*\$'
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, html)
                if matches:
                    for match in matches:
                        try:
                            price_str = match.replace(',', '')
                            price = float(price_str)
                            
                            # Reasonable rate range for container shipping
                            if 100 <= price <= 15000:
                                quote = {
                                    'origin': origin,
                                    'destination': destination,
                                    'container_type': container_type,
                                    'rate': int(price),
                                    'currency': 'USD',
                                    'service': 'CMA CGM Ocean Freight',
                                    'transit_time': '15-25 days',
                                    'valid_until': (datetime.now().date() + timedelta(days=30)).isoformat(),
                                    'source': 'cma_cgm_session_html_parse',
                                    'timestamp': datetime.now().isoformat()
                                }
                                quotes.append(quote)
                                
                                # Limit to first reasonable quote found
                                if len(quotes) >= 1:
                                    return quotes
                                    
                        except ValueError:
                            continue
            
            return quotes
            
        except Exception as e:
            self.logger.log_error("HTML_QUOTE_PARSE_FAILED", f"Failed to parse HTML quotes: {e}")
            return []
    
    def extract_quote_from_item(self, item, origin, destination, container_type):
        """Extract quote information from a data item"""
        try:
            # Try to extract rate/price
            rate = None
            for rate_key in ['rate', 'price', 'cost', 'amount', 'total']:
                if rate_key in item:
                    rate = item[rate_key]
                    break
            
            if rate is None:
                return None
            
            # Convert to number if string
            if isinstance(rate, str):
                rate_clean = re.sub(r'[^\d.]', '', rate)
                try:
                    rate = float(rate_clean)
                except ValueError:
                    return None
            
            # Validate rate range
            if not (100 <= rate <= 15000):
                return None
            
            # Extract other fields
            currency = item.get('currency', 'USD')
            service = item.get('service', item.get('product', 'CMA CGM Ocean Freight'))
            transit_time = item.get('transit_time', item.get('transitTime', '15-25 days'))
            valid_until = item.get('valid_until', item.get('validUntil', 
                                  (datetime.now().date() + timedelta(days=30)).isoformat()))
            
            return {
                'origin': origin,
                'destination': destination,
                'container_type': container_type,
                'rate': int(rate),
                'currency': currency,
                'service': service,
                'transit_time': transit_time,
                'valid_until': valid_until,
                'source': 'cma_cgm_session_api',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error("QUOTE_EXTRACTION_FAILED", f"Failed to extract quote: {e}")
            return None
    
    def try_browser_automation(self, origin, destination, container_type):
        """Fallback browser automation with session cookies"""
        if not SELENIUM_AVAILABLE:
            self.logger.log_event("BROWSER_UNAVAILABLE", "Selenium not available for browser automation")
            return []
        
        driver = None
        try:
            self.logger.log_event("BROWSER_AUTOMATION_START", "Starting browser automation with session")
            
            # Setup Chrome options
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            # Initialize driver
            if WEBDRIVER_MANAGER_AVAILABLE:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
            else:
                driver = webdriver.Chrome(options=options)
            
            # Navigate to quote page
            driver.get(f"{self.base_url}/ebusiness/pricing/instant-Quoting")
            
            # Inject session cookies
            if 'cookies' in self.session_data:
                for cookie in self.session_data['cookies']:
                    try:
                        driver.add_cookie({
                            'name': cookie.get('name', ''),
                            'value': cookie.get('value', ''),
                            'domain': cookie.get('domain', '.cma-cgm.com')
                        })
                    except Exception as e:
                        continue
            
            # Add session token as cookie if available
            if 'sessionToken' in self.session_data:
                try:
                    driver.add_cookie({
                        'name': 'session',
                        'value': self.session_data['sessionToken'],
                        'domain': '.cma-cgm.com'
                    })
                except Exception as e:
                    pass
            
            # Refresh to apply cookies
            driver.refresh()
            time.sleep(3)
            
            # Check if logged in
            if self.is_logged_in_browser(driver):
                self.logger.log_event("BROWSER_SESSION_VALID", "Browser session is authenticated")
                return self.fill_quote_form_browser(driver, origin, destination, container_type)
            else:
                self.logger.log_event("BROWSER_SESSION_INVALID", "Browser session is not authenticated")
                return []
                
        except Exception as e:
            self.logger.log_error("BROWSER_AUTOMATION_FAILED", f"Browser automation failed: {e}")
            return []
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    def is_logged_in_browser(self, driver):
        """Check if browser session is authenticated"""
        try:
            # Look for logged-in indicators
            logged_in_selectors = [
                "//span[contains(text(), 'Welcome')]",
                "//div[contains(@class, 'user-menu')]",
                "//button[contains(text(), 'Logout')]",
                "//a[contains(text(), 'My Account')]",
                "//div[contains(@class, 'logged-in')]"
            ]
            
            for selector in logged_in_selectors:
                try:
                    elements = driver.find_elements(By.XPATH, selector)
                    if elements:
                        return True
                except:
                    continue
            
            # Check page content for login indicators
            page_source = driver.page_source.lower()
            if any(indicator in page_source for indicator in ['welcome', 'logout', 'my account']):
                return True
            
            return False
            
        except Exception as e:
            self.logger.log_error("LOGIN_CHECK_FAILED", f"Failed to check login status: {e}")
            return False
    
    def fill_quote_form_browser(self, driver, origin, destination, container_type):
        """Fill quote form using browser automation"""
        try:
            self.logger.log_event("FORM_FILLING_START", "Filling quote form in browser")
            
            # Wait for form to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "form"))
            )
            
            # Try different selectors for origin field
            origin_selectors = [
                'input[placeholder*="origin"]',
                'input[name*="origin"]',
                'input[id*="origin"]',
                'input[class*="origin"]'
            ]
            
            origin_filled = False
            for selector in origin_selectors:
                try:
                    origin_field = driver.find_element(By.CSS_SELECTOR, selector)
                    origin_field.clear()
                    origin_field.send_keys(origin)
                    origin_filled = True
                    break
                except:
                    continue
            
            if not origin_filled:
                raise Exception("Could not find origin field")
            
            # Try different selectors for destination field
            dest_selectors = [
                'input[placeholder*="destination"]',
                'input[name*="destination"]',
                'input[id*="destination"]',
                'input[class*="destination"]'
            ]
            
            dest_filled = False
            for selector in dest_selectors:
                try:
                    dest_field = driver.find_element(By.CSS_SELECTOR, selector)
                    dest_field.clear()
                    dest_field.send_keys(destination)
                    dest_filled = True
                    break
                except:
                    continue
            
            if not dest_filled:
                raise Exception("Could not find destination field")
            
            # Try to select container type
            container_selectors = [
                'select[name*="container"]',
                'select[id*="container"]',
                'select[class*="container"]'
            ]
            
            for selector in container_selectors:
                try:
                    container_select = driver.find_element(By.CSS_SELECTOR, selector)
                    container_select.send_keys(container_type)
                    break
                except:
                    continue
            
            # Submit form
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button[class*="submit"]',
                'button[class*="search"]'
            ]
            
            form_submitted = False
            for selector in submit_selectors:
                try:
                    submit_button = driver.find_element(By.CSS_SELECTOR, selector)
                    submit_button.click()
                    form_submitted = True
                    break
                except:
                    continue
            
            if not form_submitted:
                raise Exception("Could not find submit button")
            
            # Wait for results
            try:
                WebDriverWait(driver, 30).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.quote-results')),
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.rate-results')),
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.results')),
                        EC.presence_of_element_located((By.CSS_SELECTOR, '[class*="result"]'))
                    )
                )
            except TimeoutException:
                self.logger.log_event("RESULTS_TIMEOUT", "Timeout waiting for results")
                # Still try to extract quotes from current page
            
            # Extract quotes from results page
            return self.extract_quotes_from_browser(driver, origin, destination, container_type)
            
        except Exception as e:
            self.logger.log_error("FORM_FILLING_FAILED", f"Form filling failed: {e}")
            return []
    
    def extract_quotes_from_browser(self, driver, origin, destination, container_type):
        """Extract quotes from browser results page"""
        quotes = []
        try:
            self.logger.log_event("QUOTE_EXTRACTION_START", "Extracting quotes from browser")
            
            # Look for quote/rate elements
            quote_selectors = [
                '.quote-item',
                '.rate-item',
                '.result-item',
                '[class*="quote"]',
                '[class*="rate"]',
                '[class*="price"]'
            ]
            
            quote_elements = []
            for selector in quote_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        quote_elements.extend(elements)
                except:
                    continue
            
            if not quote_elements:
                # Try to extract from page source
                page_source = driver.page_source
                return self.parse_html_quotes(page_source, origin, destination, container_type)
            
            for element in quote_elements:
                try:
                    # Extract rate information
                    rate_selectors = ['.price', '.rate', '.cost', '[class*="price"]', '[class*="rate"]']
                    rate_text = None
                    
                    for rate_selector in rate_selectors:
                        try:
                            rate_element = element.find_element(By.CSS_SELECTOR, rate_selector)
                            rate_text = rate_element.text
                            break
                        except:
                            continue
                    
                    if not rate_text:
                        rate_text = element.text
                    
                    # Extract service information
                    service_selectors = ['.service', '.product', '[class*="service"]']
                    service_text = 'CMA CGM Ocean Freight'
                    
                    for service_selector in service_selectors:
                        try:
                            service_element = element.find_element(By.CSS_SELECTOR, service_selector)
                            service_text = service_element.text
                            break
                        except:
                            continue
                    
                    # Extract transit time
                    transit_selectors = ['.transit', '.time', '[class*="transit"]', '[class*="time"]']
                    transit_text = '15-25 days'
                    
                    for transit_selector in transit_selectors:
                        try:
                            transit_element = element.find_element(By.CSS_SELECTOR, transit_selector)
                            transit_text = transit_element.text
                            break
                        except:
                            continue
                    
                    # Parse rate from text
                    rate_match = re.search(r'[\$€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', rate_text)
                    if rate_match:
                        rate = float(rate_match.group(1).replace(',', ''))
                        
                        if 100 <= rate <= 15000:  # Reasonable range
                            quote = {
                                'origin': origin,
                                'destination': destination,
                                'container_type': container_type,
                                'rate': int(rate),
                                'currency': 'USD',
                                'service': service_text,
                                'transit_time': transit_text,
                                'valid_until': (datetime.now().date() + timedelta(days=30)).isoformat(),
                                'source': 'cma_cgm_session_browser',
                                'timestamp': datetime.now().isoformat()
                            }
                            quotes.append(quote)
                            
                except Exception as e:
                    continue
            
            self.logger.log_event("QUOTE_EXTRACTION_COMPLETE", f"Extracted {len(quotes)} quotes from browser")
            return quotes
            
        except Exception as e:
            self.logger.log_error("QUOTE_EXTRACTION_FAILED", f"Quote extraction failed: {e}")
            return []

class SessionBasedAutomation:
    """Main automation class for session-based quote retrieval"""
    
    def __init__(self, logger):
        self.logger = logger
        self.max_execution_time = 120  # 2 minutes
        self.start_time = time.time()
    
    def check_time_limit(self):
        """Check if execution time limit exceeded"""
        elapsed = time.time() - self.start_time
        if elapsed > self.max_execution_time:
            self.logger.log_error("TIME_LIMIT_EXCEEDED", f"Exceeded {self.max_execution_time} second limit")
            return False
        return True
    
    def get_quotes_with_session(self, session_data, search_params):
        """Main method to get quotes using user session"""
        try:
            self.logger.log_event("SESSION_AUTOMATION_START", {
                "search_params": search_params,
                "session_timestamp": session_data.get('timestamp'),
                "has_session_token": 'sessionToken' in session_data,
                "has_cookies": 'cookies' in session_data
            })
            
            if not self.check_time_limit():
                raise Exception("Time limit exceeded before starting")
            
            # Initialize API client with session
            api_client = CMASessionAPIClient(session_data, self.logger)
            
            if not self.check_time_limit():
                raise Exception("Time limit exceeded during initialization")
            
            # Attempt quote retrieval
            quotes = api_client.search_quotes_with_session(
                search_params['origin'],
                search_params['destination'],
                search_params['container_type']
            )
            
            if quotes:
                self.logger.log_event("SESSION_AUTOMATION_SUCCESS", f"Retrieved {len(quotes)} real quotes")
                return quotes
            else:
                self.logger.log_event("SESSION_AUTOMATION_NO_QUOTES", "No quotes found with session")
                return []
                
        except Exception as e:
            self.logger.log_error("SESSION_AUTOMATION_FAILED", f"Session automation failed: {e}")
            return []

def save_quotes_to_database(quotes, request_id, user_id, logger):
    """Save extracted quotes to Supabase database"""
    if not SUPABASE_AVAILABLE:
        logger.log_error("SUPABASE_UNAVAILABLE", "Cannot save quotes - Supabase client not available")
        return False
    
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.log_error("SUPABASE_CONFIG_MISSING", "Missing Supabase configuration")
            return False
        
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        logger.log_event("DATABASE_SAVE_START", f"Saving {len(quotes)} quotes to database")
        
        # Update session automation results table with completion
        update_data = {
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'rates': quotes,
            'github_run_id': os.getenv('RUN_ID')
        }
        
        result = supabase.table('session_automation_results').update(update_data).eq('request_id', request_id).execute()
        
        if result.data:
            logger.log_event("DATABASE_SAVE_SUCCESS", f"Successfully updated automation results for request {request_id}")
            return True
        else:
            logger.log_error("DATABASE_SAVE_FAILED", "Failed to update session automation results")
            return False
            
    except Exception as e:
        logger.log_error("DATABASE_SAVE_ERROR", f"Database save error: {e}")
        return False

def update_automation_status_on_error(request_id, error_message, logger):
    """Update automation status when an error occurs"""
    if not SUPABASE_AVAILABLE:
        return
        
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not supabase_url or not supabase_key:
            return
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        update_data = {
            'status': 'failed',
            'completed_at': datetime.now().isoformat(),
            'error_message': error_message,
            'github_run_id': os.getenv('RUN_ID')
        }
        
        supabase.table('session_automation_results').update(update_data).eq('request_id', request_id).execute()
        logger.log_event("DATABASE_ERROR_UPDATE", f"Updated error status for request {request_id}")
        
    except Exception as e:
        logger.log_error("DATABASE_ERROR_UPDATE_FAILED", f"Failed to update error status: {e}")

def main():
    """Main function for session-based automation"""
    run_id = os.getenv('RUN_ID', datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = SessionBasedLogger(run_id)
    
    request_id = None
    user_id = None
    
    try:
        # Get parameters
        search_params = {
            'origin': os.getenv('ORIGIN', 'Yokohama'),
            'destination': os.getenv('DESTINATION', 'Hong Kong'),
            'container_type': os.getenv('CONTAINER_TYPE', '20')
        }
        
        # Get additional parameters for database tracking
        request_id = os.getenv('REQUEST_ID')
        user_id = os.getenv('USER_ID')
        
        if not request_id:
            raise Exception("REQUEST_ID environment variable is required")
        
        if not user_id:
            raise Exception("USER_ID environment variable is required")
        
        # Get session data from environment
        session_data_str = os.getenv('SESSION_DATA')
        if not session_data_str:
            raise Exception("SESSION_DATA environment variable is required")
        
        try:
            session_data = json.loads(session_data_str)
        except json.JSONDecodeError:
            raise Exception("Invalid SESSION_DATA format - must be valid JSON")
        
        # Validate session data structure
        if not isinstance(session_data, dict):
            raise Exception("SESSION_DATA must be a JSON object")
        
        # Check session age
        session_timestamp = session_data.get('timestamp')
        if session_timestamp:
            session_age_minutes = (time.time() * 1000 - session_timestamp) / 1000 / 60
            if session_age_minutes > 30:  # 30 minute limit
                raise Exception(f"Session expired - age: {session_age_minutes:.1f} minutes")
        
        logger.log_event("SESSION_AUTOMATION_INIT", {
            "request_id": request_id,
            "user_id": user_id,
            "search_params": search_params,
            "session_data_keys": list(session_data.keys()),
            "session_age_minutes": session_age_minutes if session_timestamp else "unknown"
        })
        
        # Validate required session data
        if 'sessionToken' not in session_data and 'cookies' not in session_data:
            raise Exception("Session data must contain either sessionToken or cookies")
        
        # Initialize automation
        automation = SessionBasedAutomation(logger)
        
        # Get quotes using session
        quotes = automation.get_quotes_with_session(session_data, search_params)
        
        if quotes:
            logger.log_event("SESSION_AUTOMATION_COMPLETE", f"Successfully retrieved {len(quotes)} real quotes")
            
            # Save quotes to database
            database_success = save_quotes_to_database(quotes, request_id, user_id, logger)
            
            # Output for GitHub Actions
            print(f"::set-output name=rates_count::{len(quotes)}")
            print(f"::set-output name=status::success")
            print(f"::set-output name=rates::{json.dumps(quotes, default=str)}")
            print(f"::set-output name=database_saved::{database_success}")
            print(f"::set-output name=request_id::{request_id}")
            
            return 0
        else:
            logger.log_event("SESSION_AUTOMATION_NO_RESULTS", "No quotes retrieved - session may be expired or invalid")
            
            # Update database with no results status
            if request_id:
                update_automation_status_on_error(request_id, "No quotes found - session may be expired", logger)
            
            # Output for GitHub Actions
            print(f"::set-output name=rates_count::0")
            print(f"::set-output name=status::no_quotes")
            print(f"::set-output name=error::No quotes found - session may be expired")
            print(f"::set-output name=rates::[]")
            print(f"::set-output name=database_saved::false")
            print(f"::set-output name=request_id::{request_id}")
            
            return 0  # Still return success - no quotes is a valid result
        
    except Exception as e:
        error_message = str(e)
        logger.log_error("SESSION_AUTOMATION_CRITICAL_ERROR", f"Critical error: {error_message}")
        
        # Update database with error status
        if request_id:
            update_automation_status_on_error(request_id, error_message, logger)
        
        # Output for GitHub Actions
        print(f"::set-output name=rates_count::0")
        print(f"::set-output name=status::error")
        print(f"::set-output name=error::{error_message}")
        print(f"::set-output name=rates::[]")
        print(f"::set-output name=database_saved::false")
        print(f"::set-output name=request_id::{request_id or 'unknown'}")
        
        return 0  # Return success even for errors (clean failure)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

