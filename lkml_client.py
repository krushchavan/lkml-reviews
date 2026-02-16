"""HTTP client for lore.kernel.org, adapted from lkml-mcp/client.py."""

import email
import gzip
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class LKMLAPIError(Exception):
    pass


def _is_bot_message(from_field: str) -> bool:
    bot_patterns = ["lkp@intel.com", "bot@", "no-reply@", "robot@"]
    from_lower = from_field.lower()
    return any(pattern in from_lower for pattern in bot_patterns)


class LKMLClient:
    def __init__(
        self,
        base_url: str = "https://lore.kernel.org",
        timeout: int = 20,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "lkml-activity-tracker/0.1.0"})
        self.BASE_URL = base_url
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get_with_retry(self, url: str) -> requests.Response:
        for attempt in range(self.max_retries):
            self._rate_limit()
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 429:
                    wait = self.rate_limit_delay * (2 ** (attempt + 1))
                    logger.warning("Rate limited (429), waiting %.1fs before retry", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException:
                if attempt == self.max_retries - 1:
                    raise
                wait = self.rate_limit_delay * (2 ** attempt)
                logger.warning("Request failed, retrying in %.1fs", wait)
                time.sleep(wait)
        raise LKMLAPIError(f"Failed after {self.max_retries} retries: {url}")

    def _parse_atom_feed(self, content: bytes, max_results: int = 200) -> List[Dict[str, Any]]:
        root = ET.fromstring(content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title_elem = entry.find("atom:title", ns)
            link_elem = entry.find("atom:link", ns)
            updated_elem = entry.find("atom:updated", ns)
            author_elem = entry.find("atom:author/atom:name", ns)

            href = link_elem.get("href") if link_elem is not None else ""
            msg_id = ""
            if href:
                match = re.search(r"/([^/]+)/?$", href.rstrip("/"))
                if match:
                    msg_id = match.group(1)

            entries.append({
                "message_id": msg_id,
                "title": title_elem.text if title_elem is not None else "",
                "updated": updated_elem.text if updated_elem is not None else "",
                "url": href,
                "author_name": author_elem.text if author_elem is not None else "",
            })

        return entries

    def get_user_messages_for_date(
        self, email_addr: str, date_str: str, max_results: int = 200
    ) -> List[Dict[str, Any]]:
        """Fetch all messages from a user on a specific date.

        Args:
            email_addr: User email address.
            date_str: Date in YYYYMMDD format.
            max_results: Maximum entries to retrieve.

        Returns:
            List of dicts with: message_id, title, updated, url, author_name.
        """
        # Use d: filter with same date for single-day range
        url = f"{self.BASE_URL}/all/?q=f:{email_addr}+d:{date_str}..{date_str}&x=A"
        logger.debug("Fetching messages for %s on %s: %s", email_addr, date_str, url)

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=self.timeout)
            # lore.kernel.org returns 404 when no results match the query
            if response.status_code == 404:
                logger.debug("No messages found for %s on %s (404)", email_addr, date_str)
                return []
            response.raise_for_status()
            return self._parse_atom_feed(response.content, max_results)
        except requests.exceptions.RequestException as e:
            raise LKMLAPIError(f"Failed to fetch messages for {email_addr}: {e}") from e
        except ET.ParseError as e:
            raise LKMLAPIError(f"Failed to parse Atom feed for {email_addr}: {e}") from e

    def get_user_messages_for_range(
        self, email_addr: str, start_date: str, end_date: str, max_results: int = 200
    ) -> List[Dict[str, Any]]:
        """Fetch all messages from a user within a date range.

        Args:
            email_addr: User email address.
            start_date: Start date in YYYYMMDD format (inclusive).
            end_date: End date in YYYYMMDD format (inclusive).
            max_results: Maximum entries to retrieve.

        Returns:
            List of dicts with: message_id, title, updated, url, author_name.
        """
        url = f"{self.BASE_URL}/all/?q=f:{email_addr}+d:{start_date}..{end_date}&x=A"
        logger.debug("Fetching messages for %s from %s to %s: %s", email_addr, start_date, end_date, url)

        try:
            self._rate_limit()
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 404:
                logger.debug("No messages found for %s in range %s..%s (404)", email_addr, start_date, end_date)
                return []
            response.raise_for_status()
            return self._parse_atom_feed(response.content, max_results)
        except requests.exceptions.RequestException as e:
            raise LKMLAPIError(f"Failed to fetch messages for {email_addr}: {e}") from e
        except ET.ParseError as e:
            raise LKMLAPIError(f"Failed to parse Atom feed for {email_addr}: {e}") from e

    def get_thread(self, message_id: str, include_bots: bool = False) -> Dict[str, Any]:
        """Fetch a full thread by message-id via mbox.gz.

        Returns:
            Dict with 'message_id' and 'messages' list. Each message has:
            subject, from, date, body, message_id, in_reply_to.
        """
        message_id = message_id.strip("<>")
        url = f"{self.BASE_URL}/r/{message_id}/t.mbox.gz"
        logger.debug("Fetching thread: %s", url)

        try:
            response = self._get_with_retry(url)
            mbox_data = gzip.decompress(response.content)
            mbox_text = mbox_data.decode("utf-8", errors="ignore")

            raw_messages = []
            current_message: List[str] = []
            for line in mbox_text.split("\n"):
                if line.startswith("From ") and current_message:
                    raw_messages.append("\n".join(current_message))
                    current_message = []
                else:
                    current_message.append(line)
            if current_message:
                raw_messages.append("\n".join(current_message))

            messages = []
            for raw_msg in raw_messages:
                if not raw_msg.strip():
                    continue
                msg = email.message_from_string(raw_msg)

                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body += payload.decode(errors="ignore")
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="ignore")

                from_field = msg.get("from", "")
                if not include_bots and _is_bot_message(from_field):
                    continue

                messages.append({
                    "subject": msg.get("subject", ""),
                    "from": from_field,
                    "date": msg.get("date", ""),
                    "body": body,
                    "message_id": msg.get("message-id", "").strip("<>"),
                    "in_reply_to": msg.get("in-reply-to", "").strip("<>"),
                })

            return {"message_id": message_id, "messages": messages}

        except requests.exceptions.RequestException as e:
            raise LKMLAPIError(f"Failed to fetch thread {message_id}: {e}") from e
        except Exception as e:
            raise LKMLAPIError(f"Failed to parse thread {message_id}: {e}") from e

    def get_raw_message(self, message_id: str) -> str:
        """Fetch a single message's raw body text.

        Returns:
            The plain text body of the message.
        """
        message_id = message_id.strip("<>")
        url = f"{self.BASE_URL}/r/{message_id}/raw"
        logger.debug("Fetching raw message: %s", url)

        try:
            response = self._get_with_retry(url)
            raw_text = response.text

            # Parse the raw RFC822 message to extract just the body
            msg = email.message_from_string(raw_text)
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode(errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode(errors="ignore")

            return body

        except requests.exceptions.RequestException as e:
            raise LKMLAPIError(f"Failed to fetch raw message {message_id}: {e}") from e
        except Exception as e:
            raise LKMLAPIError(f"Failed to parse raw message {message_id}: {e}") from e
