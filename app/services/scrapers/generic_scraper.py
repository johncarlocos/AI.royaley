"""
Generic/Custom Web Scraper
==========================
A flexible scraper template for any website.

This scraper is designed to be easily customizable for any data source.
Copy this file and modify it for your specific needs.

CONFIGURATION:
1. Add your target URLs to the URLS list below
2. Define your extraction rules in EXTRACTION_RULES
3. Update parse_page() if you need more complex logic
4. Run the scraper
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from bs4 import BeautifulSoup
import re

from .base_scraper import (
    BaseWebScraper, 
    ScraperConfig, 
    ScrapedData,
    extract_text,
    extract_texts,
    extract_attr,
    extract_number
)


# ============================================================================
# CONFIGURATION - CUSTOMIZE EVERYTHING BELOW
# ============================================================================

# Your target URLs
URLS = [
    # Add your URLs here
    # Example: "https://www.example.com/data"
    # Example: "https://www.example.com/page/1"
    # Example: "https://www.example.com/page/2"
]

BASE_URL = ""  # Optional base URL

# Scraper settings
SCRAPER_NAME = "custom_scraper"
REQUESTS_PER_MINUTE = 20
CACHE_TTL_SECONDS = 300

# Define what data to extract
# Format: "field_name": "CSS selector"
EXTRACTION_RULES = {
    # Single value extractions
    "title": "h1, .title, .heading",
    "description": ".description, .summary, p:first-of-type",
    "date": ".date, time, .published",
    "author": ".author, .byline",
    
    # Add more fields as needed:
    # "price": ".price, .cost",
    # "rating": ".rating, .stars",
    # "category": ".category, .tag",
}

# Define list extractions (multiple items)
# Format: "field_name": {"container": "CSS selector", "item": "CSS selector"}
LIST_EXTRACTION_RULES = {
    # Example: Extract all links
    # "links": {
    #     "container": "body",
    #     "item": "a",
    #     "extract": "href"  # Extract attribute instead of text
    # },
    
    # Example: Extract all list items
    # "items": {
    #     "container": ".item-list, ul",
    #     "item": "li, .item"
    # },
}

# ============================================================================


class GenericScraper(BaseWebScraper):
    """
    Generic/custom scraper for any website.
    
    This scraper uses configurable extraction rules to pull data
    from any website. Customize the rules above for your use case.
    
    Usage:
        # Using default configuration
        scraper = GenericScraper()
        results = await scraper.scrape_all()
        
        # With custom URLs
        scraper = GenericScraper(urls=["https://example.com/page1"])
        results = await scraper.scrape_all()
        
        # With custom extraction rules
        scraper = GenericScraper(
            extraction_rules={"title": "h1", "content": ".main-content"}
        )
    """
    
    def __init__(
        self,
        urls: Optional[List[str]] = None,
        extraction_rules: Optional[Dict[str, str]] = None,
        list_rules: Optional[Dict[str, Dict]] = None,
        scraper_name: Optional[str] = None
    ):
        """
        Initialize the generic scraper.
        
        Args:
            urls: List of URLs to scrape
            extraction_rules: Custom extraction rules (overrides defaults)
            list_rules: Custom list extraction rules
            scraper_name: Custom name for this scraper instance
        """
        config = ScraperConfig(
            base_url=BASE_URL,
            urls=urls or URLS,
            scraper_name=scraper_name or SCRAPER_NAME,
            scraper_version="1.0.0",
            requests_per_minute=REQUESTS_PER_MINUTE,
            min_delay_seconds=2.0,
            max_delay_seconds=4.0,
            cache_ttl_seconds=CACHE_TTL_SECONDS,
        )
        super().__init__(config)
        
        self.extraction_rules = extraction_rules or EXTRACTION_RULES
        self.list_rules = list_rules or LIST_EXTRACTION_RULES
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """
        Parse page using configured extraction rules.
        
        Override this method for more complex parsing logic.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        data = {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
        }
        
        # Apply single-value extraction rules
        for field_name, selector in self.extraction_rules.items():
            try:
                value = extract_text(soup, selector)
                data[field_name] = value
            except Exception:
                data[field_name] = None
        
        # Apply list extraction rules
        for field_name, rule in self.list_rules.items():
            try:
                data[field_name] = self._extract_list(soup, rule)
            except Exception:
                data[field_name] = []
        
        # Add page metadata
        data["_meta"] = {
            "page_title": extract_text(soup, "title"),
            "canonical_url": extract_attr(soup, "link[rel='canonical']", "href"),
            "meta_description": extract_attr(soup, "meta[name='description']", "content"),
        }
        
        return data
    
    def _extract_list(self, soup: BeautifulSoup, rule: Dict) -> List[Any]:
        """Extract a list of items based on rule configuration."""
        container_selector = rule.get("container", "body")
        item_selector = rule.get("item", "")
        extract_type = rule.get("extract", "text")  # "text", "href", or any attribute
        
        container = soup.select_one(container_selector)
        if not container:
            return []
        
        items = container.select(item_selector)
        
        results = []
        for item in items:
            if extract_type == "text":
                value = item.get_text(strip=True)
            elif extract_type == "html":
                value = str(item)
            else:
                value = item.get(extract_type, "")
            
            if value:
                results.append(value)
        
        return results
    
    def add_extraction_rule(self, field_name: str, selector: str) -> None:
        """Add a new extraction rule at runtime."""
        self.extraction_rules[field_name] = selector
    
    def add_list_rule(self, field_name: str, container: str, item: str, 
                      extract: str = "text") -> None:
        """Add a new list extraction rule at runtime."""
        self.list_rules[field_name] = {
            "container": container,
            "item": item,
            "extract": extract
        }


class TableScraper(GenericScraper):
    """
    Specialized scraper for table data.
    
    Automatically extracts data from HTML tables.
    """
    
    def __init__(self, urls: Optional[List[str]] = None, table_selector: str = "table"):
        super().__init__(urls=urls, scraper_name="table_scraper")
        self.table_selector = table_selector
    
    async def parse_page(self, html: str, url: str) -> Dict[str, Any]:
        """Parse tables from the page."""
        soup = BeautifulSoup(html, 'html.parser')
        
        tables = []
        
        for table_idx, table in enumerate(soup.select(self.table_selector)):
            table_data = self._parse_table(table)
            if table_data:
                tables.append({
                    "index": table_idx,
                    "headers": table_data["headers"],
                    "rows": table_data["rows"],
                    "row_count": len(table_data["rows"])
                })
        
        return {
            "url": url,
            "scraped_at": datetime.utcnow().isoformat(),
            "table_count": len(tables),
            "tables": tables
        }
    
    def _parse_table(self, table: BeautifulSoup) -> Optional[Dict[str, Any]]:
        """Parse a single table element."""
        # Extract headers
        headers = []
        header_row = table.select_one("thead tr, tr:first-child")
        if header_row:
            headers = [th.get_text(strip=True) for th in header_row.select("th, td")]
        
        # Extract rows
        rows = []
        for row in table.select("tbody tr, tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if cells and cells != headers:
                if headers:
                    row_dict = dict(zip(headers, cells))
                    rows.append(row_dict)
                else:
                    rows.append(cells)
        
        if not rows:
            return None
        
        return {"headers": headers, "rows": rows}


# Convenience functions
async def scrape_generic(urls: List[str], rules: Optional[Dict[str, str]] = None) -> List[ScrapedData]:
    """Quick function to scrape with custom rules."""
    scraper = GenericScraper(urls=urls, extraction_rules=rules)
    return await scraper.scrape_all()


async def scrape_tables(urls: List[str], table_selector: str = "table") -> List[ScrapedData]:
    """Quick function to scrape tables from pages."""
    scraper = TableScraper(urls=urls, table_selector=table_selector)
    return await scraper.scrape_all()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example: Scrape with custom rules
        urls = ["https://example.com"]  # Replace with your URL
        
        rules = {
            "title": "h1",
            "paragraphs": "p",
            "links": "a"
        }
        
        results = await scrape_generic(urls, rules)
        
        for result in results:
            print(f"URL: {result.url}")
            print(f"Success: {result.success}")
            if result.success:
                print(f"Data: {result.data}")
    
    asyncio.run(main())
