import requests
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Optional
import logging

class WebContentScraper:
    def __init__(self, delay_range=(1, 3), timeout=30):
        """
        Initialize the web scraper
        
        Args:
            delay_range: Tuple of (min, max) seconds to wait between requests
            timeout: Request timeout in seconds
        """
        self.delay_range = delay_range
        self.timeout = timeout
        self.session = requests.Session()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load(self, urls: List[str]) -> List[Dict]:
        """
        Load and scrape content from multiple URLs
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of dictionaries containing scraped data
        """
        docs = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        for i, url in enumerate(urls):
            try:
                self.logger.info(f"Scraping URL {i+1}/{len(urls)}: {url}")
                
                # Add random delay to avoid being detected as bot
                if i > 0:
                    delay = random.uniform(*self.delay_range)
                    time.sleep(delay)
                
                # Make request with headers
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Parse content
                content_data = self._parse_content(response, url)
                if content_data:
                    docs.append(content_data)
                    
            except requests.RequestException as e:
                self.logger.error(f"Error scraping {url}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error for {url}: {str(e)}")
                continue
                
        return docs
    
    def _parse_content(self, response: requests.Response, url: str) -> Dict:
        """
        Parse content from HTTP response
        
        Args:
            response: HTTP response object
            url: Original URL
            
        Returns:
            Dictionary containing parsed content
        """
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic information
        title = self._extract_title(soup)
        text_content = self._extract_text_content(soup)
        meta_data = self._extract_meta_data(soup)
        links = self._extract_links(soup, url)
        images = self._extract_images(soup, url)
        
        return {
            'url': url,
            'title': title,
            'content': text_content,
            'meta_data': meta_data,
            'links': links,
            'images': images,
            'word_count': len(text_content.split()) if text_content else 0,
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try h1 as fallback
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
            
        return "No title found"
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        main_content_selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '#content', '.main-content',
            '.post-content', '.entry-content', '.article-content'
        ]
        
        main_content = None
        for selector in main_content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Get text and clean it up
            text = main_content.get_text()
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        return soup.get_text()
    
    def _extract_meta_data(self, soup: BeautifulSoup) -> Dict:
        """Extract meta data from page"""
        meta_data = {}
        
        # Extract meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_data[name] = content
        
        # Extract other useful information
        lang = soup.find('html', {'lang': True})
        if lang:
            meta_data['language'] = lang.get('lang')
            
        return meta_data
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all links from page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip()
            
            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)
            
            links.append({
                'url': absolute_url,
                'text': text,
                'title': link.get('title', '')
            })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all images from page"""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, src)
                
                images.append({
                    'url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height')
                })
        
        return images
    
    def save_to_file(self, docs: List[Dict], filename: str = 'scraped_content.txt'):
        """Save scraped content to text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(docs, 1):
                f.write(f"=== Document {i} ===\n")
                f.write(f"URL: {doc['url']}\n")
                f.write(f"Title: {doc['title']}\n")
                f.write("Content:\n")
                f.write(doc['content'])
                f.write("\n\n" + "="*50 + "\n\n")
    
    def get_content_summary(self, docs: List[Dict]) -> Dict:
        """Get summary statistics of scraped content"""
        if not docs:
            return {}
            
        total_words = sum(doc['word_count'] for doc in docs)
        avg_words = total_words / len(docs)
        
        return {
            'total_documents': len(docs),
            'total_words': total_words,
            'average_words_per_document': round(avg_words, 2),
            'successful_scrapes': len([doc for doc in docs if doc['content']]),
            'urls_scraped': [doc['url'] for doc in docs]
        }


class MedicalContentProcessor:
    def __init__(self):
        """Initialize medical content processor with relevant keywords"""
        # Keywords related to skin diseases and medical content
        self.medical_keywords = [
            'skin', 'cancer', 'melanoma', 'dermatology', 'lesion', 'mole', 'tumor',
            'diagnosis', 'treatment', 'symptom', 'disease', 'condition', 'patient',
            'medical', 'clinical', 'therapy', 'biopsy', 'carcinoma', 'basal cell',
            'squamous cell', 'actinic keratosis', 'dermatitis', 'eczema', 'psoriasis',
            'rash', 'inflammation', 'pigmentation', 'benign', 'malignant', 'pathology',
            'oncology', 'dermatologist', 'physician', 'doctor', 'healthcare',
            'prevention', 'screening', 'early detection', 'risk factor', 'sun exposure',
            'UV radiation', 'sunscreen', 'protection', 'mole mapping', 'dermoscopy'
        ]
        
        # Patterns to remove (non-medical content)
        self.noise_patterns = [
            r'cookie\s+policy',
            r'privacy\s+policy', 
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+newsletter',
            r'follow\s+us\s+on',
            r'social\s+media',
            r'advertisement',
            r'sponsored\s+content',
            r'related\s+products',
            r'buy\s+now',
            r'add\s+to\s+cart',
            r'shopping\s+cart',
            r'copyright\s+\d{4}',
            r'all\s+rights\s+reserved',
            r'website\s+navigation',
            r'breadcrumb',
            r'search\s+results',
            r'page\s+\d+\s+of\s+\d+',
            r'\bfaq\b',
            r'contact\s+us',
            r'about\s+us'
        ]
    
    def clean_medical_content(self, text: str) -> str:
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Remove noise patterns
        cleaned_text = text
        for pattern in self.noise_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        
        # Filter sentences that contain medical keywords
        medical_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            
            # Check if sentence contains medical keywords
            has_medical_content = any(keyword in sentence_lower for keyword in self.medical_keywords)
            
            if has_medical_content:
                medical_sentences.append(sentence)
        
        # Join filtered sentences
        result = '. '.join(medical_sentences)
        
        # Additional cleaning
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = re.sub(r'\s*\.\s*\.', '.', result)  # Fix double periods
        result = result.strip()
        
        return result
    
    def chunk_content(self, text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[str]:
        if not text:
            return []
        
        # Simple recursive splitting approach
        separators = ['\n\n', '\n', '. ', ' ', '']
        
        def split_text_recursive(text: str, separators: List[str]) -> List[str]:
            if len(text) <= chunk_size:
                return [text] if text.strip() else []
            
            # Try each separator
            for separator in separators:
                if separator in text:
                    splits = text.split(separator)
                    
                    chunks = []
                    current_chunk = ""
                    
                    for split in splits:
                        # Add separator back except for empty string separator
                        if separator != '':
                            split = split + separator
                        
                        if len(current_chunk + split) <= chunk_size:
                            current_chunk += split
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = split
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Handle overlap
                    if chunk_overlap > 0 and len(chunks) > 1:
                        overlapped_chunks = []
                        for i, chunk in enumerate(chunks):
                            if i == 0:
                                overlapped_chunks.append(chunk)
                            else:
                                # Add overlap from previous chunk
                                prev_chunk = chunks[i-1]
                                overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
                                overlapped_chunk = overlap_text + " " + chunk
                                overlapped_chunks.append(overlapped_chunk)
                        return overlapped_chunks
                    
                    return chunks
            
            # If no separator works, split by character count
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        return split_text_recursive(text, separators)
    
    def process_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Process scraped documents: clean and chunk medical content
        
        Args:
            docs: List of scraped documents
            
        Returns:
            List of processed documents with cleaned and chunked content
        """
        processed_docs = []
        
        for doc in docs:
            original_content = doc.get('content', '')
            
            # Clean content to keep only medical information
            cleaned_content = self.clean_medical_content(original_content)
            
            if not cleaned_content:
                continue  # Skip documents with no medical content
            
            # Create chunks
            chunks = self.chunk_content(cleaned_content)
            
            # Create document for each chunk
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    chunk_doc = {
                        'url': doc['url'],
                        'title': doc['title'],
                        'original_content': original_content,
                        'cleaned_content': cleaned_content,
                        'chunk_content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'word_count': len(chunk.split()),
                        'scraped_at': doc.get('scraped_at', ''),
                        'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    processed_docs.append(chunk_doc)
        
        return processed_docs
    
    def save_processed_content(self, processed_docs: List[Dict], filename: str = 'processed_medical_content.txt'):
        """
        Save processed and chunked content to file
        
        Args:
            processed_docs: List of processed documents
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(processed_docs, 1):
                f.write(f"=== Chunk {i} (Document: {doc['title']}, Chunk {doc['chunk_index']+1}/{doc['total_chunks']}) ===\n")
                f.write(f"URL: {doc['url']}\n")
                f.write(f"Word Count: {doc['word_count']}\n")
                f.write("Content:\n")
                f.write(doc['chunk_content'])
                f.write("\n\n" + "="*80 + "\n\n")
    
    def generate_rag_csv(self, processed_docs: List[Dict], filename: str = 'skin_cancer_rag.csv') -> str:
        """
        Generate CSV format suitable for RAG system ingestion
        Args:
            processed_docs: List of processed documents
            filename: Output CSV filename
        Returns:
            CSV content as string
        """
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['category', 'source_url', 'source_link', 'answer'])
        for doc in processed_docs:
            category = self._extract_category(doc['title'], doc['chunk_content'])
            answer = doc['chunk_content']
            writer.writerow([
                category,
                doc['url'],
                f"[{doc['title']}]({doc['url']})",
                answer
            ])
        csv_content = output.getvalue()
        output.close()
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        return csv_content
    
    def _extract_category(self, title: str, content: str) -> str:
        content_lower = (title + " " + content).lower()
        
        # Define category keywords
        categories = {
            'Melanoma': ['melanoma', 'malignant melanoma', 'cutaneous melanoma'],
            'Basal Cell Carcinoma': ['basal cell carcinoma', 'basal cell cancer', 'bcc'],
            'Squamous Cell Carcinoma': ['squamous cell carcinoma', 'squamous cell cancer', 'scc'],
            'Melanocytic Nevi': ['mole', 'moles', 'nevus', 'nevi', 'melanocytic nevi'],
            'Actinic Keratosis': ['actinic keratosis', 'solar keratosis', 'precancerous lesion'],
            'Seborrheic Keratosis': ['seborrheic keratosis', 'seborrheic keratoses'],
            'Dermatitis': ['dermatitis', 'eczema', 'skin inflammation'],
            'Skin Cancer Prevention': ['prevention', 'sunscreen', 'sun protection', 'uv protection'],
            'Skin Cancer Diagnosis': ['diagnosis', 'biopsy', 'dermoscopy', 'screening'],
            'Skin Cancer Treatment': ['treatment', 'therapy', 'surgery', 'chemotherapy', 'immunotherapy']
        }
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return 'General Dermatology'
    
    def _generate_qa_pairs(self, content: str) -> List[tuple]:
        qa_pairs = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Group sentences into logical chunks for answers
        chunk_size = 3  # Group 3 sentences per answer
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i+chunk_size]
            if not chunk_sentences:
                continue
                
            answer = '. '.join(chunk_sentences).strip()
            if not answer.endswith('.'):
                answer += '.'
            
            # Generate questions based on content
            questions = self._generate_questions_for_content(answer)
            
            for question in questions:
                qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def _generate_questions_for_content(self, content: str) -> List[str]:
        """
        Generate relevant questions for given content
        
        Args:
            content: Content to generate questions for
            
        Returns:
            List of questions
        """
        content_lower = content.lower()
        questions = []
        
        # Question templates based on content type
        question_patterns = {
            'symptoms': [
                "What are the symptoms of {}?",
                "How can you identify {}?",
                "What signs should I look for with {}?"
            ],
            'treatment': [
                "How is {} treated?",
                "What are the treatment options for {}?",
                "What therapy is used for {}?"
            ],
            'diagnosis': [
                "How is {} diagnosed?",
                "What tests are used to diagnose {}?",
                "How do doctors detect {}?"
            ],
            'prevention': [
                "How can {} be prevented?",
                "What are ways to prevent {}?",
                "How to reduce risk of {}?"
            ],
            'characteristics': [
                "What does {} look like?",
                "What are the characteristics of {}?",
                "How can you recognize {}?"
            ]
        }
        
        # Extract key medical terms from content
        medical_terms = []
        for keyword in self.medical_keywords:
            if keyword in content_lower and len(keyword) > 3:
                medical_terms.append(keyword)
        
        # Generate questions based on content context
        if any(word in content_lower for word in ['symptom', 'sign', 'appear', 'look', 'characteristic']):
            category = 'symptoms'
        elif any(word in content_lower for word in ['treatment', 'therapy', 'surgery', 'medication']):
            category = 'treatment'
        elif any(word in content_lower for word in ['diagnosis', 'test', 'biopsy', 'examination']):
            category = 'diagnosis'
        elif any(word in content_lower for word in ['prevention', 'protect', 'avoid', 'reduce risk']):
            category = 'prevention'
        else:
            category = 'characteristics'
        
        # Generate 1-2 questions per content chunk
        if medical_terms:
            primary_term = medical_terms[0]
            templates = question_patterns.get(category, question_patterns['characteristics'])
            
            # Use first template and create a general question
            if templates:
                questions.append(templates[0].format(primary_term))
                
            # Add a general question about the content
            questions.append(f"What should I know about {primary_term}?")
        else:
            # Fallback general questions
            questions.append("What is this skin condition?")
            questions.append("What information is provided about this topic?")
        
        return questions[:2]  # Limit to 2 questions per content chunk
    
    def export_for_rag(self, processed_docs: List[Dict], output_format: str = 'csv') -> str:
        """
        Export processed documents in format suitable for RAG system
        
        Args:
            processed_docs: List of processed documents
            output_format: 'csv' or 'text'
            
        Returns:
            Formatted content string
        """
        if output_format.lower() == 'csv':
            return self.generate_rag_csv(processed_docs)
        else:
            # Return as plain text chunks for direct RAG insertion
            chunks = []
            for doc in processed_docs:
                chunk_text = f"Title: {doc['title']}\nSource: {doc['url']}\nContent: {doc['chunk_content']}"
                chunks.append(chunk_text)
            return "\n\n".join(chunks)
    

# Example usage
if __name__ == "__main__":
    urls = [
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG6IH3ybChsMwVfSzqmAD4GQRi-ltwXX3wAG3Ai2sfAKDL7uSxPROFXsDfp72c19geUuX8U2Fcj2ei8iz8jjKAZWrXd3-AQRAT5DbxUNzWPqGBwSaLzPNsl8jKRVtE3B3dEqVO4-vK9Ri9e9E5hfnB6wbVuVA==", # Yale Medicine - Melanocytic Nevi (Moles)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFFBRToYcXNFFDZ7OsXKU5GmdItjoYnKBZxj9ya_md6VfHMoRRPo8aoKSbo92D__dDcJmHu-kqiwhE-dcc-dofvZCjh2xjhRz0OrQprSNboThwBtYuONlBgoeubcNYi-ZeHqNJm8g==", # Wikipedia - Melanocytic nevus
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEp2foNAv7x1oyL8u3hzPzaxD2Pnk4ggH31MWlYlG9KUDKS9ABUZeWXyJagEP2dBpHS09yWva5EQNAP_xCL-XW_yIfJQOp-X5r1_co8v6q5Ud_hpGFFL8gdaxxFII_iviJP1r_A8g==", # DermNet - Moles (melanocytic naevi, pigmented nevi)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEOZwutAgxcu07s3JjsfEyi2Ra1LH8e6R0tKn-HUAq74NSh247ZXZy_ZbxwKqGCIrvTNTDv7eZKrRfb2WDNd_rSurQMNwpal3nJYXrR-Q3v9FNLTZTprp4UkjHLVvjyOicW4pvO6HFRkcd8XRrl", # Medscape Reference - Melanocytic Nevi
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEygtcUb6s5KoDujzdlnw9X-mVq_RnS9C4_-A6HJV3-dDNlQ6-wNcyh19HBAunTVpZ30qrbbl8Ckb7g4mszIE9CHu6DYr4o6QyJEp0ccvVvqDJp2c8l3Sbef-98EtZ87pvej2gG3y5GNGCix6xVIKq-hFb3ENDnF595ShULGifYxHCC-FKt6xg=", # AccessMedicine - McGraw Hill - Melanocytic Nevi
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFFLsv2jWYBpZ-sgKjAlTT7pIa5PuKOqNCKHLytgaEiHxhUkFI0lMIRzpN7IoDBbCRm3DzU0-8vMhDC9hADskP_FE5STXleWlrpAaGD2AAhxlN_dEKWcrf1JDt_QxHuZ76JxDzL6g==", # tamanhhospital.vn - Dấu hiệu nốt ruồi báo ung thư dễ nhận biết (kèm hình ảnh)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGs97d8eXHzKIE2jQ1F2JBFJlEkGbbyJw0g9kVJg8-EblbNvunCcre40u1uq8SVeXSoOSCTGMJziKS9_7uuIkBmE4HC-_JKogSmHvO86QpvW6CXtlGXQS9HAZlsjddup2O0OsK6O4fIK2kMbC0viMYMGmKN", # Vinmec - U hắc tố lành tính của da
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXESxPysRqTFri4RMVRDjaz7KwwX2_l5QddwVZ7rTzo8opMQ4feGO-sqJ1Sl7FxPpvBhpu97I29B-Tuz11Gry_8RdMZpnKG3Kaa9jAkM_ix4KjwTHbA2ijC9vWer0zRAM4QilQZ8E9q1HDDhJNR2b7v28f_M7AvGnT5q7fYQ1YBgxsgZvdFtZH-DmGtKB4zdehSaX4KZPiVw9iJdHm6UiShO_W-YacsIvFOuc4P7XblqKDBMPeldkC8tfgdpHCSUVYpsYjNpYpRQ35UvJp3aPUlLYS0=", # Cẩm nang MSD - Phiên bản dành cho chuyên gia - Ung thư tế bào hắc tố
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG78SnsHkxj3BnriHi_vKSAZe_Qmh3KRvdasPZA7W40aS9R6mz74a7MM7iZO-arJaH3YR9LmyVC9SbQkPTgFjocUi2glRGCzsu58L5e2a7AqCOuJico4aOl0uWQAEZOMK9pZ1x_TNEyA5sArxcpMr3C5FIYWaR63KeyokcGcgQ788dN8FbbX-ZehHfiCQJmhrwl4I7zjxTe", # Long Châu - Dấu hiệu nhận biết hình ảnh nốt ruồi ung thư hắc tố
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFwvXMxWGk7lUau7doJ2_VYwcFWJ5RGIWvAuB3QgCu5le1NzAW54u7vQeY_EySkH4Yp1Iyk1Ad8M5xbuuBs_NajmxCr03s94rbIqFjPyUcrIBuqqWfsKxGmD1_WXZubbur8qZBb7VO1zms=", # Pharmacity - U tế bào hắc tố là gì?
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEbqvAkKCZ77GbXgR3k0o7BzZJ2C_Jhl3Y-HhYubCMubHnOLwfbbJYgUKuukQAPk5nOd3E_zQ6q6Xqpes6CjzS7AKJTed1LfRf5O-5qwZe6UPzqFKv8_F7x4ZYGhfp5vUTzTkvkvz59w4RB1cVo9acBsQCIoEgb8IpFBtygjh-FcYQFP71h-fYAEBs=", # Mayo Clinic - Melanoma - Symptoms and causes
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG-unu0B8gxBMGOOEzXbA0KZAzDBRbgS_gT8EHxVCigXnM32BvcnsX3g0NXkDBGHrD9H6x6M_wJm4blZXgc81WaQsoELNzOm03hIPBoIMpkpnWb8dcPi0ckiYENMA==", # Wikipedia - Melanoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGyDocJhRSkLFHuvPnjS7b9DwF3ZOuZVvY9HkyNuYzEUCC4if4NdHP_tqickCIVIoZW0plesbk6MjYYrj5qeB7bRS5jXRJdIAMyWvw4p1mJDpNSfOCsafaQFOO1wEpgdJsULBIHVUtrmmcllYnml_Sf4MwN", # Cleveland Clinic - Melanoma: Symptoms, Staging & Treatment
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXH4i42rz0CAwOykxoByOuQWx7YY-AjF9koPLj7fw_ia6eBbAgK2x8fS6xUDDp5OwAJpmk7FuuVxSuaimfQNAVPj-CmA0I7SfTNxtmjOjwrSl0T9v3IxLy6AUTU_J4tDNSYwhEiJJCIbc0vfaRiDZcg5gcg=", # The Skin Cancer Foundation - Melanoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEXrCVTi1dFy-rwdQ57EK22CXp2W7wJv7Z9hNfH4U475_7rok5tGtTMTqnHPdcMqxAWvw9AnbzY7g_G60-PKyNZIyrs3dzcDMjKa2e7a-LmagcYNgS8UJs30HMnqJ1bVlVksGNvK29oHVkVJUY=", # Cancer Research UK - Melanoma skin cancer
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXF-n_MIt0A42vLB7EyfyVSOlHvRqNexhVV8ZDbSS7b8L9aAI66ZrXVBBaAtyOF7ccxUG_gWPYZ6kayekCt5O5LZDT8lzbif90c5AjhZSbQLH4qlikLvyWHrrFPmuHHUb11XAT1gR2BRkPtmRg==", # Vinmec - Ung thư hắc tố da
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHPrfGOVGHQZ8YRjF-cFbZpRpp7fC5Ph9oGfyL56g01OC6WxxKJaexvL2xpoYdLZEHlZ1W1pDkGAU4yANn6J0pZgKwYMKyTl95pJRbhx2HvjF1_lg2bNJWxSwi0KUlv7uU0s8YmJ7PL1e69radxs7EKXsU=", # Long Châu - U tế bào hắc tố là gì?
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFvUKmVNQ6gKKkKIixeHcU8urYXLw_-SBqiwvLQA2KHtw8cZtSx6Im5BCf8tdjrRfJev1LuJFTDH6I6l5WPr45yo86gA0sZ5Vg-ezRaxIdXRnRMfvvTUOuxrFi6z-GVN30eYgnm5P-k33OxLIudEmR-rZssdisx", # Wikipedia tiếng Việt - Ung thư hắc tố
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHtLQLX9ZY-NVua6olTTCYas7oeLeyvUvOynOXAl1mCmcf-sPodkQKpLpdtM3B4ZLmUod9sXDvSKWqTXu_N1dVJoaB4UgMBvrq3iKgtOdQTsENUV_sz4kaW_bYnLiGdHqorGZdijSxnjlJm6qeDSL28eHg=", # Vinmec - U hắc tố lành tính của da (possible duplicate content, different link)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGTEjQ8Pw8Yxg-xYJOr9Ru6m_GDj2EqjzcXicIYrQvlz8d8hd5hVUDGWjPL4DUa_iegg8Er-a26z1ER0uWem7gH-m6cIajXojKzjA0_yidGsfIl12sL2jydHgo3dSM8svbLCmawfZV_4f9MWfE58pAjTROrDswXNmmYWhr0HDSSsNKjIRiY3N51EzxQkyNTXAuOGk0FwkgeyyE=", # overseasvietnam.com (Singapore) - Ung Thư Hắc Tố
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFVkew0SGtZOl4vbHMGwFHueKy5WkXmrGWiqaYb1nVqGdwYP_t1dcaGc0Pv_ihsldzaa0eReN2bWh4POmeYs1EYpvI3ztNgIFzzpsVpJf486WKwOwOXjqTOcYI3BdngRRnDvBTrXZCbgw==", # PMC - NCBI - Benign Keratosis: A Useful Term?
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEy_19PGrxL5XDaqVfU6lEpKXXPBuXCi35FpvZmhBc6461TA_fYsFeY_IeFEvi_XLw_s3nc_ihRGearJ9T3oUnJ0uhb8vBr8C1bre78PKfDn-pOa0fb-mzYJyFRxEn-pTiHpOOSIooTCw==", # DermNet - Seborrhoeic keratoses
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG4Bbwams0kqaowtjh8qYo1NyJ2Y69c26utVMhMQHNKBvccXPzkypxOYdFKnt8pqsLOaltLsvh3hKMUUuAaCi3ZD27BGSCNKTnpRdrAZuD1qqBTQha-ydZN1deQaFoj2wzMhlCum13S", # PMC - NCBI - Cutaneous Lesions Masquerading as Seborrheic Keratoses
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHfzaNppxTJQYRtkicqQlhUD7bfgLqrdc_cKYCO4NkrZ3CCi6dbQml-LBv096j8ZsLSW8v6k4si6_7yY8fNr0DBgF5dIT94km8jSlHpwrAUZDZ74b5UdyO3dX_8E2xJUFEE7_79IhE=", # DermNet - Common benign skin lesions
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGAs579alJpQuvMA2-CW2_dkIhyQ7y5tKjPHyI0ds2SVFYUxoKyPJiXm7jeuNp78ei6ijjTiLR8OQdOJUribXfuRbUjcPtVd7qE_GL_6zsb3KOJtEf3cl2-56kXmnINN3i9", # dermoscopedia.org - Benign lesions
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHDJ1RYJTybapiEmf_IBgpwbYmFV6i5Vm3BSpOjXEhl0N25-lMKbTV7g42KYdQfF3JDFrbDW4km6LeqiUxeXj05l-552gUCh5qQKczRhjYkilmSNDRuhh1S2unMdsYanjHcApDog-lJzaSyC_kO8u60M5C7Z9dWfOs9INSbHT7eFaPRQkyYgcH_Oe4GVqDVHdxw", # Medlatec - Dày sừng da dầu
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFm55v_dfO0FXirWjnmFqn7rCGi8lKE85AstuNX1TfY65nlWubAZdO1BxTPUwCUtlc7h6Y5kpKMTOOkQayVptWzj1QoXoS4b01idOpP7HxppZPS-yVxLDEMVeqAqtluxPkRBSyHKhiEP4jjyBTziX4DAzqrivu0U9dUf-G6_A==", # Vinmec - Dày sừng ánh nắng
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFWYISvsWdHKnr4uXaJTlxgSQ3V0fnyZnkAZ5UdPp3Vcb6Px9u-riuG5S7DssT0DxShwSv95ekuOyGr-Y9vzEgRoLr0vpLLgktpksAcJkm93hUNJOu0t5roqkML1TGPq1p-CvYXvKk2ZNP_IjdGmW_Pco5nDujJqj5rROP8oUK5CA7onZFJbgiXXA==", # Trung tâm Da liễu Hải Phòng - SỪNG DA
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGujypxdcCvyYmsuNvVQSUUbEel3kPI7AcjiAwdiBllwvA36Jcz8a69e1wjObeDojjglctMNwuYXo5-G638GHKV6i6Pkc6HBiQMTYWiQeZpQq8IgXb6Ei_XJbR3hSxEi1_ygQ4M2GQrBw3Kxs-mPq4Pd1bsfgLhqSC2ITomrsLUcf4=", # YouMed - Dày sừng ánh sáng
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG1HQc0ehl-xJQASiVE69IY1QLAgct-9i71wRjBW24YD5p4hfcVEz577QOq6J-EWQbsuCX5TX7LBjdClGV12KSSAsehqRLO1fmFYgmVF-pyvSxLsiEw0THLgd7KB-RW6RlulWfb--qkvFII5hwkmvKpQcOFHr1yZe486v0QQbDl03gsn6WR1TM0nXrdx_jJZ5HVaOYV-_PF2bVp", # Long Châu - Dày sừng tiết bã
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEYaSrnnhZzy56Q2pqDmCxnl35jw4oST-PAEgeF5YujJP4xfJUuika3b1gt6tB8vH0fz7Hec0-tCZaAh-qIdlmc-OmUiYkFIp7bWqbSMqEI4wVIYwDU4s9i41P-zs_7C67OZSGwlzwB3_4l0flz4-yv8RLoOYfgeC6HkqdNc-bK0YRfZW33dsnSXD3dTO24JxLn5rgopw==", # Mayo Clinic - Basal cell carcinoma - Symptoms & causes
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHrmlAw7v1In5pdeV9bncCmqQm0EZqpBynOlvfklqFZQFiW1vEpgFHDpXLVsIPnYpn55OV0NTWjwd2wFpCpma58vFT-wLvjBEtsbOtzk0u6lmkd9eA9kHV4tNZF-pNVJJVaxPWdq3OJWaPcoJc7Y7AUM7rBTb0r_qZSTbScDw==", # The Skin Cancer Foundation - Basal Cell Carcinoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXH6tmjg7d46TQxhraGvOixWrUBwV1F5JgfkyzHeAJ5oe1Z3TCYMbEv4HJJL0tjlHAXx7TldMvyIhibUCpyiYk8iZFV_ABX2DW6N-AP4ttUI82kH9HW-hQEX2zn1Sm3oOE56QwfHXZ2U", # Wikipedia - Basal-cell carcinoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFBY8qhKwDEt_nF_zn-v8L5-1_GYZSz0tKNYhgP7vfqTpG5xbcp5CP-pPfONPMt7Tf-Jt6CvNYjMTWpuACJbfudqeQGyHXUaiY3ZmmP35pN9SPtn3M3R4XCds6uR0FDQRg6g2HcU2NL2EZbFDNT4gF3koZByY9K5NR2NkknZg==", # Cleveland Clinic - Basal Cell Carcinoma: What it is, Symptoms & Treatment
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEKvvdsVHK0xx1XWQygnr1oG0GfeK1fAAqm-r_leu6ynhqncrFPTn8DVQieLjI8KRUw_LggvVCsItjdBSHdT1obGT4riDyvng4jU7WQnHsw7npwHZA2lNKc5k8FUsEUEIRVy06kva4=", # DermNet - Basal Cell Carcinoma: Symptoms, Causes, and Treatment
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXE5r9QQHWzdu669FyrbluZKUaHG4R3wWnAE6D0wf30FtLgTSYQZjaYbl3VGDpIfDTzd5cu9kt2GSarvTvJiQn055t_83GexgnFy66TVFaR6IT5MuTApwr0rl2oqfTOCzypoOCd1iMy3DpkyA7XI83EWvgl8ywiiG-UVSEJdIHWmbrW8k-_phxgKlvB3Uaw=", # Vinmec - Ung thư biểu mô tế bào đáy: Chẩn đoán và điều trị
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG-9BqaqO3dfTUUNcEHFetmjmxZjae_BdBIuMOr1-zA-I2h_oI9fc9DQPaM8hCc_epSWPNUu94eIqAH5yzP3E94Opg7QUsZWajtfghUP8rS0QhiRO3fxoeYJ67NdFHE_ZgD-zSfcB4Ksi2AOTEv19QIb9li3CcyWj_NWNsQChXaz4ubeqyzUqSVxmX5z5M3gTBXgD3hPdym5-EyjKia2JwyrgqxgUmb", # medlatec.vn - Ung thư biểu mô tế bào đáy (dated 2025)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXH5US-el8PuXlzYk0ljy87v6UxzqGJeGnGV8FxRGtZZFEZYhYRaxwjFdJ_Sj8UTS23TqtTB6cZ6CBTtzxrtheR1VedseGP8esKPMqb0xWPNi6jb6PElLJ5Rs2sxKHohPIAXzdcLMAww-jrNC3ZXYqiYvfVAn5Ej9kFJPUtCuEdfZK11_LlBOdy0E11kkzd91ZdEo0JHfLrTQEN0dTJO5Q56r2gdLA==", # Long Châu - Ung thư biểu mô tế bào đáy: Nguyên nhân, dấu hiệu và cách điều trị
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFcjD4hUoLxpfFlBmAYleZhiKhANieIb2A2g2LCrD4sF51D8Uun3y3LRtdAeUyVtDAJrLFDgycC7D3rREjZAeLGwlETtXxOOdltpVJZz09da0hNlCuKRTrou-xPbEbw5Ji5G-j7611VuxhjiyCCsjOwYH3G3xQ5MEmnJ9df1PMaXWcU6Q-9JiTZuAHBmigfDtCCeDup4A==", # Bệnh viện Da liễu Đà Nẵng - BASAL CELL CARCINOMA
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEEY4uCLaP-gK3KdrDNJEgU_d9SgYrNPDTN4n-gnWcSPwxNW_p-MhJMxMtiT82A282lcM2_NY1o1oLAwbaKg7eknLU6ghguqBWifBKUApvG9BiISoFi3r26YMoEiU8y4rwPQQ-49GX_lx-tMPbjlktcTHJk1jDhwllwsA1ahAcbetxTNQ_KRLa3yR767N7O_IR3dnerJbgSmKXIlveZjZYJuOBAE-yvL9EfNzAnS8w63bTd-Olwfep-AQsHR9Di10It__KlB6o0KipwVkpKVVJpJZaaGFd1fkQFn3PCDjw=", # MSD Manuals - Ung thư Biểu mô Tế bào Đáy
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEmWdXUwthE-cgFS0w86LVzBgNbdCM2ww1miuRJ00PWdsxSdRa3S08gLEnjrZKhwViGLEy6AYEFZv2MRQi0GcvQpofWgYivv7G7LjKmIFPJuKhBfqdkZHZvb_0s5KTCuozqzChtBA==", # DermNet - Pyogenic granuloma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGhFu0OjxUQvogKvDLHycirNXrG8JfXO_GJXI19iinzN9wK9hANLoTW-i-mNOIcyf_ArrVeOFbpxCDI8vuPRqsuoRFgbcSJlBRYfDOzrqATXHtV87yWtVLqHaDfz6-if_0Wd1M2xLrqzO_F3CNhXl8GtGsx4IYnDobEDkZe7A==", # Cleveland Clinic - Pyogenic Granuloma: Symptoms, Causes & Treatment
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG8RGVSct0eIL9LUMoJft9TuyFuVjrVT5VCVPwlZwL0gHXF0pi0BHor3pt_DLEc2jTJuqh58929rVRFe3BpEpciHOUi_8_lxX3YU65n3PuAD9yHbnHyQE22tTfwUv_u9PKtjlJEgA==", # British Association of Dermatologists - Pyogenic granuloma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHrzXMcIiWjRb3NoGMC1AWv5miNubUem9N11qIDj027iI-9XS-B8a2Ujxa3Q1kD2ufICsrvUDnaqDrZ0RXRAjTPMnOuLj4y86Zjqt2SwUC1qAZH6bPGwH_q94DI0NUhV7LccTNx-a7YeXTO", # Healthline - Pyogenic Granuloma: Causes, Diagnosis, and Treatments
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFlb5knIww6vQFtIr8vZ2fNaqH7sstiu9PV3ZjQovBehjC3iocEmfZLalPgmC-147ZTarTuQAiKX_DK5NKvyFaTlpb9R8JtEGh4iGblTNmiq-YzDO6wHVH9MMO9oMVG3guZV4AhbFXbjz9WEto=", # UBC CPD - Pyogenic granulomas
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEJ5m8Z3Ov2n-ncT-mjwxS88mURvZ4KKuwy9nLPrLlYNzJAbcTXJvS3542EYS_01TjIzZ-y1iyPAD9guhgmlUgiOkzkM0gOMBFGOKC14vED-5J7xqQc0tnG3S69a91K7XqFsOI6xBulsix362qpKG8lkA7_kkMjMa2qcB2GpxBd-mMywQN14XlRNncN6qsa0XG-iO6Xyyim4SvmnRYr_Wpmjw5dmA1GKQVSLsYBODd-iQLCcnYCNDgOx7AYlbT55nSpuuvIS_rJ9DQMG6f_r5a6RAdvjPZGjgWDi7Ag8Ut3ap0mkM_zECTePS6qrWLQOaIRxJMV5ySVq3UA8PhPS9MfeQlrWCFzjBFzVQ0JrcVjcAa3vO-V6EDWgsZaCx9x0cF01b73P2v9dzCINhbemQKsp0Tvuil1ckJrAJ585Xwpvvg-arkWbeM=", # Cẩm nang MSD - Phiên bản dành cho chuyên gia - U hạt nhiễm khuẩn
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFwILK-idpzR0f2aR9tP7gnilUddLq0qEztWIk5x-C3lWp4X_uRqmyabARxLH_oiqqHXmlMk2ug9iKAm7JRmP-X3BV6XxXAATrLbzJL1PsSLunc_7jJ7RuIV1_OwxWKhzXG4XZWFi5tc786y7hP-z4U1GrO_8yrtCiSq0oArtckzcfy4SfdXMxWoshBjpaqPDv8zyRyoBCikcc=", # Long Châu - U hạt nhiễm khuẩn: Những thông tin quan trọng cần biết
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFVgkuBC4Gv-hJ-J3RN4CB3G9Yz_g-11HK195AcekZ5GsmgazrRB2-LTxrEfl9nLLve0R6q30BkrzQUL5jL8GpgbrLVZsrgrYDxhPRf3mDgMukq9sDSNuPOEcRrRZ3YXSFgksBARdUWDA==", # colinhong.com - U hạt sinh mủ
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG9pMwZdDTnBpbog1MIU0HOLgF6-7rUzVyfeCz78CzrtLEDGsv9SXyaXuxe3dBcCBqLh9myJcxk8XZQM04ZebOoToCU4el3cPrMMcVImg5Mz7WmLtDtD7_eI-z4JsaFZJgylnB30KBANdY=", # Laboratorios KIN - Bệnh u hạt sinh mủ
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFTPuz8kRxiPeckjMtmZ3sTJHFjkJ8vVJa-VrFTtYXaQJPJDANdnrsTBZbSP5-mi9eUsa_bTdpsf4otb6o26C9cy1NYheuwb5D7axQQiQHI6C20Q1Dr9OfFC3o6yM8y_65B1SB0R5SEYNlmOs_4n9J6tzdXFLGE9l3bbBGI0YVY", # MyPathologyReport.ca - U hạt sinh mủ
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFCK9HHxb5YbKwq6KOcpAoMJOiubdjyF1CqHSo38RrUehpwutu_x1pwReTiZdluh_iUR6LDwUuc72fpu8atCP5mblu0Rnn6XWmuruKOIvPKhhljB_N3OAUusPceH9OlhA==", # pubmed.ncbi.nlm.nih.gov - Nonmelanoma skin cancer - from actinic keratosis to cutaneous squamous cell carcinoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGRMyUVBGpsHPSl3aFB-5aurSS_CR3L6srAMnw0M_QJ-jKJ9ek63e2rClZQ8gdedTfbxfzt0TK4EZb-JeqtX0_nyE1EaxPwO1jRvB-TzY-kEpAQ1OIbjoVegeVthB2rPLsnphAl", # DermNet - Actinic keratoses (Solar keratosis): Diagnosis and Treatment
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXH7geUr3dCXjfNP1o0QLEA8mWEp9WAFsgz2iNWp35fI-j3zspMjBinlDHa7DqC7RRzeDCjjF_ESAZ9JCds-B__hF6hcXKhaIVKOzue_YqC05mdEi3OSaXqJDplYGNsFIi890PE=", # NCBI Bookshelf - StatPearls - Actinic Keratosis
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXERaobZn_DD7zHPEFlDumV2-wRQgzvnuU5qcZEizySXh1fDW7tyfKFkx-V6TUAQixhovgvdd0gIgGqBxSvlDvfovuYQNl1SAUJaSt0MN5J9V_LzYd_nYnfczSnh72f7tJshvXWRCyz9Fa3cdjiAkxuGHG6kRLF6b8l5-CYwxfpd", # SciELO - Actinic keratosis: a clinical and epidemiological revision
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGLNtMHMDB13kyJOa034NZp4aUf92xAq5Q3gEkjX6upd0A2l66ky0dI2E9xUnZmFTaZKryOECvsUr6Be3EQ5OzmCcH2e8nt6kJQvm4CadMbDH8hDbBge5_MLpRU_rTAehg6Lh9xnT2zBU8bp1REKmQGCmP60zG-awMarawRTPMYVwkAeiXZd_4RnyzXDo37c7-77ju6t_KUCORCSRR4oWw5XYZSalxSv-y7MA3yVbYOFTZVTmpbBGakkUe6S_TPafqlmAQTNl6Slnr3GAOc", # JDDonline - Dermoscopy and Reflectance Confocal Microscopy in Actinic Keratosis, Intraepithelial Carcinoma, and Invasive Squamous Cell Carcinoma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHdLR3NYF1evyXeNAORRlNaB2KFmsyAu3rjSiCcd3zNE1CkC_d5jg6p__vqgi56WBbvloZI-FdHkT-PphCrTr7TkK3SKjDrhDSPG-YBbeWTJ6UnPeU_MYq9OJHEuSycxp0d-Xks9tpy0jl8YrexpiCPZfDGxE_WWGI=", # Bệnh viện Nguyễn Tri Phương - Dày sừng quang hoá
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGrKFB-jaLc3xqwmQCuRAMhFNX3GLCNRUgbrw2uh-5gXH2l186YRASCADsgz8_mmZfRN0DVOgP8oeb6X9nIobzlQq2p3BZMLVz55qkWtlszjFQnW97FA38xyESyPP2sx9aitOij8nphuJC-XsBTdV6_gT1dBv389y53-dMUXo0s8KjFZwAiClnUSb-H1nls3LP5yYv8NQDxEOyuWISyVRmmDEgP8P7irSOEzdUrJYuQA-A11C1uRHmdlEQWFeAMVLzJqR5XgpvyKpwJ4Ncqi2EXm-W3aFhE-e9z_l2yLzYCppxvyUq5Xug9RFF686SZfNTR7mSxCuntMXigjTXLhKhpcX5LiA==", # Cẩm nang MSD - Phiên bản dành cho chuyên gia - Dày sừng ánh sáng
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFpscICt5VoYJIiEoEUqRDtlcpWG4vQ46jKmeW9Z5H5SRNXw8TWIwK-kVKIiaaAEwk3ViVD_zelagULQKaC8uARgb-bOPVN6hZVwXnP6s4mDLYUSpOiMl2LBSY8njw19CLEDRc2nkH8hoQJVStxusaWrNNhOVO7n5uRjnRbTATVw5MEafxofg==", # The Skin Cancer Foundation - Bệnh sừng hóa ánh sáng
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG_7Ol88VlYvl4Pt6hTdpsX2jCzwqL7hv_S4NvAMb5M8qNKq9VycGg6PTDGROH0bNKvl11DjEylAb_LsaR8tqg6mIuAtV_mvXk7BgiwShc8ecdjdyiVv7FT3mLfd3qvt7deWE_aSbWPre8qo1gukv6KhR0vFZIOhLMYWir6", # Medlatec - Ung thư biểu mô vảy tại chỗ của da (dated 2025)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGtPibK_I3ucfJ3IYU6guDyLvCPPf4V1mvYMe5idiNbyQhBP_SEqZ7wuMXm5CP3RBu-jhg1s4XNliw4a584pOmjOWXXIP5fyfVuPzrBPVjwbPHyOOVdJVUu-roFRQC8myMDQHV9SiZ4jOy_8EiGMkHpbrh9o1gZtae4D7lw3Q==", # Vinmec - Dấu hiệu da bị dày sừng ánh nắng
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXF6rM1FW3KOc9XKPTULQOKyW6zhelOw4spq-nYuzMOMFAlwXd2YbecOUE_0Y83yOH0_1HQ1tesLS0PtzgmUF5iWYXHWYosn161_7mHWyN9qRDUYXG-cXxIkIK1GlC-X8751", # DermNet - Dermatofibroma (histiocytoma)
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHpdFBF2_VKv9ikmk8vXGD5FawaDcPTDtzlthaLG4CXneKkUFckev5CBtShz-z1-SPgf42DDbvbb_ytnIafDaxVnLY16cvMuGT0T1gX8Lpr_O9mPsYdwyRHfnmxEy0AQlAx90Xqn6HzIhlsF6lA_5dZRTiGXN5C4J4uxMk9", # British Association of Dermatologists - Dermatofibroma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFIjRDhqQY7XlI1liIk4pIEhCVSJ93SSRgf6K_ERdiyQf1D18ro02tcdxcwb6_jQFZL9-KwwxXb60OXu6D2YBXFxVv5L8gf3Qc8RLYVBw5jv7auH1xLs7FAmhSgmFITu6HqKQ==", # Wikipedia - Dermatofibroma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEfe4XZvhmYiKmxUCapufWYbg2sKKBJVfXdOpBWxeOnaQ0Oczb_qW-clyho__IHCC47PTiC_T8RK7sDheyV06JbpQjvSC2uPXRE-x_fWkd-M6Z4UM1AJpvT2tBztpoPtrSc5Rba7bxSvanXd7P_mMRuvhJtCKod-btEhCZUoQ6a5Kap9e4eZ-cQgWFE8tS9ef9uAZazoqDKUMR5yvMYa6Ylqdq2y_YMcVNDnA==", # Newcastle Hospitals NHS Foundation Trust - Dermatofibroma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFRyYuQR1TycZagHPxL-aXx9VnADCtXC8oAJ5_5oqHBf6R2ZSHtTDuX9erIDY9Nmr-WLCqoSw95PKYz6L1riD2tigPeqmq7QSKI233W4thxn5bh2hHtc1rCTT-8FfFPrABdMPZAXeStxLyxlkqn", # Medscape Reference - Dermatofibroma
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXElK6QAf2UDlhf-gSP-FAxRTX6EqBImia4PyC-ViYuBMiJTF8PCEEqLhS2PqqFba7kNKS1BbR43qXUyrF_v1SZCrw-3SZ_qUYIMBjyDxLIFYyY8Q4jSip6fgYUyalDnVWgt7VC8k332F0YxpkJxUu9Bey4H0WhVtQvixA==", # MyPathologyReport.ca - U xơ da
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHIEGX8JVniJ0Qs1mqeFI1j87qk_1W_RO9sQ3-YKfWS3V1KPeG9ThxPVBaoCUwgEA2JNXc_s3Iov6ofIUKiZoXcLbJUfWSZO9q4tojvVgh01JtaTsxnxFFA6k5Z3J5JurHNOENIfaYnroqEXA42y4d2nUo4uLSPBhMnCwyIm9Ggc_poeQCynAlOHLwv26EH4spuuGKtbw==", # Long Châu - U xơ dưới da là gì?
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXGpbA2XXV6NRBOb83elB3Y1qNNRxGAH01BGigrt0CHCx9mZHAlKIvfUsjp1EgMAauYVdVzpl84xg3aAwWe6ZlBMtrtCi_R_neE89hfRSuwAqfazR3tv9LelndJXbKfqKxWfXMEUYGliMiLNDh2-EXcdqYbpdeBjWxxPpRiEEOlzi_Qf", # Medlatec - U mỡ dưới da là gì? Phân biệt u mỡ và u xơ dưới da
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXEdF5XJqOEdaF0LPmhjvSRfhWEJ2G_fJZNy2LdnGgPT7P_pZ8f_PMNiwSYs9hMmEWfNPdEJ9RzOxmPpnu0fhoc3S7Q5dGU1Q-iiArbiCfJiwtQVCfzwY4Bu_lNI0KoZXGq7IxVHz6C7XNmjUv1d_LC6r6JGFqXo8L772cO3VINMb7FDIDJSUf3IjHpdF3RKzlsj7wJusEJNo70hHfNkFN480jLd1xnwBcrjBh-gx9Wimccko0hTDjGGTFydA41q1P40lC4EloG9C2qT3yDZPit6m2pzJeD7cPjOt8x2bXTSM2xJ9FuUw0IcrthI5e1zyPIAtwJs5FHQ-JuiPj8Tb1nWvXfC0Qi1K1jrzldphizbO0CcobM5d_jUOua0BZhrMKrc7QsUQ4KLYBg=", # Cẩm nang MSD - Phiên bản dành cho chuyên gia - U xơ da
    "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXHBDfFHSngyEcGB4DRkSjpMscPS8WYus0a6jebeX7LNDsk10JdG7OKhnP-T56iaN2uV3ay6C0osyPencH_lP3xDUuqUAc2L03DKPQy0niVSK6O3rADeIs-rEv5-inSl90zi-5jA4x7LQIdEZ6XSgMnvoA==" # gsv.com.vn - U XƠ DA - TRIỆU CHỨNG VÀ CÁCH ĐIỀU TRỊ
    ]
    
    # Create scraper instance
    scraper = WebContentScraper(delay_range=(1, 3))
    
    # Scrape content
    print("Starting web scraping...")
    docs = scraper.load(urls)
    
    # Display results
    print(f"\nScraping completed! Found {len(docs)} documents.")
    
    # Show summary
    summary = scraper.get_content_summary(docs)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save original content
    scraper.save_to_file(docs, 'scraped_medical_content.txt')
    print("\nOriginal content saved to 'scraped_medical_content.txt'")
    
    # Process content for medical information and chunking
    print("\nProcessing content for medical information...")
    processor = MedicalContentProcessor()
    processed_docs = processor.process_documents(docs)
    
    print(f"Processing completed! Created {len(processed_docs)} chunks from medical content.")
    print("\nGenerating RAG-compatible data...")
    rag_csv_content = processor.generate_rag_csv(processed_docs, 'skin_cancer_rag_data.csv')
    print("\nRAG CSV data saved to 'skin_cancer_rag_data.csv'")
    
    # Display sample chunks
    if processed_docs:
        print(f"\nSample of processed chunks:")
        for i, chunk_doc in enumerate(processed_docs[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"Title: {chunk_doc['title']}")
            print(f"Word Count: {chunk_doc['word_count']}")
            print(f"Content preview: {chunk_doc['chunk_content'][:200]}...")
            print('----')
    
    # Display RAG integration example
    print("\n=== RAG Integration Example ===")
    print("\n1. CSV format for structured Q&A:")
    csv_lines = rag_csv_content.split('\n')
    for line in csv_lines[:5]:  # Show first 5 lines
        print(line)
    print("...")
    
    print("\n2. Text chunks for direct RAG insertion:")
    if processed_docs:
        print("First chunk for RAG:")
        print(f"Title: {processed_docs[0]['title']}")
        print(f"Source: {processed_docs[0]['url']}")
        print(f"Content: {processed_docs[0]['chunk_content'][:300]}...")
        print('\n' + '='*50)
    
    print("\n=== Usage Instructions ===")
    print("'skin_cancer_rag_data.csv' for structured Q&A RAG systems")