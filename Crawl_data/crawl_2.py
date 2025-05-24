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
        """
        Clean text content to keep only medical-relevant information
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text with only medical content
        """
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
        """
        Split text into chunks using RecursiveCharacterTextSplitter approach
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
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
        
        # Create CSV content in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header compatible with RAG system
        writer.writerow(['category', 'source_url', 'source_link', 'question', 'answer'])
        
        for doc in processed_docs:
            # Extract category from title or content
            category = self._extract_category(doc['title'], doc['chunk_content'])
            
            # Generate question-answer pairs from content
            qa_pairs = self._generate_qa_pairs(doc['chunk_content'])
            
            for question, answer in qa_pairs:
                writer.writerow([
                    category,
                    doc['url'],
                    f"[{doc['title']}]({doc['url']})",
                    question,
                    answer
                ])
        
        csv_content = output.getvalue()
        output.close()
        
        # Save to file
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        
        return csv_content
    
    def _extract_category(self, title: str, content: str) -> str:
        """
        Extract medical category from title and content
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Medical category string
        """
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
        """
        Generate question-answer pairs from content
        
        Args:
            content: Text content to process
            
        Returns:
            List of (question, answer) tuples
        """
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
    # Your URLs
    urls = [
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXG6IH3ybChsMwVfSzqmAD4GQRi-ltwXX3wAG3Ai2sfAKDL7uSxPROFXsDfp72c19geUuX8U2Fcj2ei8iz8jjKAZWrXd3-AQRAT5DbxUNzWPqGBwSaLzPNsl8jKRVtE3B3dEqVO4-vK9Ri9e9E5hfnB6wbVuVA==",
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbF9wXFFBRToYcXNFFDZ7OsXKU5GmdItjoYnKBZxj9ya_md6VfHMoRRPo8aoKSbo92D__dDcJmHu-kqiwhE-dcc-dofvZCjh2xjhRz0OrQprSNboThwBtYuONlBgoeubcNYi-ZeHqNJm8g==",
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
    
    # Save processed content
    processor.save_processed_content(processed_docs, 'processed_medical_chunks.txt')
    print("\nProcessed and chunked content saved to 'processed_medical_chunks.txt'")
    
    # Generate RAG-compatible CSV data
    print("\nGenerating RAG-compatible data...")
    rag_csv_content = processor.generate_rag_csv(processed_docs, 'skin_cancer_rag_data.csv')
    print("\nRAG CSV data saved to 'skin_cancer_rag_data.csv'")
    
    # Export for direct RAG insertion
    rag_text_content = processor.export_for_rag(processed_docs, 'text')
    with open('skin_cancer_rag_chunks.txt', 'w', encoding='utf-8') as f:
        f.write(rag_text_content)
    print("\nRAG text chunks saved to 'skin_cancer_rag_chunks.txt'")
    
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
    print("1. Use 'skin_cancer_rag_data.csv' for structured Q&A RAG systems")
    print("2. Use 'skin_cancer_rag_chunks.txt' for direct text chunk insertion")
    print("3. Both formats are optimized for skin cancer domain knowledge")
    print("4. Content is pre-filtered for medical relevance and properly chunked")