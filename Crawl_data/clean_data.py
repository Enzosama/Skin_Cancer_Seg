import os
import csv
import json
import time
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class SkinCancerDatasetValidator:
    """Validates whether CSV datasets contain skin cancer related content"""
    
    def __init__(self):
        self.client = Groq()
        self.skin_cancer_keywords = {
            'primary': [
                'skin cancer', 'melanoma', 'basal cell carcinoma', 'squamous cell carcinoma',
                'carcinoma', 'malignant melanoma', 'cutaneous melanoma', 'skin tumor',
                'skin lesion', 'dermatology', 'dermatologist', 'oncology', 'skin oncology'
            ],
            'secondary': [
                'mole', 'moles', 'nevus', 'nevi', 'melanocytic nevi', 'dysplastic nevus',
                'actinic keratosis', 'seborrheic keratosis', 'keratosis', 'pigmented lesion',
                'skin biopsy', 'dermoscopy', 'dermatoscopy', 'skin screening',
                'sun exposure', 'uv radiation', 'sunburn', 'skin protection',
                'atypical mole', 'irregular mole', 'changing mole'
            ],
            'medical_terms': [
                'biopsy', 'pathology', 'histology', 'diagnosis', 'prognosis',
                'metastasis', 'staging', 'treatment', 'therapy', 'surgery',
                'chemotherapy', 'immunotherapy', 'radiation therapy',
                'excision', 'mohs surgery', 'cryotherapy'
            ],
            'symptoms': [
                'asymmetry', 'border irregularity', 'color variation', 'diameter',
                'evolving', 'abcde', 'bleeding mole', 'itching mole', 'painful mole',
                'ulceration', 'nodule', 'papule', 'plaque', 'pigmentation'
            ]
        }
        
        self.validation_stats = {
            'total_files': 0,
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'keyword_matches': 0,
            'llm_validations': 0
        }
    
    def check_keyword_relevance(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains skin cancer related keywords"""
        if not text:
            return False, []
        
        text_lower = text.lower()
        found_keywords = []
        
        # Check all keyword categories
        for category, keywords in self.skin_cancer_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        # Consider relevant if found primary keywords or multiple secondary keywords
        has_primary = any('primary:' in kw for kw in found_keywords)
        secondary_count = len([kw for kw in found_keywords if 'secondary:' in kw])
        medical_count = len([kw for kw in found_keywords if 'medical_terms:' in kw])
        symptom_count = len([kw for kw in found_keywords if 'symptoms:' in kw])
        
        is_relevant = (
            has_primary or 
            secondary_count >= 2 or 
            (secondary_count >= 1 and medical_count >= 1) or
            (secondary_count >= 1 and symptom_count >= 1)
        )
        
        return is_relevant, found_keywords
    
    def validate_with_llm(self, content: str, context: str = "") -> str:
        """Use LLM to validate if content is related to skin cancer"""
        prompt = (
            f"You are a medical expert specializing in dermatology and skin cancer. "
            f"Determine if the following content is related to skin cancer, skin lesions, "
            f"dermatology, or skin health. Consider topics like melanoma, basal cell carcinoma, "
            f"squamous cell carcinoma, moles, skin screening, sun protection, and skin examination.\n\n"
            f"Context: {context}\n"
            f"Content: {content}\n\n"
            f"Respond with only 'VALID' if the content is related to skin cancer/dermatology, "
            f"or 'INVALID' if it is not related."
        )
        
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a medical expert in dermatology and skin cancer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.0
            )
            result = completion.choices[0].message.content.strip().upper()
            self.validation_stats['llm_validations'] += 1
            return result if result in ['VALID', 'INVALID'] else 'INVALID'
        except Exception as e:
            print(f"[ERROR] LLM validation failed: {e}")
            return 'INVALID'
    
    def validate_row(self, row: Dict, row_index: int) -> Tuple[bool, Dict]:
        """Validate a single row from CSV data"""
        validation_result = {
            'row_index': row_index,
            'is_valid': False,
            'validation_method': None,
            'keywords_found': [],
            'confidence': 'low',
            'content_analyzed': []
        }
        
        # Combine all text fields for analysis
        text_fields = []
        for key, value in row.items():
            if isinstance(value, str) and value.strip():
                text_fields.append(f"{key}: {value}")
                validation_result['content_analyzed'].append(key)
        
        combined_text = " ".join(text_fields)
        
        # First check with keywords
        is_keyword_relevant, found_keywords = self.check_keyword_relevance(combined_text)
        validation_result['keywords_found'] = found_keywords
        
        if is_keyword_relevant:
            validation_result['is_valid'] = True
            validation_result['validation_method'] = 'keyword_matching'
            validation_result['confidence'] = 'high' if any('primary:' in kw for kw in found_keywords) else 'medium'
            self.validation_stats['keyword_matches'] += 1
            return True, validation_result
        
        # If no clear keyword match, use LLM for validation
        if len(combined_text) > 50:  # Only use LLM for substantial content
            context = f"Disease: {row.get('disease_name', 'Unknown')}"
            llm_result = self.validate_with_llm(combined_text[:1000], context)  # Limit text length
            
            if llm_result == 'VALID':
                validation_result['is_valid'] = True
                validation_result['validation_method'] = 'llm_validation'
                validation_result['confidence'] = 'medium'
                return True, validation_result
        
        validation_result['validation_method'] = 'rejected'
        return False, validation_result
    
    def process_csv_file(self, csv_path: str) -> Dict:
        """Process a single CSV file and validate its content"""
        print(f"\n=== Processing: {os.path.basename(csv_path)} ===")
        
        file_results = {
            'file_path': csv_path,
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0,
            'validation_details': [],
            'sample_valid_rows': [],
            'sample_invalid_rows': []
        }
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                # Detect delimiter
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                
                for row_index, row in enumerate(reader):
                    file_results['total_rows'] += 1
                    self.validation_stats['total_rows'] += 1
                    
                    is_valid, validation_detail = self.validate_row(row, row_index)
                    file_results['validation_details'].append(validation_detail)
                    
                    if is_valid:
                        file_results['valid_rows'] += 1
                        self.validation_stats['valid_rows'] += 1
                        
                        # Store sample valid rows
                        if len(file_results['sample_valid_rows']) < 3:
                            file_results['sample_valid_rows'].append({
                                'row_index': row_index,
                                'content': dict(row),
                                'validation': validation_detail
                            })
                    else:
                        file_results['invalid_rows'] += 1
                        self.validation_stats['invalid_rows'] += 1
                        
                        # Store sample invalid rows
                        if len(file_results['sample_invalid_rows']) < 3:
                            file_results['sample_invalid_rows'].append({
                                'row_index': row_index,
                                'content': dict(row),
                                'validation': validation_detail
                            })
                    
                    # Progress indicator
                    if row_index % 100 == 0 and row_index > 0:
                        print(f"Processed {row_index} rows...")
                    
                    # Rate limiting for LLM calls
                    if validation_detail['validation_method'] == 'llm_validation':
                        time.sleep(1.2)
        
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")
            return file_results
        
        # Calculate validation rate
        validation_rate = (file_results['valid_rows'] / file_results['total_rows'] * 100) if file_results['total_rows'] > 0 else 0
        file_results['validation_rate'] = validation_rate
        
        print(f"Results: {file_results['valid_rows']}/{file_results['total_rows']} rows valid ({validation_rate:.1f}%)")
        
        return file_results
    
    def find_csv_files(self, directory: str) -> List[str]:
        """Find all CSV files in the directory"""
        csv_files = []
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(directory, file))
        return csv_files
    
    def generate_report(self, results: List[Dict], output_path: str):
        """Generate a comprehensive validation report"""
        report = {
            'validation_summary': self.validation_stats.copy(),
            'file_results': results,
            'overall_validation_rate': (self.validation_stats['valid_rows'] / self.validation_stats['total_rows'] * 100) if self.validation_stats['total_rows'] > 0 else 0,
            'recommendations': []
        }
        
        # Add recommendations based on results
        overall_rate = report['overall_validation_rate']
        if overall_rate >= 80:
            report['recommendations'].append("Excellent: Dataset is highly relevant to skin cancer research.")
        elif overall_rate >= 60:
            report['recommendations'].append("Good: Dataset is mostly relevant, consider filtering out invalid entries.")
        elif overall_rate >= 40:
            report['recommendations'].append("Moderate: Dataset has mixed relevance, significant filtering recommended.")
        else:
            report['recommendations'].append("Poor: Dataset has low relevance to skin cancer, major revision needed.")
        
        # Save detailed report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SKIN CANCER DATASET VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Total files processed: {self.validation_stats['total_files']}")
        print(f"Total rows analyzed: {self.validation_stats['total_rows']}")
        print(f"Valid rows: {self.validation_stats['valid_rows']}")
        print(f"Invalid rows: {self.validation_stats['invalid_rows']}")
        print(f"Overall validation rate: {overall_rate:.1f}%")
        print(f"Keyword matches: {self.validation_stats['keyword_matches']}")
        print(f"LLM validations: {self.validation_stats['llm_validations']}")
        print(f"\nDetailed report saved to: {output_path}")
        
        for recommendation in report['recommendations']:
            print(f"• {recommendation}")
    
    def save_valid_data(self, results: List[Dict], output_path: str):
        """Save only valid rows to a new file"""
        valid_data = []
        
        for file_result in results:
            csv_path = file_result['file_path']
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    rows = list(reader)
                    
                    for validation_detail in file_result['validation_details']:
                        if validation_detail['is_valid']:
                            row_index = validation_detail['row_index']
                            if row_index < len(rows):
                                row_data = rows[row_index].copy()
                                row_data['_validation_method'] = validation_detail['validation_method']
                                row_data['_confidence'] = validation_detail['confidence']
                                row_data['_keywords_found'] = '; '.join(validation_detail['keywords_found'])
                                row_data['_source_file'] = os.path.basename(csv_path)
                                valid_data.append(row_data)
            
            except Exception as e:
                print(f"[ERROR] Failed to extract valid data from {csv_path}: {e}")
        
        # Save valid data
        if valid_data:
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in valid_data:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f"\nValid data saved to: {output_path}")
            print(f"Total valid entries: {len(valid_data)}")
        else:
            print("\nNo valid data found to save.")
    
    def validate_datasets(self, data_directory: str):
        """Main method to validate all CSV datasets in the directory"""
        print("SKIN CANCER DATASET VALIDATOR")
        print("=" * 50)
        
        # Find all CSV files
        csv_files = self.find_csv_files(data_directory)
        
        if not csv_files:
            print(f"No CSV files found in {data_directory}")
            return
        
        print(f"Found {len(csv_files)} CSV files to validate:")
        for csv_file in csv_files:
            print(f"  • {os.path.basename(csv_file)}")
        
        self.validation_stats['total_files'] = len(csv_files)
        
        # Process each CSV file
        all_results = []
        for csv_file in csv_files:
            result = self.process_csv_file(csv_file)
            all_results.append(result)
        
        # Generate outputs
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(data_directory, f'validation_report_{timestamp}.json')
        valid_data_path = os.path.join(data_directory, f'valid_skin_cancer_data_{timestamp}.txt')
        
        self.generate_report(all_results, report_path)
        self.save_valid_data(all_results, valid_data_path)


def main():
    """Main function to run the skin cancer dataset validator"""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use Data directory instead of script_dir
    data_dir = os.path.join(script_dir, 'Data')
    
    # Initialize validator
    validator = SkinCancerDatasetValidator()
    
    # Run validation
    validator.validate_datasets(data_dir)


if __name__ == "__main__":
    main()
