import re
import os
import unicodedata
from pathlib import Path
from collections import Counter
import json

class TextDataCleaner:
    def __init__(self, min_length=50, max_length=100000, min_alpha_ratio=0.6):
        """
        Initialize the data cleaner with quality thresholds.
        
        Args:
            min_length: Minimum character count for a document
            max_length: Maximum character count for a document
            min_alpha_ratio: Minimum ratio of alphabetic characters (filters gibberish)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_alpha_ratio = min_alpha_ratio
        self.stats = {
            'total_docs': 0,
            'removed_too_short': 0,
            'removed_too_long': 0,
            'removed_low_quality': 0,
            'removed_duplicates': 0,
            'cleaned_docs': 0
        }
        self.seen_hashes = set()
    
    def normalize_unicode(self, text):
        """
        Normalize Unicode characters.
        """
        text = unicodedata.normalize('NFKC', text)
        return text
    
    def remove_excessive_whitespace(self, text):
        """Remove excessive whitespace while preserving structure."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()
    
    def remove_html_tags(self, text):
        """Remove HTML/XML tags from web scrapes."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        return text
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        return text

    def remove_special_patterns(self, text):
        """Remove special patterns like excessive punctuation."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if len(line) == 0:
                cleaned_lines.append(line)
                continue
            
            special_count = len(re.findall(r'[^\w\s]', line))
            total_count = len(line.replace(' ', ''))
            
            if total_count > 0:
                special_ratio = special_count / total_count
                # Skip lines that are mostly special characters (>70%)
                if special_ratio > 0.7:
                    continue
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive punctuation (more than 3 in a row)
        text = re.sub(r'([!?.,;:]){4,}', r'\1\1\1', text)
        
        return text

    def calculate_text_quality(self, text):
        """Calculate quality metrics for the text."""
        if len(text) == 0:
            return 0.0
        
        alpha_count = sum(1 for c in text if c.isalpha())
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 0.0
        
        alpha_ratio = alpha_count / total_chars
        return alpha_ratio
    
    def is_duplicate(self, text):
        """Check if text is a duplicate using simple hash."""
        text_hash = hash(text[:1000])
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def clean_text(self, text):
        """Apply all cleaning steps to a text."""
        self.stats['total_docs'] += 1
        
        # Step 1: Normalize Unicode
        text = self.normalize_unicode(text)
        
        # Step 2: Remove HTML tags
        text = self.remove_html_tags(text)
        
        # Step 3: Remove URLs
        text = self.remove_urls(text)
        
        # Step 4: Remove special patterns
        text = self.remove_special_patterns(text)
        
        # Step 5: Remove excessive whitespace
        text = self.remove_excessive_whitespace(text)
        
        # Quality checks
        text_length = len(text)
        
        # Check minimum length
        if text_length < self.min_length:
            self.stats['removed_too_short'] += 1
            return None
        
        # Check maximum length
        if text_length > self.max_length:
            self.stats['removed_too_long'] += 1
            return None
        
        # Check quality (alpha ratio)
        quality = self.calculate_text_quality(text)
        if quality < self.min_alpha_ratio:
            self.stats['removed_low_quality'] += 1
            return None
        
        # Check for duplicates
        if self.is_duplicate(text):
            self.stats['removed_duplicates'] += 1
            return None
        
        self.stats['cleaned_docs'] += 1
        return text
    
    def clean_file(self, input_path, output_path):
        """Clean a single text file."""
        try:
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                return True
            return False
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False
    
    def clean_directory(self, input_dir, output_dir):
        """Clean all text files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all text files
        text_files = list(input_path.glob('**/*.txt'))
        
        print(f"Found {len(text_files)} text files to process...")
        
        for i, file_path in enumerate(text_files, 1):
            # Create relative output path
            relative_path = file_path.relative_to(input_path)
            out_file = output_path / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.clean_file(file_path, out_file)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(text_files)} files...")
        
        print("\n" + "="*50)
        print("DATA CLEANING STATISTICS")
        print("="*50)
        for key, value in self.stats.items():
            print(f"{key}: {value}")
        print("="*50)
        
        # Save statistics
        stats_file = output_path / 'cleaning_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\nStatistics saved to: {stats_file}")
        
# Example usage
if __name__ == "__main__":
    # Initialize cleaner with your preferences
    cleaner = TextDataCleaner(
        min_length=50,          # Minimum 50 characters
        max_length=100000,      # Maximum 100k characters
        min_alpha_ratio=0.6     # At least 60% alphabetic characters
    )
    
    # Clean a single file
    # cleaner.clean_file('input.txt', 'output.txt')
    
    # Clean entire directory
    # cleaner.clean_directory('raw_data/', 'cleaned_data/')
    
    # Example: Clean a sample text
    sample_text = """
    <html><body>
    Bu bir örnek metindir. This is sample text with Turkish characters: çğıöşü.
    
    Visit our website at https://example.com for more info!!!!!!
    
    @@@@####$$$$%%%%   (gibberish line to be removed)
    
    This is good quality text that should be kept.
    Türkçe karakterler doğru şekilde işlenmelidir.
    </body></html>
    """
    
    cleaned = cleaner.clean_text(sample_text)
    print("Original text:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    print("Cleaned text:")
    print(cleaned)