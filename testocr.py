import easyocr
import re
import os
from collections import defaultdict
import numpy as np # For spatial calculations

# --- Initialization ---
# Initialize EasyOCR Reader for English and Thai.
# Set gpu=True if you have a compatible GPU and CUDA installed for faster processing.
# The model will be downloaded automatically on the first run.
print("Initializing EasyOCR Reader (this may take a moment)...")
try:
    # Consider adding more languages if needed, e.g., ['en', 'th', 'ch_sim']
    reader = easyocr.Reader(['en', 'th'], gpu=False)
    print("EasyOCR Reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR Reader: {e}")
    print("Please ensure PyTorch and EasyOCR are installed correctly.")
    reader = None # Set reader to None to prevent further errors

# --- Constants and Keywords ---
CURRENCY_SYMBOLS = ['$', '฿', '£', '€'] # Add more currency symbols if needed
# Regex to find prices: handles optional symbol, commas, decimals, and common currency codes/words
PRICE_REGEX = re.compile(
    r'([$' + ''.join(re.escape(s) for s in CURRENCY_SYMBOLS) + r']?\s*\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?|\d{1,3}(?:[,.]\d{3})*(?:\.\d{1,2})?\s*(?:[$' + ''.join(re.escape(s) for s in CURRENCY_SYMBOLS) + r']|\bbaht\b|\bthb\b))',
    re.IGNORECASE
)

# Keywords for categorization (lowercase for case-insensitive matching)
CATEGORY_KEYWORDS = {
    # Food Categories
    "Appetizer": ["appetizer", "appetizers", "starter", "starters", "ของว่าง", "อาหารเรียกน้ำย่อย"],
    "Main Course": ["main course", "mains", "entree", "entrees", "signature", "อาหารจานหลัก", "จานหลัก"],
    "Soup": ["soup", "soups", "ซุป"],
    "Salad": ["salad", "salads", "สลัด"],
    "Dessert": ["dessert", "desserts", "sweet", "sweets", "ของหวาน"],
    "Side Dish": ["side", "sides", "extra", "extras", "เครื่องเคียง"],
    "Curry": ["curry", "curries", "แกง"],
    "Noodles": ["noodle", "noodles", "ก๋วยเตี๋ยว"],
    "Rice": ["rice", "ข้าว"],
    "Seafood": ["seafood", "อาหารทะเล"],
    "Pizza": ["pizza", "พิซซ่า"],
    "Pasta": ["pasta", "พาสต้า"],
    "Grill": ["grill", "grilled", "bbq", "ย่าง"],
    "Vegetarian": ["vegetarian", "veggie", "มังสวิรัติ"],
    "Breakfast": ["breakfast", "อาหารเช้า"],
    "Lunch": ["lunch", "อาหารกลางวัน"],
    "Dinner": ["dinner", "อาหารเย็น"],
    "Set Menu": ["set menu", "combo", "ชุดอาหาร"],

    # Beverage Categories (Specific rules applied below)
    "Alcoholic Beverage": [
        "beer", "beers", "wine", "wines", "spirit", "spirits", "whiskey", "whisky", "vodka", "gin", "rum", "liqueur",
        "cocktail", "cocktails", "เบียร์", "ไวน์", "เหล้า", "ค็อกเทล", "สุรา",
        # Specific Brands (add more as needed)
        "singha", "chang", "heineken", "tsingtao", "leo", "asahi", "tiger", "carlsberg", "san miguel", "budweiser",
        "johnnie walker", "chivas", "smirnoff", "absolut", "bacardi", "jack daniel's", "jameson",
        "santa vittoria sparkling" # Specific case
        ],
    "Non-Alcoholic Beverage": [
        "beverage", "beverages", "drink", "drinks", "soft drink", "soft drinks",
        "soda", "sodas", "juice", "juices", "tea", "teas", "coffee", "coffees",
        "mocktail", "mocktails", "water", "mineral water", "smoothie", "shakes",
        "เครื่องดื่ม", "น้ำอัดลม", "น้ำผลไม้", "ชา", "กาแฟ", "ม็อกเทล", "น้ำเปล่า", "น้ำแร่", "สมูทตี้",
        # Specific Brands (add more as needed)
        "pepsi", "coke", "coca-cola", "sprite", "fanta", "7up", "schweppes",
        "lipton", "dilmah",
        "orange juice", "apple juice", "pineapple juice", "mango juice", "coconut water",
        "latte", "cappuccino", "espresso", "americano", "mocha",
        "santa vittoria" # General case (non-sparkling)
        ]
}

# Reverse map for faster keyword lookup
KEYWORD_TO_CATEGORY = {}
for category, keywords in CATEGORY_KEYWORDS.items():
    for keyword in keywords:
        KEYWORD_TO_CATEGORY[keyword] = category

# --- Helper Functions ---

def is_thai(text):
    """Check if the text contains Thai characters."""
    return any('\u0E00' <= char <= '\u0E7F' for char in text)

def find_price(text):
    """
    Extracts price from a text string using regex.
    Cleans the extracted price and standardizes the format.
    """
    # Remove potential noise like commas before matching, handle spaces around symbols
    cleaned_text = text.replace(',', '').strip()
    match = PRICE_REGEX.search(cleaned_text)
    if match:
        price_str = match.group(0).strip()
        # Extract symbol if present at start or end
        symbol = next((s for s in CURRENCY_SYMBOLS if price_str.startswith(s) or price_str.endswith(s)), None)

        # Extract digits and decimal point
        digits = re.sub(r'[^\d\.]', '', price_str) # Keep only digits and dot

        # Handle cases like "100 baht" -> ฿100
        if not symbol:
            if re.search(r'\b(baht|thb)\b', price_str, re.IGNORECASE):
                symbol = "฿"
            # Add similar checks for other currency words if needed (e.g., usd, eur)

        if digits:
            # Basic validation: check if it looks like a plausible price number
            try:
                # Ensure there's at most one decimal point
                if digits.count('.') <= 1:
                    # Attempt conversion to float to catch invalid formats like '...'
                    float(digits)
                    # Format with symbol first if found, otherwise just digits
                    return f"{symbol}{digits}" if symbol else digits
                else:
                    return None # Invalid number format (multiple decimals)
            except ValueError:
                return None # Cannot convert to float, likely not a valid price
        else:
             return None # No valid digits found
    return None # No price pattern matched

def determine_category(text, current_category_context):
    """
    Determines the category based on keywords in the text.
    Prioritizes beverage rules and handles specific cases.
    Returns a tuple: (determined_category, is_likely_header)
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return current_category_context, False # No text, maintain context

    best_match_category = current_category_context # Default to current context
    found_keyword = False
    matched_categories = set()

    # --- Strict Beverage Checks First ---
    is_sv_sparkling = "santa vittoria sparkling" in text_lower
    is_sv_general = "santa vittoria" in text_lower and not is_sv_sparkling

    if is_sv_sparkling:
        # If it's *only* these words, it's likely the item. If part of a header, header logic handles it.
        if text_lower == "santa vittoria sparkling":
             return "Alcoholic Beverage", False
        # Otherwise, let general logic proceed but prioritize this if ambiguous
        matched_categories.add("Alcoholic Beverage")
        found_keyword = True

    if is_sv_general:
        if text_lower == "santa vittoria":
             return "Non-Alcoholic Beverage", False
        matched_categories.add("Non-Alcoholic Beverage")
        found_keyword = True


    # --- General Keyword Check ---
    words = text_lower.split()
    keyword_word_count = 0
    for word in words:
        # Clean word: remove leading/trailing punctuation but keep internal hyphens/apostrophes
        cleaned_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        if cleaned_word in KEYWORD_TO_CATEGORY:
            matched_categories.add(KEYWORD_TO_CATEGORY[cleaned_word])
            found_keyword = True
            keyword_word_count += 1

    if len(matched_categories) == 1:
        best_match_category = matched_categories.pop()
    elif len(matched_categories) > 1:
        # Handle ambiguity (e.g., "Beer Battered Fish")
        # Prioritize beverage categories if present
        if "Alcoholic Beverage" in matched_categories:
            best_match_category = "Alcoholic Beverage"
        elif "Non-Alcoholic Beverage" in matched_categories:
            best_match_category = "Non-Alcoholic Beverage"
        else:
            # If multiple non-beverage categories match, stick to context or pick one (e.g., first found)
            best_match_category = list(matched_categories)[0] # Simple fallback


    # --- Header Detection Heuristic ---
    is_likely_header = False
    # Condition 1: Keywords found, and it's a potential change from context
    if found_keyword and (best_match_category != current_category_context or current_category_context == "Uncategorized"):
        # Condition 2: Line consists mostly of keywords or is very short
        # Allow for '&', '-', '/' as connectors
        non_keyword_words = [w for w in words if w.lower() not in KEYWORD_TO_CATEGORY and w not in ['&', '-', '/']]
        # Check if the line is predominantly keywords or all caps
        if (len(non_keyword_words) <= 1 and len(words) < 6) or \
           (text.isupper() and len(words) < 5):
             # Condition 3: High ratio of keyword words to total words
             if len(words) > 0 and (keyword_word_count / len(words)) > 0.6:
                  is_likely_header = True

    # If it's not likely a header, but a keyword was found, use that category for the item
    # Otherwise, stick with the context category passed in.
    final_category = best_match_category if found_keyword and not is_likely_header else current_category_context

    # Ensure final category isn't None or empty
    final_category = final_category if final_category else "Uncategorized"

    # If it IS likely a header, the category returned is the *new* context category
    return (best_match_category if is_likely_header else final_category, is_likely_header)


def get_vertical_center(bbox):
    """Calculates the vertical center of a bounding box."""
    # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
    return (bbox[0][1] + bbox[2][1]) / 2

def group_lines(results, y_tolerance_ratio=0.8):
    """
    Groups OCR results into lines based on vertical proximity and text height.
    Increased default tolerance slightly.
    Args:
        results: List of tuples from reader.readtext (bbox, text, confidence).
        y_tolerance_ratio: Tolerance for vertical distance relative to text height.
                           Smaller value means stricter grouping (e.g., 0.5).
                           Larger value means looser grouping (e.g., 1.0).
    Returns:
        List of lines, where each line is a list of results sorted horizontally.
    """
    if not results:
        return []

    # Sort primarily by vertical position, secondarily by horizontal
    sorted_results = sorted(results, key=lambda r: (get_vertical_center(r[0]), r[0][0][0]))

    lines = []
    current_line = []
    if not sorted_results:
        return lines

    current_line.append(sorted_results[0])
    # Use the y-center of the first box as the initial reference for the line
    current_line_y_center = get_vertical_center(sorted_results[0][0])
    # Use the height of the first box as the initial reference height
    current_line_ref_height = sorted_results[0][0][2][1] - sorted_results[0][0][0][1]

    for i in range(1, len(sorted_results)):
        result = sorted_results[i]
        bbox, text, conf = result
        y_center = get_vertical_center(bbox)
        height = bbox[2][1] - bbox[0][1]

        # Calculate dynamic tolerance based on the reference height of the current line being built
        y_tolerance = current_line_ref_height * y_tolerance_ratio

        # Check if the vertical distance between the box's center and the line's average center is within tolerance
        if abs(y_center - current_line_y_center) <= y_tolerance:
            current_line.append(result)
            # Update line's average y_center and reference height (use max height in line)
            current_line_y_center = np.mean([get_vertical_center(r[0]) for r in current_line])
            current_line_ref_height = max(current_line_ref_height, height)
        else:
            # Start a new line: sort the completed line horizontally
            lines.append(sorted(current_line, key=lambda r: r[0][0][0]))
            # Reset for the new line
            current_line = [result]
            current_line_y_center = y_center
            current_line_ref_height = height

    # Add the last line
    if current_line:
        lines.append(sorted(current_line, key=lambda r: r[0][0][0]))

    return lines

# --- Main Extraction Function ---

def extract_menu_items_from_image(image_path, min_confidence=0.2):
    """
    Extracts menu items from a single image using EasyOCR and applies formatting rules.
    Improved logic for handling line parts and separating price/names.

    Args:
        image_path (str): Path to the menu image file.
        min_confidence (float): Minimum confidence score for OCR results to be considered.

    Returns:
        list: A list of strings, each formatted as:
              'English Name Thai Name $Price Category'
              Returns an empty list if the image cannot be processed or no items are found.
    """
    if reader is None:
        print("Error: EasyOCR Reader not initialized. Cannot process image.")
        return []

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return []

    print(f"\nProcessing image: {image_path}...")
    try:
        # Perform OCR - paragraph=False is usually better for menus
        # detail=1 gives bounding boxes needed for grouping
        results = reader.readtext(image_path, detail=1, paragraph=False)
        print(f"Initial OCR found {len(results)} text blocks.")
    except Exception as e:
        print(f"Error processing image {image_path} with EasyOCR: {e}")
        return []

    # --- Filter low confidence results ---
    filtered_results = [r for r in results if r[2] >= min_confidence]
    print(f"Filtered to {len(filtered_results)} blocks with confidence >= {min_confidence}.")
    if not filtered_results:
        print("No text blocks passed the confidence threshold.")
        return []

    # --- Line Grouping and Processing ---
    # Adjust y_tolerance_ratio if needed based on menu layout (0.6 stricter, 1.0 looser)
    lines = group_lines(filtered_results, y_tolerance_ratio=0.8)
    print(f"Grouped into {len(lines)} potential lines.")

    extracted_items_output = []
    current_category = "Uncategorized" # Track the category section

    for line_index, line in enumerate(lines):
        # line is a list of tuples: (bbox, text, confidence), sorted horizontally

        # Combine text from all blocks in the line for analysis
        full_line_text = " ".join([res[1] for res in line]).strip()
        # print(f"\nProcessing Line {line_index+1}: '{full_line_text}'") # Debug print

        # --- Category Detection ---
        potential_category, is_header = determine_category(full_line_text, current_category)

        if is_header:
            if potential_category != current_category:
                 print(f"  Line {line_index+1}: Detected Category Header: '{full_line_text}' -> Setting context to '{potential_category}'")
                 current_category = potential_category
            # else:
            #      print(f"  Line {line_index+1}: Interpreted as Header (already in category '{current_category}'): '{full_line_text}'")
            continue # Skip processing this line as a menu item

        # --- Item Extraction (if not a header) ---
        price = "(unclear)"
        item_category = current_category # Start with the current section's category
        name_blocks_text = []
        price_block_found = False

        # Iterate backwards through the line's blocks to find the price first (usually at the end)
        possible_name_parts = []
        for i in range(len(line) - 1, -1, -1):
            bbox, text, conf = line[i]
            if not price_block_found:
                found_price = find_price(text)
                if found_price:
                    # print(f"    Found price '{found_price}' in block '{text}'") # Debug print
                    price = found_price
                    price_block_found = True
                    continue # Stop looking for price, move to name parts

            # If not the price block, add its text to potential name parts (in reverse order for now)
            possible_name_parts.append(text)

        # Reverse the collected name parts to get the correct order
        name_blocks_text = possible_name_parts[::-1]
        combined_name_text = " ".join(name_blocks_text).strip()

        # If no name text remains after extracting price, skip
        if not combined_name_text:
            # print(f"  Line {line_index+1}: Skipped (No name text identified after price removal: '{full_line_text}')")
            continue

        # Separate English and Thai from the combined name text
        english_name_parts = []
        thai_name_parts = []
        # Split combined text and check each part
        for part in combined_name_text.split():
             if is_thai(part):
                 thai_name_parts.append(part)
             else:
                 # Basic filter: avoid adding very short, likely noise parts unless it's all there is
                 if len(part.strip()) > 1 or len(combined_name_text.strip()) <= 1:
                      english_name_parts.append(part)

        english_name = " ".join(english_name_parts).strip()
        thai_name = " ".join(thai_name_parts).strip()

        # If BOTH names are empty (e.g., only short noise words were filtered out), skip
        if not english_name and not thai_name:
             # print(f"  Line {line_index+1}: Skipped (Both English and Thai names empty after filtering: '{combined_name_text}')")
             continue

        # --- Refine Category Based on Item Text ---
        item_text_for_category = f"{english_name} {thai_name}".strip()
        item_specific_category, _ = determine_category(item_text_for_category, current_category) # Ignore header flag here

        # Apply beverage rules strictly: If item text indicates a beverage, use that category.
        if item_specific_category in ["Alcoholic Beverage", "Non-Alcoholic Beverage"]:
            item_category = item_specific_category
        # If the section category was Uncategorized, use the item-derived one if found
        elif current_category == "Uncategorized" and item_specific_category != "Uncategorized":
            item_category = item_specific_category
        # Otherwise, stick with the category determined by the section header (current_category)


        # --- Final Formatting ---
        formatted_english = english_name if english_name else ""
        formatted_thai = thai_name if thai_name else ""
        formatted_price = price # Already defaults to "(unclear)" or has extracted price

        # Construct the final output string according to the strict format
        # Ensure only single spaces between components
        output_parts = [formatted_english, formatted_thai, formatted_price, item_category]
        output_line = " ".join(part for part in output_parts if part) # Join non-empty parts

        # Clean up potential multiple spaces resulting from joining parts or within names
        output_line = re.sub(r'\s+', ' ', output_line).strip()

        # Basic check to avoid adding lines that are essentially empty or just category/price placeholders
        if len(output_line.replace("(unclear)", "").replace(item_category, "").strip()) > 0:
             # Avoid adding exact duplicates
             if output_line not in extracted_items_output:
                 print(f"  Line {line_index+1}: Extracted Item: '{output_line}'")
                 extracted_items_output.append(output_line)
             # else:
             #     print(f"  Line {line_index+1}: Skipped (Duplicate): '{output_line}'")
        # else:
        #      print(f"  Line {line_index+1}: Skipped (Empty after formatting): '{full_line_text}'")


    print(f"Finished processing {image_path}. Extracted {len(extracted_items_output)} items.")
    return extracted_items_output

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy image files for testing if they don't exist
    # In a real scenario, replace these with actual image paths
    dummy_image_files = ["menu_example_th_en.png"]
    for img_file in dummy_image_files:
        if not os.path.exists(img_file):
            print(f"Attempting to create dummy image: {img_file}...")
            try:
                # Create a simple placeholder image (requires Pillow)
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (700, 900), color = (255, 255, 255))
                d = ImageDraw.Draw(img)
                try:
                    font_path_reg = "Arial.ttf" # Example for Windows/macOS
                    font_path_th = "Tahoma.ttf" # Example Thai font
                    if not os.path.exists(font_path_reg): font_path_reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Linux fallback
                    if not os.path.exists(font_path_th): font_path_th = font_path_reg # Fallback if no Thai font

                    if os.path.exists(font_path_reg):
                        font_l = ImageFont.truetype(font_path_reg, 30)
                        font_m = ImageFont.truetype(font_path_reg, 20)
                        font_s = ImageFont.truetype(font_path_reg, 18)
                    else: raise IOError("Regular font not found")
                    if os.path.exists(font_path_th):
                        font_th_m = ImageFont.truetype(font_path_th, 20)
                        font_th_s = ImageFont.truetype(font_path_th, 18)
                    else: # Fallback if Thai font not found
                         font_th_m = font_m
                         font_th_s = font_s

                except IOError:
                    print("Warning: Could not load specific fonts, using default.")
                    font_l = ImageFont.load_default()
                    font_m = ImageFont.load_default(); font_th_m = font_m
                    font_s = ImageFont.load_default(); font_th_s = font_s

                # Sample Menu Layout (using separate fonts for clarity if available)
                y = 20
                d.text((300, y), "OUR MENU", fill=(0,0,0), font=font_l); y += 50

                d.text((50, y), "Appetizers", fill=(50,50,50), font=font_m); d.text((200, y), "ของว่าง", fill=(50,50,50), font=font_th_m); y += 35
                d.text((50, y), "Classic Caesar Salad", fill=(0,0,0), font=font_s); d.text((250, y), "ซีซาร์สลัดคลาสสิค", fill=(0,0,0), font=font_th_s); d.text((550, y), "$12.99", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Spring Rolls", fill=(0,0,0), font=font_s); d.text((250, y), "ปอเปี๊ยะทอด", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿150", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Garlic Bread", fill=(0,0,0), font=font_s); d.text((550, y), "100", fill=(0,0,0), font=font_s); y += 45 # Price without symbol

                d.text((50, y), "Main Courses", fill=(50,50,50), font=font_m); d.text((220, y), "อาหารจานหลัก", fill=(50,50,50), font=font_th_m); y += 35
                d.text((50, y), "Grilled Salmon Fillet", fill=(0,0,0), font=font_s); d.text((250, y), "ปลาแซลมอนย่าง", fill=(0,0,0), font=font_th_s); d.text((550, y), "$24.50", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Pad Thai", fill=(0,0,0), font=font_s); d.text((250, y), "ผัดไทย", fill=(0,0,0), font=font_th_s); d.text((550, y), "180 THB", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Green Curry", fill=(0,0,0), font=font_s); d.text((250, y), "แกงเขียวหวาน", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿200", fill=(0,0,0), font=font_s); y += 45

                d.text((50, y), "Desserts", fill=(50,50,50), font=font_m); y += 35
                d.text((50, y), "New York Cheesecake", fill=(0,0,0), font=font_s); d.text((250, y), "นิวยอร์กชีสเค้ก", fill=(0,0,0), font=font_th_s); d.text((550, y), "$8.00", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Mango Sticky Rice", fill=(0,0,0), font=font_s); d.text((250, y), "ข้าวเหนียวมะม่วง", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿160", fill=(0,0,0), font=font_s); y += 45

                d.text((50, y), "Beverages", fill=(50,50,50), font=font_m); d.text((200, y), "เครื่องดื่ม", fill=(50,50,50), font=font_th_m); y += 35
                d.text((50, y), "Singha Beer", fill=(0,0,0), font=font_s); d.text((250, y), "สิงห์เบียร์", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿110.00", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Tsingtao Beer", fill=(0,0,0), font=font_s); d.text((250, y), "青岛啤酒", fill=(0,0,0), font=font_th_s); d.text((550, y), "130 BAHT", fill=(0,0,0), font=font_s); y += 25 # Note: Dummy font likely won't render Chinese
                d.text((50, y), "House Wine (Red/White)", fill=(0,0,0), font=font_s); d.text((550, y), "$9.00", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Pepsi", fill=(0,0,0), font=font_s); d.text((550, y), "$3.50", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Orange Juice", fill=(0,0,0), font=font_s); d.text((250, y), "น้ำส้ม", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿80", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Iced Tea", fill=(0,0,0), font=font_s); d.text((250, y), "ชาเย็น", fill=(0,0,0), font=font_th_s); d.text((550, y), "75", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Coffee", fill=(0,0,0), font=font_s); d.text((250, y), "กาแฟ", fill=(0,0,0), font=font_th_s); d.text((550, y), "฿90", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Santa Vittoria Sparkling", fill=(0,0,0), font=font_s); d.text((550, y), "฿150", fill=(0,0,0), font=font_s); y += 25
                d.text((50, y), "Santa Vittoria Water", fill=(0,0,0), font=font_s); d.text((550, y), "฿80", fill=(0,0,0), font=font_s); y += 25

                img.save(img_file)
                print(f"Created dummy image: {img_file}")
            except ImportError:
                print(f"Error: Pillow library not installed. Cannot create dummy image: {img_file}.")
                print("Please install Pillow: pip install Pillow")
            except Exception as e:
                 print(f"Error creating dummy image {img_file}: {e}")


    # --- Processing ---
    # Replace this list with the actual paths to your menu images
    image_paths_to_process = ["./sample/ENG&THAI MENUS/DIMM/Screenshot 2025-02-24 143159.png"] # Add paths like "/path/to/your/menu.jpg"

    all_extracted_items = []
    for image_path in image_paths_to_process:
        # You can adjust min_confidence here if needed, e.g., extract_menu_items_from_image(image_path, min_confidence=0.3)
        items = extract_menu_items_from_image(image_path)
        all_extracted_items.extend(items)

    # --- Final Output ---
    print("\n--- Final Extracted Menu Items (Formatted) ---")
    if all_extracted_items:
        # Print each item exactly as required, one item per line
        for item_line in all_extracted_items:
            print(item_line)
    else:
        # Print nothing if no items were extracted, as per instructions.
        pass

