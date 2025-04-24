from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
import re
import json
from collections import defaultdict

# Retrieve environment variables
import os
from dotenv import load_dotenv

# Debugging
import logging

# API keys
from anthropic import Anthropic
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Access the claude API key
claude_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# To test the API
@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask!"})

# Base upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the base upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_path):
    """
    Encode an image file to base64.

    :param image_path: Path to the image file
    :return: Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def extract_text_from_image(api_key, image_paths, prompt_text):
    """
    Use Claude to extract text from multiple images.

    :param api_key: Your Anthropic API key
    :param image_paths: List of paths to image files
    :param prompt_text: Text prompt for Claude
    :return: Extracted text from the images
    """
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)

    try:
        # Create content array
        content = []

        # Add each image to the content array
        image_added = False
        for image_path in image_paths:
            # Determine media type based on file extension
            media_type = "image/jpeg"  # Default
            if image_path.lower().endswith(".png"):
                media_type = "image/png"
            elif image_path.lower().endswith(".gif"):
                media_type = "image/gif"
            elif image_path.lower().endswith(".webp"):
                media_type = "image/webp" # Add webp support


            # Encode the image
            logging.info(f"Encoding image: {os.path.basename(image_path)}")
            base64_image = encode_image(image_path)

            if base64_image:
                # Add image to content array
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                })
                image_added = True
            else:
                logging.warning(f"Skipping image due to encoding error: {os.path.basename(image_path)}")

        if not image_added:
            logging.error("No images could be successfully encoded and added.")
            return None

        # Add text prompt at the end
        content.append({
            "type": "text",
            "text": prompt_text
        })

        # Send request to Claude
        logging.info("Sending request to Claude API...")
        response = client.messages.create(
            model="claude-3-haiku-20240307", # Using Haiku for speed/cost balance
            # model="claude-3-opus-20240229", # Opus might give better results but is slower/more expensive
            max_tokens=4000, # Increased max_tokens slightly
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        # Return the extracted text
        extracted_text = response.content[0].text
        logging.info(f"Received response from Claude. Content length: {len(extracted_text)} characters.")
        # Log first few lines of response for debugging
        # logging.debug("Claude Response Head:\n" + "\n".join(extracted_text.split('\n')[:5]))
        return extracted_text

    except Exception as e:
        logging.error(f"An error occurred during Claude API call: {e}", exc_info=True) # Log traceback
        return None

def parse_menu_items(extracted_text):
    """
    Parse menu items and their prices from extracted text.
    
    :param extracted_text: Text extracted from the image
    :return: Dictionary of menu items and their prices, plus confidence scores
    """
    menu_items = {}
    confidence_scores = {}
    
    # Basic check if extraction returned anything
    if not extracted_text or not extracted_text.strip():
        logging.warning("Received empty or null text from extraction.")
        return {}, {}

    # Split the text into lines
    lines = extracted_text.strip().split('\n')
    total_non_empty_lines = sum(1 for line in lines if line.strip())
    successful_extractions = 0
    unclear_items = 0
    default_category_items = 0
    
    logging.info(f"Parsing {total_non_empty_lines} non-empty lines from Claude output.")

    # --- Refined Regex Patterns ---
    # Price pattern: Handles various currency symbols (optional), commas, and decimals
    # Makes currency symbol and spacing optional. Allows integer prices.
    price_pattern = r'(?:[$€£¥]\s*|\b)(\d+(?:,\d{3})*(?:\.\d{1,2})?)\b'
    # Regex to find item name, price, and category (assumes format: NAME PRICE CATEGORY)
    # It tries to be flexible with spacing and potential currency symbols.
    # Group 1: Item Name (non-greedy)
    # Group 2: Price (using price_pattern logic)
    # Group 3: Category (rest of the line)
    # This regex is complex and might need tuning based on Claude's actual output format.
    # Let's try a simpler approach first: find price, then split.

    for line_num, line in enumerate(lines):
        line = line.strip()
        # Skip empty lines or potential headers/footers from Claude
        if not line or line.startswith("Here are the items") or line.startswith("---"):
            continue

        line_confidence = 1.0  # Start with full confidence

        # Check for uncertainty indicators in the text (global line check)
        uncertainty_phrases = ["unclear", "can't make out", "illegible", "not visible", "hard to read", "possibly", "maybe", "appears to be"]
        is_unclear = False
        for phrase in uncertainty_phrases:
            if phrase in line.lower():
                line_confidence *= 0.6  # Reduce confidence if uncertainty is indicated globally
                is_unclear = True
                # break # Found one, no need to check others for this line

        # 1. Find the price first (more reliable anchor)
        price_match = re.search(price_pattern, line)

        if price_match:
            # Extract the price and convert to float
            price_str = price_match.group(1).replace(',', '')
            try:
                price = float(price_str)
            except ValueError:
                logging.warning(f"Line {line_num+1}: Could not convert price '{price_str}' to float. Skipping line: '{line}'")
                continue

            # 2. Extract Item Name (everything before the price match)
            item_name = line[:price_match.start()].strip()
            # Clean up common trailing characters before price
            item_name = re.sub(r'[.…\-_*\s]+$', '', item_name).strip()

            # 3. Extract Category (everything after the price match)
            category_name = line[price_match.end():].strip()
            # Clean up common leading characters after price
            category_name = re.sub(r'^[.…\-_*\s]+', '', category_name).strip()
            
            # --- Data Cleaning and Validation ---
            # Handle empty item name (likely parsing error or header)
            if not item_name:
                logging.warning(f"Line {line_num+1}: Extracted empty item name. Skipping line: '{line}'")
                continue

            # Handle empty or default category
            if not category_name or category_name.lower() in ["unclear", "unknown", "n/a", "none", "-", "--"]:
                category_name = "Uncategorized" # Standardize default
                line_confidence *= 0.8 # Slightly reduce confidence if category was unclear/missing
                default_category_items += 1

            # Remove any "(unclear)" tags added by Claude from name/category
            item_name = item_name.replace("(unclear)", "").strip()
            category_name = category_name.replace("(unclear)", "").strip()

            # If the global 'unclear' flag was set, mark item count
            if is_unclear:
                unclear_items += 1

            # Check name/category quality (very short strings might be noise)
            if len(item_name) < 3:
                line_confidence *= 0.8
            if len(category_name) < 3 and category_name != "Uncategorized": # Allow short defaults
                 line_confidence *= 0.9 # Less penalty for short category
            
            # Check price reasonableness
            if price <= 0 or price > 1000: # Adjusted range slightly
                line_confidence *= 0.7
            
            # Store the item
            successful_extractions += 1
            item_key = item_name # Use name as the key

            # Handle duplicate item names (append index if price/category differ)
            if item_key in menu_items:
                existing_item = menu_items[item_key]
                # Only create variant if price OR category is different
                if existing_item['price'] != price or existing_item['category'] != category_name:
                    count = 1
                    new_key = f"{item_name} ({count})"
                    while new_key in menu_items:
                        count += 1
                        new_key = f"{item_name} ({count})"
                    item_key = new_key
                    logging.info(f"Duplicate item name '{item_name}' found with different price/category. Storing as '{item_key}'.")
                else:
                    # Exact duplicate, maybe increase confidence slightly? Or just skip. Let's skip.
                    logging.info(f"Exact duplicate item found: '{item_name}'. Skipping.")
                    successful_extractions -= 1 # Decrement success counter as we skipped it
                    continue # Skip to next line

            menu_items[item_key] = {'price': price, 'category': category_name}
            confidence_scores[item_key] = max(0.0, min(1.0, line_confidence)) # Clamp confidence 0-1

        else:
            # Line did not contain a recognizable price pattern
            logging.warning(f"Line {line_num+1}: No valid price pattern found. Skipping line: '{line}'")

    # Calculate overall extraction quality metrics
    extraction_rate = successful_extractions / total_non_empty_lines if total_non_empty_lines > 0 else 0
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0

    # Log the extraction metrics
    logging.info(f"Parsing complete. Successfully extracted {successful_extractions} items.")
    logging.info(f"Extraction rate: {extraction_rate:.1%} of non-empty lines.")
    logging.info(f"Items marked unclear by Claude or parser: {unclear_items}")
    logging.info(f"Items assigned default 'Uncategorized' category: {default_category_items}")
    logging.info(f"Average confidence score for extracted items: {avg_confidence:.1%}")

    return menu_items, confidence_scores

def categorize_menu_items(menu_items, num_categories=3):
    """
    Categorize menu items based on their price into groups like A, B, C.

    :param menu_items: Dictionary {name: {'price': float, 'category': str}}
    :param num_categories: Number of price categories (A, B, C...)
    :return: Dictionary {price_category: [{'name': str, 'price': float, 'category': str}, ...]}
    """
    if not menu_items:
        logging.warning("No menu items provided for price categorization.")
        return {}

    # Prepare items list: [{'name': name, 'price': price, 'category': category}, ...]
    items_list = [{'name': name, **data} for name, data in menu_items.items()]

    # Sort items by price (descending)
    sorted_items_list = sorted(items_list, key=lambda x: x['price'], reverse=True)

    # Determine price ranges
    prices = [item['price'] for item in sorted_items_list]
    if not prices: # Should not happen if menu_items is not empty, but safety check
        return {}

    max_price = max(prices)
    min_price = min(prices)
    price_range = max_price - min_price

    logging.info(f"\nPrice range for categorization: ${min_price:.2f} to ${max_price:.2f} (Spread: ${price_range:.2f})")

    # Create categories based on price ranges
    # Result structure: {'A': [item_dict1, item_dict2], 'B': [...]}
    categories = defaultdict(list)

    # Handle edge cases: zero range or only one category requested
    if num_categories <= 1 or price_range == 0:
        logging.info("Assigning all items to Category A (single category requested or zero price range).")
        categories['A'].extend(sorted_items_list)
        return dict(categories) # Convert back to dict from defaultdict

    # --- Adaptive Categorization Logic (from original code) ---
    # Decide whether to use equal price range division or percentile-based division
    large_range_threshold = 3.0 # Use adaptive if range > 3x min price
    use_adaptive = price_range > min_price * large_range_threshold and min_price > 0

    if use_adaptive:
        logging.info(f"Large price range detected (range/min > {large_range_threshold:.1f}) - using adaptive (percentile-based) categorization.")

        num_items = len(sorted_items_list)
        category_boundaries = [] # List of lower bounds for categories B, C, D...

        # Calculate boundary prices based on item count percentiles
        for i in range(1, num_categories):
            idx = int((i * num_items) / num_categories)
            # Use the price of the item at the boundary index as the threshold
            # Ensure index is valid
            if 0 <= idx < num_items:
                # The boundary is the price *below which* items fall into the next category
                # So, category A is >= boundary[0], B is < boundary[0] and >= boundary[1], etc.
                boundary_price = sorted_items_list[idx]['price']
                # Add a small epsilon if needed to handle exact matches at boundaries consistently
                # Let's adjust logic: boundary price IS the lower limit for the higher category
                category_boundaries.append(boundary_price)
            else:
                # Should not happen with valid indices, but handle gracefully
                logging.warning(f"Could not determine boundary price for category {chr(65 + i)}")
                # Use the previous boundary or min price as fallback
                category_boundaries.append(category_boundaries[-1] if category_boundaries else min_price)

        # Ensure boundaries are unique and sorted descending
        category_boundaries = sorted(list(set(category_boundaries)), reverse=True)

        # Add the minimum price as the effective floor for the last category
        category_boundaries.append(min_price - 0.01) # Ensure min_price items are included

        boundary_strs = [f">${b:.2f}" for b in category_boundaries[:-1]] # Don't print the floor boundary
        logging.info(f"Adaptive category price boundaries (Min price for Cat A, B, C...): {boundary_strs}")

        # Assign items to categories based on these boundaries
        for item_dict in sorted_items_list:
            item_price = item_dict['price']
            assigned = False
            # Find the first boundary the price is >= to
            for i, boundary in enumerate(category_boundaries[:-1]): # Iterate through A, B, C... boundaries
                if item_price >= boundary:
                    category_letter = chr(65 + i)
                    categories[category_letter].append(item_dict)
                    assigned = True
                    break
            # If not assigned (should only happen for min_price items falling below last explicit boundary)
            if not assigned:
                # Assign to the last category
                category_letter = chr(65 + num_categories - 1)
                categories[category_letter].append(item_dict)

    else:
        # --- Standard Equal Price Range Division ---
        logging.info("Using standard categorization (equal price range division).")
        # Avoid division by zero if num_categories is 0 or less (handled earlier, but safety)
        if num_categories <= 0: num_categories = 1
        category_range_size = price_range / num_categories

        # Print calculated category ranges
        logging.info(f"Calculated category range size: ${category_range_size:.2f}")
        for i in range(num_categories):
            cat_letter = chr(65 + i)
            cat_max = max_price - (i * category_range_size)
            cat_min = max_price - ((i + 1) * category_range_size)
            # Ensure the last category includes the minimum price exactly
            if i == num_categories - 1: cat_min = min_price
            logging.info(f"  Target range for Category {cat_letter}: ~${cat_min:.2f} - ${cat_max:.2f}")


        for item_dict in sorted_items_list:
            item_price = item_dict['price']
            # Determine category index based on price position within the total range
            # Handle max_price edge case: should be in Category A (index 0)
            if item_price == max_price:
                category_index = 0
            # Avoid division by zero if category_range_size is 0 (handled earlier, but safety)
            elif category_range_size > 0:
                # Calculate how many 'ranges' down from the max price this item is
                category_index = int((max_price - item_price) / category_range_size)
                # Clamp index to valid range [0, num_categories - 1]
                category_index = max(0, min(num_categories - 1, category_index))
            else: # category_range_size is 0 (all prices same) -> should be handled by initial check
                category_index = 0

            category_letter = chr(65 + category_index)
            categories[category_letter].append(item_dict)

    # Log the distribution of items across final categories
    logging.info("\n--- Final Item Distribution by Price Category ---")
    for category_letter, items in sorted(categories.items()):
        category_min_price = min(item['price'] for item in items) if items else 0
        category_max_price = max(item['price'] for item in items) if items else 0
        logging.info(f"Category {category_letter}: {len(items)} items. Actual price range: ${category_min_price:.2f} - ${category_max_price:.2f}")

    return dict(categories) # Convert back to regular dict

def process_menu_images(api_key, image_folder, num_price_categories=3):
    """
    Process all menu images in a folder, extract items (name, price, category),
    and categorize them by price tiers (A, B, C...).

    :param api_key: Your Anthropic API key
    :param image_folder: Path to folder containing menu images
    :param num_price_categories: Number of price categories (A, B, C...)
    :return: Price-categorized menu items dict, and overall confidence scores dict
    """
    # List all image files in the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp'] # Added webp
    image_paths = []

    if not os.path.isdir(image_folder):
        logging.error(f"Image folder not found or is not a directory: {image_folder}")
        return {}, {}

    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(image_folder, file))

    if not image_paths:
        logging.error(f"No supported image files ({', '.join(image_extensions)}) found in {image_folder}")
        return {}, {}

    logging.info(f"Found {len(image_paths)} images to process in folder: {image_folder}")

    # Store all items extracted across all images before final categorization
    # Using the structure: {name: {'price': float, 'category': str}}
    all_menu_items_combined = {}
    all_confidence_scores_combined = {}
    image_extraction_counts = []

    # --- Updated Prompt for Claude ---
    prompt_text = f"""
    Please extract all menu items from the provided image(s).
    For each distinct menu item found, provide the following information on a single line:
    1. The complete Item Name.
    2. The exact Price (including currency symbol like $ or € if visible, otherwise just the number).
    3. The item's Category (e.g., Appetizer, Main Course, Dessert, Beverage, Side Dish).

    Format each item STRICTLY as:
    Item Name $Price Category

    Examples:
    Classic Caesar Salad $12.99 Appetizer
    Grilled Salmon Fillet $24.50 Main Course
    New York Cheesecake $8.00 Dessert
    Iced Tea $3.50 Beverage
    French Fries $5.00 Side Dish

    IMPORTANT INSTRUCTIONS:
    - List EVERY visible menu item, even if unsure about details.
    - If you cannot clearly determine the Item Name, Price, or Category, use the placeholder "(unclear)" for that specific part. For example: "House Special (unclear) $16.99 Main Course" or "Spicy Tuna Roll $15.00 (unclear)". If the category is totally unknown, use "Uncategorized".
    - Ensure the price is extracted accurately.
    - Do NOT add any introductory text, explanations, summaries, or formatting like bullet points or markdown. Only output the list of items in the specified format, one item per line.
    - If an item seems to span multiple lines in the menu, combine it into a single logical item name if possible.
    - Pay attention to sections or headers in the menu image to help determine the category.
    """

    # Process images (can be done one-by-one or batched if API/model supports it well)
    # Let's stick to one-by-one for simplicity and robustness against single image failures.
    for idx, image_path in enumerate(image_paths):
        logging.info(f"\n--- Processing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")

        extracted_text = extract_text_from_image(api_key, [image_path], prompt_text)

        if extracted_text:
            logging.info(f"Text extraction successful for {os.path.basename(image_path)}. Parsing content...")

            # Parse menu items (name, price, category) from this image's text
            menu_items_single, confidence_scores_single = parse_menu_items(extracted_text)
            item_count_single = len(menu_items_single)
            logging.info(f"Parsed {item_count_single} items from {os.path.basename(image_path)}.")
            image_extraction_counts.append((os.path.basename(image_path), item_count_single))

            # Merge items from this image into the combined dictionary
            for item_name, item_data in menu_items_single.items():
                confidence = confidence_scores_single.get(item_name, 0.5) # Default confidence if missing

                # Check if item already exists in combined list
                if item_name in all_menu_items_combined:
                    existing_data = all_menu_items_combined[item_name]
                    # If price or category differs, create a variant
                    if existing_data['price'] != item_data['price'] or existing_data['category'] != item_data['category']:
                        variant_count = 1
                        variant_name = f"{item_name} (Img {idx+1})" # Add image source to variant name
                        while variant_name in all_menu_items_combined:
                            variant_count += 1
                            variant_name = f"{item_name} (Img {idx+1} Var {variant_count})"

                        logging.warning(f"Item '{item_name}' from {os.path.basename(image_path)} differs from previous entry. Storing as '{variant_name}'. "
                                        f"Old: P={existing_data['price']}, C='{existing_data['category']}'. New: P={item_data['price']}, C='{item_data['category']}'.")
                        all_menu_items_combined[variant_name] = item_data
                        all_confidence_scores_combined[variant_name] = confidence
                    else:
                        # Exact same item found again, potentially update confidence if higher?
                        # For now, let's keep the first encountered confidence.
                        logging.info(f"Item '{item_name}' is an exact duplicate from another image. Skipping.")
                else:
                    # New item, add it
                    all_menu_items_combined[item_name] = item_data
                    all_confidence_scores_combined[item_name] = confidence

            logging.info(f"Combined total unique items so far: {len(all_menu_items_combined)}")
        else:
            logging.error(f"Failed to extract text from {os.path.basename(image_path)}. Skipping this image.")
            image_extraction_counts.append((os.path.basename(image_path), 0)) # Record failure

    # --- Post-Processing and Categorization ---
    logging.info(f"\n===== Image Processing Summary =====")
    total_items_extracted = sum(count for _, count in image_extraction_counts)
    logging.info(f"Total items parsed across all images (before deduplication/variants): {total_items_extracted}")
    logging.info(f"Final unique items/variants stored: {len(all_menu_items_combined)}")

    # Print extraction performance by image
    logging.info("\nItems Parsed Per Image:")
    for image_name, item_count in image_extraction_counts:
        logging.info(f"  {image_name}: {item_count} items")

    # Calculate overall average confidence score
    if all_confidence_scores_combined:
        avg_confidence = sum(all_confidence_scores_combined.values()) / len(all_confidence_scores_combined)
        logging.info(f"Overall average extraction confidence: {avg_confidence:.1%}")
    else:
        logging.info("No confidence scores available.")


    if not all_menu_items_combined:
        logging.error("No menu items were successfully extracted from any images.")
        return {}, {}

    # --- Dynamic Adjustment of Price Categories ---
    # Adjust num_categories based on item count for better distribution
    min_items_per_category = 3 # Aim for at least 3 items per category if possible
    max_possible_categories = len(all_menu_items_combined) // min_items_per_category
    adjusted_categories = max(1, min(num_price_categories, max_possible_categories))

    if adjusted_categories != num_price_categories:
        logging.warning(f"Adjusting number of price categories from {num_price_categories} to {adjusted_categories} "
                    f"based on item count ({len(all_menu_items_combined)}) and target minimum items per category ({min_items_per_category}).")
        num_price_categories = adjusted_categories
    else:
        logging.info(f"Using requested number of price categories: {num_price_categories}")

    # Categorize the combined menu items by price tiers (A, B, C...)
    logging.info("\n--- Categorizing All Extracted Items by Price ---")
    final_price_categories = categorize_menu_items(all_menu_items_combined, num_price_categories)

    return final_price_categories, all_confidence_scores_combined

### FOR BUNDLE GENERATION ###
def generate():
    try:
        # Get restaurant name from request
        restaurant_name = request.json.get('restaurantName')
        num_categories = request.json.get('numCategories', 4)  # Default to 4
        if not restaurant_name:
            return jsonify({"error": "Restaurant name is required"}), 400

        client = genai.Client(
            api_key= gemini_api_key,
        )
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return
    
    # Check if categorized_menu_items.txt exists
    categorized_file_loc = f"./uploads/{secure_filename(restaurant_name)}/categorized_menu_items_{secure_filename(restaurant_name)}.txt"
    if not os.path.exists(categorized_file_loc):
        print(f"Error: {categorized_file_loc} file not found")
        return
        
    try:
        # Read the file content
        with open(categorized_file_loc, "r") as file:
            menu_content = file.read()
    except Exception as e:
        print(f"Error reading menu file: {e}")
        return
    
    # Create prompt with the file content
    prompt = f"""
    Based on these categorised menu data:
    
    {menu_content}
    
    Please create 4 UNIQUE menu bundles for a varying number of diners (1-6 diners). Each bundle must include some items from each available menu category (A, B, C, and D if present).
    
    For each menu bundle, include:
    1. The number of menu items from each category
    2. The number of diners the bundle is designed for
    3. The price per diner
    4. A suggested discounted bundle price

    Please follow this exact structure for each menu bundle:
    Suggested bundle price:
    Number of diners:
    Category Portions:
        Category A: [number of items]
        Category B: [number of items]
        Category C: [number of items]
        Category D: [number of items]
    Original bundle price:
    Discount percentage:
    Price per diner:
    
    Note: Include all four categories (A through D) in your response, even if some categories might have 0 items in certain bundles.
    """

    model = "gemini-2.5-pro-exp-03-25"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        system_instruction=[
            types.Part.from_text(text="""You are a menu bundle generator, and your task is to create 4 UNIQUE menu bundles based on a given input of menu categories. Each bundle should be distinctly different from the others. For each bundle:
            
1. Include items from all available categories (A, B, C, and D)
2. Specify exactly how many portions from each category are included
3. Calculate the original price based on the actual menu prices
4. Apply a reasonable discount (10-20%)
5. Calculate the per-person price
            
Always include all four categories (A through D) in your response structure, even if some categories have 0 items or don't exist in the input data."""),
        ],
    )

    # Collect the entire response
    complete_response = ""
    
    try:
        # Get the response generator
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Check if response is None
        if response_stream is None:
            print("Warning: API returned None for response stream")
            # Try non-streaming version as fallback
            try:
                print("Attempting to use non-streaming API as fallback...")
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if response and hasattr(response, 'text'):
                    complete_response = response.text
                elif response and hasattr(response, 'parts'):
                    for part in response.parts:
                        if hasattr(part, 'text') and part.text is not None:
                            complete_response += part.text
                else:
                    print("Warning: Fallback response format is unexpected")
            except Exception as fallback_error:
                print(f"Fallback request also failed: {fallback_error}")
        else:
            # Iterate through streaming response if it's not None
            for chunk in response_stream:
                # Handle the case where chunk.text might be None
                # Get the text from different potential structures
                chunk_text = ""
                
                # Try direct text property
                if hasattr(chunk, 'text') and chunk.text is not None:
                    chunk_text = chunk.text
                # Try looking in parts
                elif hasattr(chunk, 'parts') and chunk.parts:
                    # Combine text from all parts
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text is not None:
                            chunk_text += part.text
                # Try candidates structure
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content'):
                            content = candidate.content
                            if hasattr(content, 'parts'):
                                for part in content.parts:
                                    if hasattr(part, 'text') and part.text is not None:
                                        chunk_text += part.text
                
                # Add to the complete response
                complete_response += chunk_text
                
                # Print to console
                if chunk_text:
                    print(chunk_text, end="")
                else:
                    print(".", end="")  # Print a dot to indicate progress for empty chunks
                
    except Exception as e:
        print(f"\n\nError during generation: {e}")
        
        # Last-ditch effort - try non-streaming API if streaming failed
        if not complete_response:
            try:
                print("Attempting to use non-streaming API as final fallback...")
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                
                if hasattr(response, 'text') and response.text:
                    complete_response = response.text
                    print(f"Received {len(complete_response)} characters from fallback request")
            except Exception as fallback_error:
                print(f"Fallback request also failed: {fallback_error}")
    
    # Save the raw response for debugging
    try:
        menu_bundles_raw_file = f"./uploads/{secure_filename(restaurant_name)}/menu_bundles_raw_{secure_filename(restaurant_name)}.txt"
        with open(menu_bundles_raw_file, "w") as f:
            f.write(complete_response)
    except Exception as e:
        print(f"Error saving raw response: {e}")
    
    # Parse and save as JSON
    try:
        print("\n\nParsing response as JSON...")
        
        # Check if the response is empty
        if not complete_response.strip():
            raise ValueError("Empty response received from API")
            
        # Try direct JSON parsing first
        all_valid_jsons = []
        unique_bundles = set()  # Use a set to track unique bundles
        
        try:
            # First, try parsing the entire response as a single JSON object
            json_data = json.loads(complete_response)
            if isinstance(json_data, list):
                # If it's a list, use it directly
                all_valid_jsons = json_data[:4]  # Limit to 4 bundles
            else:
                # If it's an object, check if it has a 'bundles' property
                if "bundles" in json_data and isinstance(json_data["bundles"], list):
                    json_data["bundles"] = json_data["bundles"][:4]  # Limit to 4 bundles
                all_valid_jsons = [json_data]
        except json.JSONDecodeError:
            print("Direct JSON parsing failed, trying extraction methods...")
            
            # Look for JSON objects pattern
            json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
            json_matches = re.findall(json_pattern, complete_response)
            
            if json_matches:
                # Try each potential JSON match
                potential_jsons = sorted(json_matches, key=len, reverse=True)
                
                for potential_json in potential_jsons:
                    try:
                        json_data = json.loads(potential_json)
                        # Convert to string for comparison to check uniqueness
                        json_str = json.dumps(json_data, sort_keys=True)
                        if json_str not in unique_bundles:
                            unique_bundles.add(json_str)
                            all_valid_jsons.append(json_data)
                    except json.JSONDecodeError:
                        continue
            
            # Also try code blocks
            code_block_matches = re.findall(r'```(?:json)?(.*?)```', complete_response, re.DOTALL)
            for code_block in code_block_matches:
                try:
                    json_data = json.loads(code_block.strip())
                    # Check uniqueness again
                    json_str = json.dumps(json_data, sort_keys=True)
                    if json_str not in unique_bundles:
                        unique_bundles.add(json_str)
                        all_valid_jsons.append(json_data)
                except json.JSONDecodeError:
                    continue
        
        # Ensure we have only 4 bundles maximum
        if len(all_valid_jsons) > 4:
            all_valid_jsons = all_valid_jsons[:4]
        
        # Save all valid JSONs to a single file
        if all_valid_jsons:
            menu_bundles_file = f"./uploads/{secure_filename(restaurant_name)}/menu_bundles_{secure_filename(restaurant_name)}.json"
            with open(menu_bundles_file, "w") as json_file:
                json.dump(all_valid_jsons, json_file, indent=4)
            print(f"Saved {len(all_valid_jsons)} unique menu bundles to {menu_bundles_file}")
        else:
            print("No valid JSONs found in the response")
            
    except Exception as e:
        print(f"Error: Failed to parse response as JSON: {e}")
        print(f"Please check {menu_bundles_raw_file} to see the actual response.")

@app.route('/api/upload', methods=['POST'])
def upload_images():
    # Get restaurant name from form data
    restaurant_name = request.form.get("restaurantName")
    
    if not restaurant_name:
        return jsonify({"message": "Restaurant name is required"}), 400

    # Create a directory for the restaurant (use secure_filename to sanitize input)
    restaurant_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(restaurant_name))
    if not os.path.exists(restaurant_folder):
        os.makedirs(restaurant_folder)

    # Get uploaded files
    files = request.files.getlist("images")
    
    if not files:
        return jsonify({"message": "No images uploaded"}), 400

    saved_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(restaurant_folder, filename)
            file.save(filepath)
            saved_files.append(filename)

    return jsonify({
        "message": f"{len(saved_files)} images uploaded successfully for '{restaurant_name}'!",
        "files": saved_files,
        "restaurant": restaurant_name,
        "folder_path": restaurant_folder,
    }), 200

@app.route('/api/process-menu', methods=['POST'])
def process_menu():
    try:
        # Get restaurant name from request
        restaurant_name = request.json.get('restaurantName')
        num_categories = request.json.get('numCategories', 4)  # Default to 4
        if not restaurant_name:
            return jsonify({"error": "Restaurant name is required"}), 400

        # Get API key from headers
        api_key = request.headers.get('X-API-Key', claude_api_key)

        # Construct path to uploaded images
        restaurant_folder = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(restaurant_name))

        # Check if folder exists
        if not os.path.exists(restaurant_folder):
            return jsonify({
                "error": f"No images found for restaurant '{restaurant_name}'. Please upload images first."
            }), 404

        # Get all image files in the folder
        image_paths = [
            os.path.join(restaurant_folder, f)
            for f in os.listdir(restaurant_folder)
            if allowed_file(f)
        ]

        if not image_paths:
            return jsonify({
                "error": f"No valid images found in the folder for restaurant '{restaurant_name}'. Please upload menu images first."
            }), 404

        # Process with dynamic category count
        final_price_categories, confidence_scores = process_menu_images(
            api_key, 
            restaurant_folder, 
            num_price_categories=num_categories
        )

        # ---- Save output as text file ----
        output_filename = f"categorized_menu_items_{secure_filename(restaurant_name)}.txt"
        output_path = os.path.join(restaurant_folder, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Categorized Menu Items for {restaurant_name}\n\n")
            for category, items in final_price_categories.items():
                f.write(f"Category: {category}\n")
                for item in items:
                    # Adjust this line based on your item structure
                    name = item.get('name', 'Unknown')
                    price = item.get('price', 'N/A')
                    f.write(f"  - {name}: ${price}\n")
                f.write("\n")
            f.write("Confidence Scores:\n")
            for key, value in confidence_scores.items():
                f.write(f"  {key}: {value}\n")
        # ---- End save output ----

        return jsonify({
            "categories": final_price_categories,
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        app.logger.error(f"Processing failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Menu processing failed",
            "error": str(e)
        }), 500

@app.route('/api/generate-bundles', methods=['POST'])
def api_generate_bundles():
    restaurant_name = request.json.get('restaurantName')
    try:
        generate()
        # Check if output file exists
        menu_bundles_file = f"./uploads/{secure_filename(restaurant_name)}/menu_bundles_{secure_filename(restaurant_name)}.json"
        if os.path.exists(menu_bundles_file):
            with open(menu_bundles_file, "r") as f:
                bundles = json.load(f)
            return jsonify({"status": "success", "bundles": bundles}), 200
        else:
            return jsonify({"status": "error", "message": "menu_bundles.json not found"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)