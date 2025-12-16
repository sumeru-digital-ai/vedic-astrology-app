"""
KALAPURUSHA VEDIC ASTROLOGY - CORRECT IMPLEMENTATION
Converts Ascendant chart to Kalapurusha (Universal) chart with proper Graha Drishti
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import re
import pandas as pd
from io import BytesIO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

# ==================== VEDIC CONSTANTS ====================

class Planet(Enum):
    SUN = "Sun"
    MOON = "Moon"
    MARS = "Mars"
    MERCURY = "Mercury"
    JUPITER = "Jupiter"
    VENUS = "Venus"
    SATURN = "Saturn"
    RAHU = "Rahu"
    KETU = "Ketu"

PLANET_COLORS = {
    Planet.SUN: (255, 0, 0),
    Planet.MOON: (200, 200, 200),
    Planet.MARS: (255, 0, 0),
    Planet.MERCURY: (0, 200, 0),
    Planet.JUPITER: (255, 200, 0),
    Planet.VENUS: (255, 150, 200),
    Planet.SATURN: (0, 0, 255),
    Planet.RAHU: (100, 100, 100),
    Planet.KETU: (139, 69, 19)
}

PLANET_PATTERNS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER, 'PI': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS, 'UF': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU,
    'NE': Planet.MERCURY
}

# GRAHA DRISHTI (Classical Vedic)
GRAHA_DRISHTI = {
    Planet.SUN: [7],
    Planet.MOON: [7],
    Planet.MARS: [4, 7, 8],
    Planet.MERCURY: [7],
    Planet.JUPITER: [5, 7, 9],
    Planet.VENUS: [7],
    Planet.SATURN: [3, 7, 10],
    Planet.RAHU: [5, 7, 9],
    Planet.KETU: [5, 7, 9]
}

# Kalapurusha body parts (FIXED positions)
KALAPURUSHA_BODY = {
    1: "Head (Mesha)", 2: "Face/Mouth (Vrishabha)", 3: "Throat/Shoulders (Mithuna)",
    4: "Chest/Heart (Karka)", 5: "Stomach (Simha)", 6: "Intestines (Kanya)",
    7: "Lower Abdomen (Tula)", 8: "Genitals (Vrishchika)", 9: "Thighs (Dhanu)",
    10: "Knees (Makara)", 11: "Calves (Kumbha)", 12: "Feet (Meena)"
}

HOUSE_SIGNIFICATIONS = {
    1: "Self, Body, Personality", 2: "Wealth, Speech, Family",
    3: "Courage, Siblings", 4: "Mother, Home, Heart", 5: "Children, Intelligence",
    6: "Disease, Enemies", 7: "Spouse, Partnership", 8: "Longevity, Transformation",
    9: "Fortune, Dharma", 10: "Career, Status", 11: "Gains, Income", 12: "Loss, Liberation"
}

PLANET_NATURE = {
    Planet.SUN: "Malefic", Planet.MOON: "Benefic", Planet.MARS: "Malefic",
    Planet.MERCURY: "Neutral", Planet.JUPITER: "Benefic", Planet.VENUS: "Benefic",
    Planet.SATURN: "Malefic", Planet.RAHU: "Malefic", Planet.KETU: "Malefic"
}

PLANET_GEMSTONE = {
    Planet.SUN: "Ruby", Planet.MOON: "Pearl", Planet.MARS: "Red Coral",
    Planet.MERCURY: "Emerald", Planet.JUPITER: "Yellow Sapphire",
    Planet.VENUS: "Diamond", Planet.SATURN: "Blue Sapphire",
    Planet.RAHU: "Hessonite", Planet.KETU: "Cat's Eye"
}

PLANET_GRAIN = {
    Planet.SUN: "Wheat", Planet.MOON: "Rice", Planet.MARS: "Red Lentils",
    Planet.MERCURY: "Green Gram", Planet.JUPITER: "Chana Dal",
    Planet.VENUS: "White Rice", Planet.SATURN: "Black Sesame",
    Planet.RAHU: "Black Lentils", Planet.KETU: "Multi-grain"
}

@dataclass
class PlanetPosition:
    planet: Planet
    house: int  # This is the Kalapurusha house (1-12, with 1 at top)
    degrees: Optional[float] = None

# ==================== OCR ENGINE ====================

class KundaliOCR:
    """Extract planets from ascendant chart"""
    
    def extract_planets_and_ascendant(self, image) -> Tuple[List[PlanetPosition], Optional[int]]:
        """Returns (planet_positions_in_ascendant_chart, ascendant_house_number)"""
        
        img = np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        height, width = img.shape[:2]
        
        # North Indian diamond house regions
        house_regions = self._get_house_regions(width, height)
        
        planet_positions = []
        
        for house, (x1, y1, x2, y2) in house_regions.items():
            roi = img[y1:y2, x1:x2]
            text = self._ocr_region(roi)
            planets = self._parse_planets(text, house)
            planet_positions.extend(planets)
        
        # Try to detect ascendant house number
        # In North Indian chart, house 1 is at center-top area
        # We look for small house numbers (1-12) in each region
        
        planet_positions = self._deduplicate(planet_positions)
        
        return planet_positions, None  # For now, user will input ascendant
    
    def _get_house_regions(self, w: int, h: int) -> Dict[int, Tuple[int, int, int, int]]:
        """North Indian diamond house positions"""
        return {
            1: (int(w*0.40), int(h*0.15), int(w*0.60), int(h*0.35)),  # Top
            2: (int(w*0.60), int(h*0.15), int(w*0.75), int(h*0.35)),  # Top-right
            3: (int(w*0.75), int(h*0.28), int(w*0.95), int(h*0.45)),  # Right-top
            4: (int(w*0.78), int(h*0.40), int(w*0.95), int(h*0.60)),  # Right
            5: (int(w*0.75), int(h*0.55), int(w*0.95), int(h*0.72)),  # Right-bottom
            6: (int(w*0.60), int(h*0.65), int(w*0.75), int(h*0.85)),  # Bottom-right
            7: (int(w*0.40), int(h*0.65), int(w*0.60), int(h*0.85)),  # Bottom
            8: (int(w*0.25), int(h*0.65), int(w*0.40), int(h*0.85)),  # Bottom-left
            9: (int(w*0.05), int(h*0.55), int(w*0.25), int(h*0.72)),  # Left-bottom
            10: (int(w*0.02), int(h*0.40), int(w*0.22), int(h*0.60)), # Left
            11: (int(w*0.05), int(h*0.28), int(w*0.25), int(h*0.45)), # Left-top
            12: (int(w*0.25), int(h*0.15), int(w*0.40), int(h*0.35))  # Top-left
        }
    
    def _ocr_region(self, img) -> str:
        """OCR with preprocessing"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, thresh = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh, config='--psm 6')
        
        return text.upper()
    
    def _parse_planets(self, text: str, house: int) -> List[PlanetPosition]:
        """Extract planets"""
        positions = []
        text = text.replace('\n', ' ').replace('  ', ' ')
        
        for pattern, planet in PLANET_PATTERNS.items():
            if pattern in text:
                deg_match = re.search(rf'{pattern}\s*(\d+)', text)
                degrees = float(deg_match.group(1)) if deg_match else None
                positions.append(PlanetPosition(planet, house, degrees))
        
        return positions
    
    def _deduplicate(self, positions: List[PlanetPosition]) -> List[PlanetPosition]:
        """Remove duplicates"""
        seen = set()
        unique = []
        for pos in positions:
            key = (pos.planet, pos.house)
            if key not in seen:
                seen.add(key)
                unique.append(pos)
        return unique

# ==================== KALAPURUSHA CONVERTER ====================

def convert_to_kalapurusha(ascendant_positions: List[PlanetPosition], 
                          ascendant_house: int) -> List[PlanetPosition]:
    """
    Convert ascendant chart to Kalapurusha chart
    
    Example: If ascendant is Scorpio (8th sign):
    - Planet in ascendant house 1 → Kalapurusha house 1 (HEAD)
    - Planet in ascendant house 2 → Kalapurusha house 2 (FACE)
    etc.
    
    The key: We map based on HOUSE NUMBER, not sign
    """
    kalapurusha_positions = []
    
    for pos in ascendant_positions:
        # In Kalapurusha, house 1 is always HEAD (top of chart)
        # The ascendant chart's house 1 maps to Kalapurusha house 1
        # So the mapping is direct: ascendant house N → Kalapurusha house N
        
        kalapurusha_house = pos.house
        
        kalapurusha_positions.append(PlanetPosition(
            planet=pos.planet,
            house=kalapurusha_house,
            degrees=pos.degrees
        ))
    
    return kalapurusha_positions

# ==================== KALAPURUSHA CHART DRAWER ====================

class KalapurushaChartDrawer:
    """Draw proper rectangular Kalapurusha chart like the example"""
    
    def __init__(self, width: int = 1800, height: int = 900):
        self.width = width
        self.height = height
    
    def draw_chart(self, positions: List[PlanetPosition]) -> Image.Image:
        """Draw complete Kalapurusha chart with all planets and aspects"""
        
        img = Image.new('RGB', (self.width, self.height), (255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw structure
        self._draw_structure(draw)
        
        # Draw house numbers and labels
        self._draw_house_info(draw)
        
        # Place planets
        self._place_planets(draw, positions)
        
        # Draw ALL Graha Drishti arrows
        self._draw_all_aspects(draw, positions)
        
        return img
    
    def _draw_structure(self, draw: ImageDraw.Draw):
        """Draw rectangular Kalapurusha structure"""
        margin = 50
        w = self.width - 2 * margin
        h = self.height - 2 * margin
        
        # Outer rectangle
        draw.rectangle([margin, margin, self.width - margin, self.height - margin],
                      outline=(200, 100, 0), width=8)
        
        # Divide into 3 rows and 4 columns
        row_h = h / 3
        col_w = w / 4
        
        # Horizontal lines
        for i in range(1, 3):
            y = margin + i * row_h
            draw.line([margin, y, self.width - margin, y], fill=(100, 150, 255), width=3)
        
        # Vertical lines
        for i in range(1, 4):
            x = margin + i * col_w
            draw.line([x, margin, x, self.height - margin], fill=(100, 150, 255), width=3)
    
    def _get_house_bounds(self, house: int) -> Tuple[int, int, int, int]:
        """Get bounding box for each house (x1, y1, x2, y2)"""
        margin = 50
        w = self.width - 2 * margin
        h = self.height - 2 * margin
        row_h = h / 3
        col_w = w / 4
        
        # Layout like the example:
        # Row 1: [3, 2, 1, 12]
        # Row 2: [4, -, -, 11]
        # Row 3: [5, 6, 7, 10,9, 8]
        
        positions = {
            1: (3, 0),   # Col 3, Row 0
            2: (2, 0),   # Col 2, Row 0
            3: (1, 0),   # Col 1, Row 0
            4: (0, 1),   # Col 0, Row 1
            5: (0, 2),   # Col 0, Row 2
            6: (1, 2),   # Col 1, Row 2
            7: (2, 2),   # Col 2, Row 2
            8: (3, 2),   # Col 3, Row 2
            9: (3, 2),   # Sharing with 8
            10: (3, 2),  # Sharing with 8,9
            11: (3, 1),  # Col 3, Row 1
            12: (3, 0)   # Col 3, Row 0 (sharing with 1)
        }
        
        # Simpler layout: Top row 1-4, middle row 5-8, bottom row 9-12
        layout = {
            1: (2, 0), 2: (1, 0), 3: (0, 0), 4: (0, 1),
            5: (0, 2), 6: (1, 2), 7: (2, 2), 8: (3, 2),
            9: (3, 1), 10: (3, 1), 11: (3, 0), 12: (2, 0)
        }
        
        # Use standard grid layout
        col, row = layout.get(house, (0, 0))
        
        x1 = margin + col * col_w
        y1 = margin + row * row_h
        x2 = x1 + col_w
        y2 = y1 + row_h
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def _get_house_center(self, house: int) -> Tuple[int, int]:
        """Get center point of house"""
        x1, y1, x2, y2 = self._get_house_bounds(house)
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _draw_house_info(self, draw: ImageDraw.Draw):
        """Draw house numbers and body part names"""
        try:
            font_num = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font_num = ImageFont.load_default()
            font_label = font_num
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            
            # House number
            num_text = str(house)
            bbox = draw.textbbox((0, 0), num_text, font=font_num)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - w//2, y - h - 60), num_text, fill=(200, 100, 0), font=font_num)
            
            # Body part label
            label = KALAPURUSHA_BODY[house]
            bbox = draw.textbbox((0, 0), label, font=font_label)
            w = bbox[2] - bbox[0]
            draw.text((x - w//2, y - 30), label, fill=(0, 0, 0), font=font_label)
    
    def _place_planets(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Place planets in their houses"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except:
            font = ImageFont.load_default()
        
        # Group by house
        house_planets = {}
        for pos in positions:
            if pos.house not in house_planets:
                house_planets[pos.house] = []
            house_planets[pos.house].append(pos)
        
        for house, planets in house_planets.items():
            x, y = self._get_house_center(house)
            
            for i, pos in enumerate(planets):
                py = y + (i * 35) + 10
                color = PLANET_COLORS[pos.planet]
                
                text = pos.planet.value
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_all_aspects(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw ALL Graha Drishti arrows"""
        
        arrow_count = 0
        
        for pos in positions:
            aspects = GRAHA_DRISHTI[pos.planet]
            color = PLANET_COLORS[pos.planet]
            
            for offset in aspects:
                target = ((pos.house - 1 + offset) % 12) + 1
                
                x1, y1 = self._get_house_center(pos.house)
                x2, y2 = self._get_house_center(target)
                
                self._draw_arrow(draw, x1, y1, x2, y2, color)
                arrow_count += 1
        
        st.write(f"**✅ Drew {arrow_count} Graha Drishti arrows**")
    
    def _draw_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int,
                    x2: int, y2: int, color: Tuple[int, int, int]):
        """Draw arrow with arrowhead"""
        r, g, b = color
        rgba = (r, g, b, 100)
        
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=6)
        
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 30
        arrow_angle = math.pi / 6
        
        x3 = x2 - arrow_len * math.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * math.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * math.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * math.sin(angle + arrow_angle)
        
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== ANALYSIS ====================

def generate_analysis(positions: List[PlanetPosition]) -> pd.DataFrame:
    """Generate analysis table"""
    data = []
    
    for pos in positions:
        aspects = [str((pos.house - 1 + off) % 12 + 1) 
                  for off in GRAHA_DRISHTI[pos.planet]]
        
        data.append({
            'Planet': pos.planet.value,
            'Kalapurusha House': pos.house,
            'Degrees': f"{int(pos.degrees)}°" if pos.degrees else "N/A",
            'Body Part': KALAPURUSHA_BODY[pos.house],
            'Life Area': HOUSE_SIGNIFICATIONS[pos.house],
            'Nature': PLANET_NATURE[pos.planet],
            'Aspects To': ", ".join(aspects),
            'Gemstone': PLANET_GEMSTONE[pos.planet],
            'Grain': PLANET_GRAIN[pos.planet]
        })
    
    return pd.DataFrame(data)

def health_analysis(positions: List[PlanetPosition]) -> Dict[int, List[str]]:
    """Health impact per body part"""
    impacts = {i: [] for i in range(1, 13)}
    
    for pos in positions:
        effect = "✅ Strengthens" if PLANET_NATURE[pos.planet] == "Benefic" else "⚠️ Challenges"
        impacts[pos.house].append(f"{effect}: {pos.planet.value} placed")
    
    for pos in positions:
        for offset in GRAHA_DRISHTI[pos.planet]:
            target = ((pos.house - 1 + offset) % 12) + 1
            effect = "➡️ Supports" if PLANET_NATURE[pos.planet] == "Benefic" else "➡️ Stresses"
            impacts[target].append(f"{effect}: {pos.planet.value} from H{pos.house}")
    
    return impacts

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(page_title="Kalapurusha Jyotish", page_icon="🕉️", layout="wide")
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    st.markdown("**Convert Ascendant Chart to Kalapurusha (Universal) Chart**")
    
    with st.sidebar:
        st.header("📖 Kalapurusha Concept")
        st.info("""
        Kalapurusha = Universal/Cosmic birth chart
        
        House 1 (TOP) = HEAD
        Always starts at top, goes counter-clockwise
        
        Your ascendant chart is mapped to this fixed structure.
        """)
        
        st.header("🔮 Graha Drishti")
        st.code("""
Mars: 4, 7, 8
Jupiter: 5, 7, 9
Saturn: 3, 7, 10
Rahu/Ketu: 5, 7, 9
Others: 7
        """)
    
    uploaded = st.file_uploader("📤 Upload Lagna (Ascendant) Kundali", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Ascendant Kundali")
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
        
        # User inputs ascendant house
        st.markdown("**Which house is your Ascendant in the chart above?**")
        ascendant_house = st.number_input("Enter ascendant house number (1-12)", 
                                         min_value=1, max_value=12, value=1)
        
        if st.button("🔮 Convert to Kalapurusha", type="primary", use_container_width=True):
            with st.spinner("Reading chart and generating Kalapurusha..."):
                try:
                    # Extract planets
                    ocr = KundaliOCR()
                    ascendant_positions, _ = ocr.extract_planets_and_ascendant(image)
                    
                    st.success(f"✅ Detected {len(ascendant_positions)} planets")
                    
                    # Show detected planets
                    found = ", ".join([f"{p.planet.value}(H{p.house})" for p in ascendant_positions])
                    st.write(f"**Planets in Ascendant Chart:** {found}")
                    
                    if len(ascendant_positions) < 3:
                        st.warning("⚠️ Few planets detected. Continuing with available data...")
                    
                    # Convert to Kalapurusha
                    kalapurusha_positions = convert_to_kalapurusha(ascendant_positions, ascendant_house)
                    
                    # Draw Kalapurusha chart
                    drawer = KalapurushaChartDrawer(1800, 900)
                    chart = drawer.draw_chart(kalapurusha_positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart")
                        st.image(chart, use_column_width=True)
                    
                    st.divider()
                    st.header("📊 Vedic Analysis")
                    df = generate_analysis(kalapurusha_positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    st.header("🏥 Health Analysis")
                    health = health_analysis(kalapurusha_positions)
                    
                    cols = st.columns(3)
                    for i, (house, impacts) in enumerate(health.items()):
                        if impacts:
                            with cols[i % 3]:
                                st.markdown(f"**H{house}: {KALAPURUSHA_BODY[house]}**")
                                for impact in impacts:
                                    st.markdown(f"- {impact}")
                                st.markdown("---")
                    
                    st.divider()
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        buf = BytesIO()
                        chart.save(buf, format='PNG')
                        st.download_button("⬇️ Download Chart", buf.getvalue(),
                                         "kalapurusha.png", "image/png", use_container_width=True)
                    
                    with col_b:
                        csv = df.to_csv(index=False)
                        st.download_button("⬇️ Download Report", csv,
                                         "analysis.csv", "text/csv", use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload your ascendant kundali to generate Kalapurusha chart")

if __name__ == "__main__":
    main()
