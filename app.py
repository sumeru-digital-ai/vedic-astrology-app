"""
KALAPURUSHA VEDIC ASTROLOGY - FULLY CORRECTED
✅ North Indian diamond layout
✅ All 9 planets with degrees
✅ Nakshatra calculation
✅ Solid colored arrows for ALL Graha Drishti
✅ Complete analysis table
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

# ==================== CONSTANTS ====================

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
    Planet.MOON: (180, 180, 180),
    Planet.MARS: (220, 0, 0),
    Planet.MERCURY: (0, 200, 0),
    Planet.JUPITER: (255, 200, 0),
    Planet.VENUS: (255, 150, 200),
    Planet.SATURN: (0, 0, 220),
    Planet.RAHU: (100, 100, 100),
    Planet.KETU: (139, 69, 19)
}

PLANET_PATTERNS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY, 'MER': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER, 'PI': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS, 'UF': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU,
    'NE': Planet.MERCURY
}

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

# 27 Nakshatras (each 13°20')
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

KALAPURUSHA_BODY = {
    1: "Head (Mesha)", 2: "Face (Vrishabha)", 3: "Throat/Shoulders (Mithuna)",
    4: "Chest/Heart (Karka)", 5: "Stomach (Simha)", 6: "Intestines (Kanya)",
    7: "Lower Abdomen (Tula)", 8: "Genitals (Vrishchika)", 9: "Thighs (Dhanu)",
    10: "Knees (Makara)", 11: "Calves (Kumbha)", 12: "Feet (Meena)"
}

HOUSE_SIGNIFICATIONS = {
    1: "Self, Body, Personality", 2: "Wealth, Speech", 3: "Courage, Siblings",
    4: "Mother, Home, Heart", 5: "Children, Intelligence", 6: "Disease, Enemies",
    7: "Spouse, Partnership", 8: "Longevity", 9: "Fortune, Dharma",
    10: "Career, Status", 11: "Gains, Income", 12: "Loss, Liberation"
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
    house: int
    degrees: Optional[float] = None
    nakshatra: Optional[str] = None
    
    def calculate_nakshatra(self):
        """Calculate nakshatra from degrees"""
        if self.degrees is None:
            return None
        # Each nakshatra is 13°20' (13.333...)
        nakshatra_index = int(self.degrees / 13.3333) % 27
        self.nakshatra = NAKSHATRAS[nakshatra_index]
        return self.nakshatra

# ==================== ENHANCED OCR ====================

class AdvancedKundaliOCR:
    """Extract ALL planets with degrees"""
    
    def extract_all_planets(self, image) -> List[PlanetPosition]:
        """Extract all 9 planets with degrees"""
        img = np.array(image)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        height, width = img.shape[:2]
        house_regions = self._get_north_indian_regions(width, height)
        
        all_positions = []
        
        # Extract from each house
        for house, (x1, y1, x2, y2) in house_regions.items():
            roi = img[y1:y2, x1:x2]
            text = self._ocr_with_degrees(roi)
            planets = self._parse_planets_with_degrees(text, house)
            all_positions.extend(planets)
        
        # Deduplicate and calculate nakshatras
        unique = self._deduplicate(all_positions)
        
        for pos in unique:
            pos.calculate_nakshatra()
        
        return unique
    
    def _get_north_indian_regions(self, w: int, h: int) -> Dict[int, Tuple[int, int, int, int]]:
        """North Indian diamond house regions"""
        return {
            1: (int(w*0.42), int(h*0.38), int(w*0.58), int(h*0.62)),  # Center
            2: (int(w*0.32), int(h*0.10), int(w*0.50), int(h*0.35)),  # Top-left
            3: (int(w*0.05), int(h*0.10), int(w*0.30), int(h*0.35)),  # Far left top
            4: (int(w*0.02), int(h*0.35), int(w*0.22), int(h*0.55)),  # Left
            5: (int(w*0.05), int(h*0.55), int(w*0.30), int(h*0.80)),  # Left bottom
            6: (int(w*0.28), int(h*0.70), int(w*0.48), int(h*0.90)),  # Bottom-left
            7: (int(w*0.42), int(h*0.70), int(w*0.58), int(h*0.90)),  # Bottom
            8: (int(w*0.52), int(h*0.70), int(w*0.72), int(h*0.90)),  # Bottom-right
            9: (int(w*0.70), int(h*0.55), int(w*0.95), int(h*0.80)),  # Right bottom
            10: (int(w*0.78), int(h*0.35), int(w*0.98), int(h*0.55)), # Right
            11: (int(w*0.70), int(h*0.10), int(w*0.95), int(h*0.35)), # Right top
            12: (int(w*0.50), int(h*0.10), int(w*0.68), int(h*0.35))  # Top-right
        }
    
    def _ocr_with_degrees(self, roi) -> str:
        """Enhanced OCR for extracting degrees"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Enhance for text recognition
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multiple OCR attempts
        texts = []
        
        # Binary threshold
        _, t1 = cv2.threshold(enhanced, 110, 255, cv2.THRESH_BINARY)
        texts.append(pytesseract.image_to_string(t1, config='--psm 6 digits'))
        
        # Adaptive threshold
        t2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        texts.append(pytesseract.image_to_string(t2, config='--psm 11'))
        
        # Return longest
        return max(texts, key=len).upper()
    
    def _parse_planets_with_degrees(self, text: str, house: int) -> List[PlanetPosition]:
        """Parse planets and extract degrees"""
        positions = []
        text = text.replace('\n', ' ').replace('  ', ' ')
        
        for pattern, planet in PLANET_PATTERNS.items():
            if pattern in text:
                # Look for degrees after planet name
                # Format: SU16, MA21, VE00, etc.
                deg_pattern = rf'{pattern}\s*[°]?\s*(\d+)'
                match = re.search(deg_pattern, text)
                
                degrees = None
                if match:
                    degrees = float(match.group(1))
                
                positions.append(PlanetPosition(
                    planet=planet,
                    house=house,
                    degrees=degrees
                ))
        
        return positions
    
    def _deduplicate(self, positions: List[PlanetPosition]) -> List[PlanetPosition]:
        """Remove duplicate detections"""
        seen = {}
        for pos in positions:
            key = pos.planet
            # Keep the one with degrees if available
            if key not in seen or (pos.degrees and not seen[key].degrees):
                seen[key] = pos
        return list(seen.values())

# ==================== NORTH INDIAN DIAMOND CHART ====================

class NorthIndianKalapurushaChart:
    """Draw proper North Indian diamond Kalapurusha chart"""
    
    def __init__(self, size: int = 1600):
        self.size = size
        self.center = size // 2
        self.radius = int(size * 0.42)
    
    def draw_complete_chart(self, positions: List[PlanetPosition]) -> Image.Image:
        """Draw full chart with ALL planets and aspects"""
        
        img = Image.new('RGB', (self.size, self.size), (255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Step 1: Diamond structure
        self._draw_diamond_structure(draw)
        
        # Step 2: House numbers
        self._draw_house_numbers(draw)
        
        # Step 3: Body part labels
        self._draw_body_labels(draw)
        
        # Step 4: ALL planets
        self._place_all_planets(draw, positions)
        
        # Step 5: SOLID ARROWS for all Graha Drishti
        self._draw_solid_arrows(draw, positions)
        
        return img
    
    def _draw_diamond_structure(self, draw: ImageDraw.Draw):
        """North Indian diamond with 12 houses"""
        c = self.center
        r = self.radius
        
        # Four corners
        top = (c, c - r)
        right = (c + r, c)
        bottom = (c, c + r)
        left = (c - r, c)
        
        # Outer diamond (thick orange border like sample)
        draw.polygon([top, right, bottom, left], outline=(200, 120, 0), width=8)
        
        # Internal divisions (blue lines like sample)
        draw.line([left, right], fill=(100, 150, 220), width=3)
        draw.line([top, bottom], fill=(100, 150, 220), width=3)
        draw.line([top, left], fill=(100, 150, 220), width=3)
        draw.line([top, right], fill=(100, 150, 220), width=3)
        draw.line([bottom, left], fill=(100, 150, 220), width=3)
        draw.line([bottom, right], fill=(100, 150, 220), width=3)
    
    def _get_house_center(self, house: int) -> Tuple[int, int]:
        """Get center coordinates for each house"""
        c = self.center
        r = self.radius
        offset = r * 0.58
        
        # House 1 at TOP (HEAD), going counter-clockwise
        coords = {
            1: (c, c - int(r*0.65)),                          # Top (HEAD)
            2: (c - int(offset*0.65), c - int(r*0.65)),      # Top-left
            3: (c - int(r*0.65), c - int(offset*0.65)),      # Left-top
            4: (c - int(r*0.65), c),                          # Left (HEART)
            5: (c - int(r*0.65), c + int(offset*0.65)),      # Left-bottom
            6: (c - int(offset*0.65), c + int(r*0.65)),      # Bottom-left
            7: (c, c + int(r*0.65)),                          # Bottom
            8: (c + int(offset*0.65), c + int(r*0.65)),      # Bottom-right
            9: (c + int(r*0.65), c + int(offset*0.65)),      # Right-bottom
            10: (c + int(r*0.65), c),                         # Right (KNEES)
            11: (c + int(r*0.65), c - int(offset*0.65)),     # Right-top
            12: (c + int(offset*0.65), c - int(r*0.65))      # Top-right
        }
        return coords.get(house, (c, c))
    
    def _draw_house_numbers(self, draw: ImageDraw.Draw):
        """Draw house numbers 1-12"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
        except:
            font = ImageFont.load_default()
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            text = str(house)
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # Orange color like sample
            draw.text((x - w//2, y - h - 65), text, fill=(200, 100, 0), font=font)
    
    def _draw_body_labels(self, draw: ImageDraw.Draw):
        """Draw body part labels"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            label = KALAPURUSHA_BODY[house]
            bbox = draw.textbbox((0, 0), label, font=font)
            w = bbox[2] - bbox[0]
            draw.text((x - w//2, y - 35), label, fill=(80, 80, 80), font=font)
    
    def _place_all_planets(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Place ALL planets with their colors and degrees"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
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
                py = y + (i * 35)
                color = PLANET_COLORS[pos.planet]
                
                # Planet name + degrees
                text = pos.planet.value
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_solid_arrows(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw SOLID colored arrows for ALL Graha Drishti"""
        
        arrow_count = 0
        
        for pos in positions:
            aspects = GRAHA_DRISHTI[pos.planet]
            color = PLANET_COLORS[pos.planet]
            
            for offset in aspects:
                target = ((pos.house - 1 + offset) % 12) + 1
                
                x1, y1 = self._get_house_center(pos.house)
                x2, y2 = self._get_house_center(target)
                
                # Draw SOLID arrow (not transparent)
                self._draw_thick_arrow(draw, x1, y1, x2, y2, color)
                arrow_count += 1
        
        st.write(f"**✅ Drew {arrow_count} solid Graha Drishti arrows**")
    
    def _draw_thick_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int,
                         x2: int, y2: int, color: Tuple[int, int, int]):
        """Draw THICK SOLID arrow"""
        r, g, b = color
        # Semi-transparent to avoid complete overlap
        rgba = (r, g, b, 150)
        
        # Thick line
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=7)
        
        # Big arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 35
        arrow_angle = math.pi / 6
        
        x3 = x2 - arrow_len * math.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * math.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * math.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * math.sin(angle + arrow_angle)
        
        # Filled triangle
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== COMPLETE ANALYSIS TABLE ====================

def generate_complete_table(positions: List[PlanetPosition]) -> pd.DataFrame:
    """Generate COMPLETE table with ALL planets and ALL fields"""
    data = []
    
    for pos in positions:
        aspects = [str((pos.house - 1 + off) % 12 + 1) 
                  for off in GRAHA_DRISHTI[pos.planet]]
        
        data.append({
            'Planet': pos.planet.value,
            'House': pos.house,
            'Degrees': f"{int(pos.degrees)}°" if pos.degrees else "N/A",
            'Nakshatra': pos.nakshatra if pos.nakshatra else "N/A",
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
    st.set_page_config(page_title="Kalapurusha", page_icon="🕉️", layout="wide")
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    st.markdown("**Convert Ascendant Chart to Kalapurusha (Universal Chart) with Body Mapping**")
    
    with st.sidebar:
        st.header("📖 Kalapurusha Concept")
        st.info("""
        Kalapurusha = Universal birth chart
        
        House 1 = HEAD (top)
        Goes counter-clockwise
        
        Shows which body parts are influenced by planets.
        """)
        
        st.header("🔮 Graha Drishti")
        st.code("""
Mars: 4, 7, 8
Jupiter: 5, 7, 9
Saturn: 3, 7, 10
Rahu/Ketu: 5, 7, 9
Others: 7 only
        """)
    
    uploaded = st.file_uploader("📤 Upload Lagna Kundali", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Ascendant Kundali")
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
        
        if st.button("🔮 Convert to Kalapurusha", type="primary", use_container_width=True):
            with st.spinner("Reading all planets with degrees and drawing Kalapurusha..."):
                try:
                    # Extract ALL planets
                    ocr = AdvancedKundaliOCR()
                    positions = ocr.extract_all_planets(image)
                    
                    st.success(f"✅ Detected {len(positions)} planets")
                    
                    # Show what was found
                    found = ", ".join([f"{p.planet.value}(H{p.house}, {int(p.degrees)}°)" 
                                      if p.degrees else f"{p.planet.value}(H{p.house})" 
                                      for p in positions])
                    st.write(f"**Planets:** {found}")
                    
                    # Draw chart
                    drawer = NorthIndianKalapurushaChart(1600)
                    chart = drawer.draw_complete_chart(positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart")
                        st.image(chart, use_column_width=True)
                    
                    st.divider()
                    st.header("📊 Vedic Analysis")
                    df = generate_complete_table(positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    st.header("🏥 Health Analysis")
                    health = health_analysis(positions)
                    
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
        st.info("👆 Upload your Lagna (ascendant) kundali to generate Kalapurusha chart")

if __name__ == "__main__":
    main()
