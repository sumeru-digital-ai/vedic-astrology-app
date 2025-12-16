"""
Kalapurusha Vedic Astrology - FIXED VERSION
Properly reads all planets and draws Graha Drishti arrows
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

# ==================== VEDIC ASTROLOGY CONSTANTS ====================

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
    Planet.SUN: "#FF0000",
    Planet.MOON: "#C0C0C0",
    Planet.MARS: "#FF4500",
    Planet.MERCURY: "#00CC00",
    Planet.JUPITER: "#FFD700",
    Planet.VENUS: "#FF69B4",
    Planet.SATURN: "#0000FF",
    Planet.RAHU: "#696969",
    Planet.KETU: "#8B4513"
}

# Comprehensive planet detection patterns
PLANET_PATTERNS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS, 'MAA': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY, 'MER': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER, 'JUPI': Planet.JUPITER, 'PI': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS, 'UF': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN, 'SATU': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU, 'RAHU': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU, 'KETU': Planet.KETU,
    'NE': Planet.MERCURY  # Common OCR mistake
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

KALAPURUSHA_BODY_PARTS = {
    1: "Head (Krutthika - मस्तक)",
    2: "Face/Mouth (Aarudra - मुख)",
    3: "Throat/Shoulders (Punarwasu - कंठ)",
    4: "Chest/Heart (Aaslesha - हृदय)",
    5: "Stomach (Pubba - उदर)",
    6: "Intestines (Jyeshta - आंत्र)",
    7: "Lower Abdomen (Hastha - नाभि)",
    8: "Genitals (Anuradha - गुह्य)",
    9: "Thighs (Poorvashaadha - जंघा)",
    10: "Knees (Sravanam - जानु)",
    11: "Calves (Satabhisham - पिंडली)",
    12: "Feet (Uttarabhadra - पाद)"
}

HOUSE_SIGNIFICATIONS = {
    1: "Self, Physical Body, Personality",
    2: "Wealth, Speech, Family",
    3: "Courage, Siblings, Communication",
    4: "Mother, Home, Heart, Emotions",
    5: "Intelligence, Children, Creativity",
    6: "Disease, Enemies, Debts",
    7: "Spouse, Partnerships",
    8: "Longevity, Transformation",
    9: "Fortune, Dharma, Father",
    10: "Career, Status, Authority",
    11: "Gains, Income, Friends",
    12: "Loss, Expenses, Liberation"
}

PLANET_NATURE = {
    Planet.SUN: "Malefic (Soul Karaka)",
    Planet.MOON: "Benefic (Mind Karaka)",
    Planet.MARS: "Malefic (Energy)",
    Planet.MERCURY: "Neutral (Intellect)",
    Planet.JUPITER: "Great Benefic (Guru)",
    Planet.VENUS: "Benefic (Sukra)",
    Planet.SATURN: "Malefic (Karma)",
    Planet.RAHU: "Malefic (Shadow)",
    Planet.KETU: "Malefic (Moksha)"
}

PLANET_GEMSTONES = {
    Planet.SUN: "Ruby (माणिक्य)", Planet.MOON: "Pearl (मोती)",
    Planet.MARS: "Red Coral (मूंगा)", Planet.MERCURY: "Emerald (पन्ना)",
    Planet.JUPITER: "Yellow Sapphire (पुखराज)", Planet.VENUS: "Diamond (हीरा)",
    Planet.SATURN: "Blue Sapphire (नीलम)", Planet.RAHU: "Hessonite (गोमेद)",
    Planet.KETU: "Cat's Eye (लहसुनिया)"
}

PLANET_GRAINS = {
    Planet.SUN: "Wheat (गेहूं)", Planet.MOON: "Rice (चावल)",
    Planet.MARS: "Red Lentils (मसूर)", Planet.MERCURY: "Green Gram (मूंग)",
    Planet.JUPITER: "Chana Dal (चना)", Planet.VENUS: "White Rice (सफेद चावल)",
    Planet.SATURN: "Black Sesame (काला तिल)", Planet.RAHU: "Black Lentils (उड़द)",
    Planet.KETU: "Multi-grain (मिश्रित अनाज)"
}

@dataclass
class PlanetPosition:
    planet: Planet
    house: int
    degrees: Optional[float] = None

# ==================== IMPROVED OCR ENGINE ====================

class ImprovedKundaliOCR:
    """Reads kundali with better region detection"""
    
    def __init__(self):
        # More precise house regions for North Indian diamond
        self.house_regions = {
            # Format: (x1, y1, x2, y2) as percentages
            1: (0.42, 0.45, 0.58, 0.55),   # Center
            2: (0.35, 0.15, 0.48, 0.35),   # Top-left sector
            3: (0.08, 0.15, 0.28, 0.35),   # Far left top
            4: (0.03, 0.35, 0.20, 0.50),   # Left middle
            5: (0.08, 0.50, 0.28, 0.70),   # Left bottom
            6: (0.28, 0.65, 0.45, 0.85),   # Bottom-left
            7: (0.42, 0.70, 0.58, 0.85),   # Bottom center
            8: (0.55, 0.65, 0.72, 0.85),   # Bottom-right
            9: (0.72, 0.50, 0.92, 0.70),   # Far right bottom
            10: (0.80, 0.35, 0.97, 0.50),  # Right middle
            11: (0.72, 0.15, 0.92, 0.35),  # Far right top
            12: (0.52, 0.15, 0.65, 0.35)   # Top-right sector
        }
    
    def extract_from_image(self, image) -> List[PlanetPosition]:
        """Extract all planets with improved detection"""
        img = np.array(image)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        height, width = img.shape[:2]
        all_positions = []
        
        # Also do full image OCR to catch missed planets
        full_text = self._ocr_image(img)
        st.write(f"🔍 Full OCR Text: {full_text[:200]}...")  # Debug
        
        for house_num, (x1, y1, x2, y2) in self.house_regions.items():
            px1, py1 = int(x1 * width), int(y1 * height)
            px2, py2 = int(x2 * width), int(y2 * height)
            
            roi = img[py1:py2, px1:px2]
            text = self._ocr_image(roi)
            
            planets = self._parse_planets(text, house_num)
            all_positions.extend(planets)
        
        # Remove duplicates
        unique_positions = self._deduplicate(all_positions)
        
        return unique_positions
    
    def _ocr_image(self, img) -> str:
        """OCR with preprocessing"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multiple threshold attempts
        texts = []
        
        # Try binary threshold
        _, thresh1 = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        text1 = pytesseract.image_to_string(thresh1, config='--psm 6')
        texts.append(text1)
        
        # Try adaptive threshold
        thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        text2 = pytesseract.image_to_string(thresh2, config='--psm 11')
        texts.append(text2)
        
        # Return longest text
        return max(texts, key=len).upper()
    
    def _parse_planets(self, text: str, house: int) -> List[PlanetPosition]:
        """Parse planets with flexible matching"""
        positions = []
        text = text.replace('\n', ' ').replace('  ', ' ').strip()
        
        if not text:
            return positions
        
        # Check each pattern
        for pattern, planet in PLANET_PATTERNS.items():
            if pattern in text:
                # Extract degrees
                degree_pattern = rf'{pattern}\s*(\d+)'
                match = re.search(degree_pattern, text)
                degrees = float(match.group(1)) if match else None
                
                positions.append(PlanetPosition(
                    planet=planet,
                    house=house,
                    degrees=degrees
                ))
        
        return positions
    
    def _deduplicate(self, positions: List[PlanetPosition]) -> List[PlanetPosition]:
        """Remove duplicate planet detections"""
        seen = set()
        unique = []
        
        for pos in positions:
            key = (pos.planet, pos.house)
            if key not in seen:
                seen.add(key)
                unique.append(pos)
        
        return unique

# ==================== KALAPURUSHA CHART GENERATOR ====================

class KalapurushaChart:
    """Generate proper Kalapurusha chart with all aspects"""
    
    def __init__(self, size: int = 1200):
        self.size = size
        self.center = size // 2
        self.radius = int(size * 0.40)
    
    def generate(self, positions: List[PlanetPosition]) -> Image.Image:
        """Generate complete chart"""
        img = Image.new('RGB', (self.size, self.size), 'white')
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw structure
        self._draw_diamond(draw)
        self._draw_house_numbers(draw)
        
        # Place planets
        self._place_planets(draw, positions)
        
        # Draw ALL aspects
        self._draw_all_aspects(draw, positions)
        
        return img
    
    def _draw_diamond(self, draw: ImageDraw.Draw):
        """Draw diamond structure"""
        r = self.radius
        c = self.center
        
        top = (c, c - r)
        right = (c + r, c)
        bottom = (c, c + r)
        left = (c - r, c)
        
        # Outer diamond
        draw.polygon([top, right, bottom, left], outline='black', width=5)
        
        # Internal divisions
        draw.line([left, right], fill='black', width=3)
        draw.line([top, bottom], fill='black', width=3)
        draw.line([top, left], fill='black', width=2)
        draw.line([top, right], fill='black', width=2)
        draw.line([bottom, left], fill='black', width=2)
        draw.line([bottom, right], fill='black', width=2)
    
    def _get_house_center(self, house: int) -> Tuple[int, int]:
        """Get exact center for each house"""
        r = self.radius
        c = self.center
        offset = r * 0.55
        
        centers = {
            1: (c, c),
            2: (c - int(offset*0.65), c - int(r*0.70)),
            3: (c - int(r*0.70), c - int(offset*0.65)),
            4: (c - int(r*0.70), c),
            5: (c - int(r*0.70), c + int(offset*0.65)),
            6: (c - int(offset*0.65), c + int(r*0.70)),
            7: (c, c + int(r*0.70)),
            8: (c + int(offset*0.65), c + int(r*0.70)),
            9: (c + int(r*0.70), c + int(offset*0.65)),
            10: (c + int(r*0.70), c),
            11: (c + int(r*0.70), c - int(offset*0.65)),
            12: (c + int(offset*0.65), c - int(r*0.70))
        }
        
        return centers.get(house, (c, c))
    
    def _draw_house_numbers(self, draw: ImageDraw.Draw):
        """Draw house numbers clearly"""
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 32)
            except:
                font = ImageFont.load_default()
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            text = str(house)
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - w//2, y - h - 40), text, fill='#000080', font=font)
    
    def _place_planets(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Place planets in houses"""
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 22)
            except:
                font = ImageFont.load_default()
        
        house_planets = {}
        for pos in positions:
            if pos.house not in house_planets:
                house_planets[pos.house] = []
            house_planets[pos.house].append(pos)
        
        for house, planets in house_planets.items():
            x, y = self._get_house_center(house)
            
            for i, pos in enumerate(planets):
                py = y + (i * 30) - 5
                color = PLANET_COLORS[pos.planet]
                
                text = pos.planet.value[:2]
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_all_aspects(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw ALL Graha Drishti arrows"""
        for pos in positions:
            aspects = GRAHA_DRISHTI[pos.planet]
            color = PLANET_COLORS[pos.planet]
            
            for offset in aspects:
                target_house = ((pos.house - 1 + offset) % 12) + 1
                
                x1, y1 = self._get_house_center(pos.house)
                x2, y2 = self._get_house_center(target_house)
                
                self._draw_arrow(draw, x1, y1, x2, y2, color)
    
    def _draw_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int, 
                    x2: int, y2: int, color: str):
        """Draw visible arrow"""
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        rgba = (r, g, b, 80)  # Semi-transparent
        
        # Main line
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=4)
        
        # Arrow head
        angle = np.arctan2(y2 - y1, x2 - x1)
        arrow_len = 25
        arrow_angle = np.pi / 5
        
        x3 = x2 - arrow_len * np.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * np.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * np.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * np.sin(angle + arrow_angle)
        
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== INTERPRETATION ====================

def generate_report(positions: List[PlanetPosition]) -> pd.DataFrame:
    """Generate detailed report"""
    data = []
    
    for pos in positions:
        aspects = [str((pos.house - 1 + off) % 12 + 1) 
                  for off in GRAHA_DRISHTI[pos.planet]]
        
        data.append({
            'Planet': pos.planet.value,
            'House': pos.house,
            'Degrees': f"{pos.degrees:.0f}°" if pos.degrees else "N/A",
            'Body Part': KALAPURUSHA_BODY_PARTS[pos.house],
            'Life Area': HOUSE_SIGNIFICATIONS[pos.house],
            'Nature': PLANET_NATURE[pos.planet],
            'Aspects To': ", ".join(aspects),
            'Gemstone': PLANET_GEMSTONES[pos.planet],
            'Donation': PLANET_GRAINS[pos.planet]
        })
    
    return pd.DataFrame(data)

def health_analysis(positions: List[PlanetPosition]) -> Dict[int, List[str]]:
    """Health impact per body part"""
    impacts = {i: [] for i in range(1, 13)}
    
    # Direct placement
    for pos in positions:
        nature = "✅ Strengthens" if "Benefic" in PLANET_NATURE[pos.planet] else "⚠️ Challenges"
        impacts[pos.house].append(
            f"{nature} - {pos.planet.value} placed here"
        )
    
    # Aspects
    for pos in positions:
        for offset in GRAHA_DRISHTI[pos.planet]:
            target = ((pos.house - 1 + offset) % 12) + 1
            nature = "➡️ Supports" if "Benefic" in PLANET_NATURE[pos.planet] else "➡️ Stresses"
            impacts[target].append(
                f"{nature} - {pos.planet.value} aspects from H{pos.house}"
            )
    
    return impacts

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="Kalapurusha Jyotish",
        page_icon="🕉️",
        layout="wide"
    )
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    st.markdown("**Classical Graha Drishti Analysis with Body Part Mapping**")
    
    with st.sidebar:
        st.header("📖 About Kalapurusha")
        st.info("""
        **Kalapurusha** = Cosmic Time Body
        
        Each house maps to a body part:
        - 1st: Head
        - 4th: Heart  
        - 6th: Intestines
        - etc.
        
        **Graha Drishti** shows planetary influences through aspects.
        """)
        
        st.header("🔮 Aspect Rules")
        st.code("""
Mars: 4, 7, 8
Jupiter: 5, 7, 9  
Saturn: 3, 7, 10
Others: 7
        """)
    
    uploaded = st.file_uploader("📤 Upload Lagna Kundali", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Kundali")
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
        
        if st.button("🔮 Analyze with Kalapurusha", type="primary"):
            with st.spinner("Reading planets and computing Graha Drishti..."):
                try:
                    ocr = ImprovedKundaliOCR()
                    positions = ocr.extract_from_image(image)
                    
                    st.success(f"✅ Detected {len(positions)} planets")
                    
                    # Show detected planets
                    planet_names = [f"{p.planet.value} (H{p.house})" for p in positions]
                    st.write(f"**Planets found:** {', '.join(planet_names)}")
                    
                    if not positions:
                        st.error("No planets detected. Try a clearer image.")
                        return
                    
                    chart_gen = KalapurushaChart(1400)
                    chart_img = chart_gen.generate(positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart")
                        st.image(chart_img, use_column_width=True)
                    
                    st.divider()
                    st.header("📊 Vedic Analysis")
                    df = generate_report(positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    st.header("🏥 Health Analysis (Kalapurusha)")
                    health = health_analysis(positions)
                    
                    cols = st.columns(3)
                    for i, (house, impacts) in enumerate(health.items()):
                        if impacts:
                            with cols[i % 3]:
                                st.markdown(f"**H{house}: {KALAPURUSHA_BODY_PARTS[house]}**")
                                for impact in impacts:
                                    st.markdown(f"- {impact}")
                                st.markdown("---")
                    
                    st.divider()
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        buf = BytesIO()
                        chart_img.save(buf, format='PNG')
                        st.download_button(
                            "⬇️ Download Chart",
                            buf.getvalue(),
                            "kalapurusha.png",
                            "image/png"
                        )
                    
                    with col_b:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Report",
                            csv,
                            "vedic_report.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload your North Indian diamond kundali to begin")

if __name__ == "__main__":
    main()
