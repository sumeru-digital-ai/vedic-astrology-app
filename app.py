"""
KALAPURUSHA VEDIC ASTROLOGY SYSTEM - COMPLETE REWRITE
Correctly implements Graha Drishti with visible arrows and proper planet detection
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

# ==================== PLANET DEFINITIONS ====================

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

# Planet colors (exact as per requirements)
PLANET_COLORS = {
    Planet.SUN: (255, 0, 0),      # Red
    Planet.MOON: (255, 255, 255), # White
    Planet.MARS: (255, 0, 0),     # Red
    Planet.MERCURY: (0, 255, 0),  # Green
    Planet.JUPITER: (255, 255, 0),# Yellow
    Planet.VENUS: (255, 255, 255),# White
    Planet.SATURN: (0, 0, 255),   # Blue
    Planet.RAHU: (128, 128, 128), # Smoky
    Planet.KETU: (139, 69, 19)    # Brown
}

# ALL possible OCR variations
PLANET_PATTERNS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS, 'MAA': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY, 'MER': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER, 'JUPI': Planet.JUPITER, 'PI': Planet.JUPITER, 'PL': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS, 'UF': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN, 'SATU': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU, 'RAHU': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU, 'KETU': Planet.KETU,
    'NE': Planet.MERCURY  # Common OCR error
}

# GRAHA DRISHTI - Classical Vedic Aspects (NON-NEGOTIABLE)
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

# Kalapurusha body mapping
KALAPURUSHA_BODY = {
    1: "Head", 2: "Face/Mouth", 3: "Throat", 4: "Chest/Heart",
    5: "Stomach", 6: "Intestines", 7: "Lower Abdomen", 8: "Genitals",
    9: "Thighs", 10: "Knees", 11: "Calves", 12: "Feet"
}

HOUSE_SIGNIFICATIONS = {
    1: "Self, Physical Body", 2: "Wealth, Speech", 3: "Courage, Siblings",
    4: "Mother, Home, Heart", 5: "Intelligence, Children", 6: "Disease, Enemies",
    7: "Spouse, Partnership", 8: "Longevity, Transformation", 9: "Fortune, Dharma",
    10: "Career, Status", 11: "Gains, Income", 12: "Loss, Liberation"
}

PLANET_NATURE = {
    Planet.SUN: "Malefic", Planet.MOON: "Benefic", Planet.MARS: "Malefic",
    Planet.MERCURY: "Neutral", Planet.JUPITER: "Benefic", Planet.VENUS: "Benefic",
    Planet.SATURN: "Malefic", Planet.RAHU: "Malefic", Planet.KETU: "Malefic"
}

PLANET_GEMSTONE = {
    Planet.SUN: "Ruby", Planet.MOON: "Pearl", Planet.MARS: "Red Coral",
    Planet.MERCURY: "Emerald", Planet.JUPITER: "Yellow Sapphire", Planet.VENUS: "Diamond",
    Planet.SATURN: "Blue Sapphire", Planet.RAHU: "Hessonite", Planet.KETU: "Cat's Eye"
}

PLANET_GRAIN = {
    Planet.SUN: "Wheat", Planet.MOON: "Rice", Planet.MARS: "Red Lentils",
    Planet.MERCURY: "Green Gram", Planet.JUPITER: "Chana Dal", Planet.VENUS: "White Rice",
    Planet.SATURN: "Black Sesame", Planet.RAHU: "Black Lentils", Planet.KETU: "Multi-grain"
}

@dataclass
class PlanetPosition:
    planet: Planet
    house: int
    degrees: Optional[float] = None

# ==================== ADVANCED OCR ENGINE ====================

class VedicKundaliOCR:
    """Robust OCR for North Indian diamond charts"""
    
    def extract_planets(self, image) -> List[PlanetPosition]:
        """Extract all planets from kundali image"""
        img = np.array(image)
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        height, width = img.shape[:2]
        
        # Define precise house regions (North Indian diamond)
        house_regions = self._get_house_regions(width, height)
        
        all_positions = []
        
        # Extract from each house region
        for house, (x1, y1, x2, y2) in house_regions.items():
            roi = img[y1:y2, x1:x2]
            text = self._ocr_region(roi)
            
            planets = self._parse_planets(text, house)
            all_positions.extend(planets)
        
        # Also do full image OCR as backup
        full_text = self._ocr_region(img)
        st.write(f"**OCR Full Text:** {full_text[:300]}")
        
        # Deduplicate
        return self._deduplicate(all_positions)
    
    def _get_house_regions(self, w: int, h: int) -> Dict[int, Tuple[int, int, int, int]]:
        """Get pixel coordinates for each house"""
        return {
            1: (int(w*0.40), int(h*0.43), int(w*0.60), int(h*0.57)),  # Center
            2: (int(w*0.32), int(h*0.12), int(w*0.50), int(h*0.35)),  # Top-left
            3: (int(w*0.05), int(h*0.12), int(w*0.30), int(h*0.35)),  # Far left top
            4: (int(w*0.02), int(h*0.33), int(w*0.22), int(h*0.52)),  # Left middle
            5: (int(w*0.05), int(h*0.50), int(w*0.30), int(h*0.73)),  # Left bottom
            6: (int(w*0.25), int(h*0.65), int(w*0.48), int(h*0.88)),  # Bottom-left
            7: (int(w*0.40), int(h*0.70), int(w*0.60), int(h*0.88)),  # Bottom center
            8: (int(w*0.52), int(h*0.65), int(w*0.75), int(h*0.88)),  # Bottom-right
            9: (int(w*0.70), int(h*0.50), int(w*0.95), int(h*0.73)),  # Far right bottom
            10: (int(w*0.78), int(h*0.33), int(w*0.98), int(h*0.52)), # Right middle
            11: (int(w*0.70), int(h*0.12), int(w*0.95), int(h*0.35)), # Far right top
            12: (int(w*0.50), int(h*0.12), int(w*0.68), int(h*0.35))  # Top-right
        }
    
    def _ocr_region(self, img) -> str:
        """OCR with preprocessing"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Enhance
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Try multiple thresholds
        texts = []
        
        _, t1 = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
        texts.append(pytesseract.image_to_string(t1, config='--psm 6'))
        
        t2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        texts.append(pytesseract.image_to_string(t2, config='--psm 11'))
        
        return max(texts, key=len).upper()
    
    def _parse_planets(self, text: str, house: int) -> List[PlanetPosition]:
        """Extract planets from text"""
        positions = []
        text = text.replace('\n', ' ').replace('  ', ' ')
        
        for pattern, planet in PLANET_PATTERNS.items():
            if pattern in text:
                # Try to get degrees
                deg_match = re.search(rf'{pattern}\s*(\d+)', text)
                degrees = float(deg_match.group(1)) if deg_match else None
                
                positions.append(PlanetPosition(
                    planet=planet,
                    house=house,
                    degrees=degrees
                ))
        
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

# ==================== KALAPURUSHA CHART DRAWER ====================

class KalapurushaChartDrawer:
    """Draws complete Kalapurusha chart with ALL Graha Drishti arrows"""
    
    def __init__(self, size: int = 1600):
        self.size = size
        self.center = size // 2
        self.radius = int(size * 0.42)
    
    def draw_complete_chart(self, positions: List[PlanetPosition]) -> Image.Image:
        """Draw EVERYTHING: structure, planets, ALL aspect arrows"""
        
        # Create blank white canvas
        img = Image.new('RGB', (self.size, self.size), (255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Step 1: Draw diamond structure
        self._draw_diamond_structure(draw)
        
        # Step 2: Draw house numbers
        self._draw_house_numbers(draw)
        
        # Step 3: Place ALL planets in houses
        self._place_planets(draw, positions)
        
        # Step 4: Draw ALL Graha Drishti arrows (MOST IMPORTANT)
        self._draw_all_drishti_arrows(draw, positions)
        
        return img
    
    def _draw_diamond_structure(self, draw: ImageDraw.Draw):
        """Draw North Indian diamond"""
        c = self.center
        r = self.radius
        
        # Four corner points
        top = (c, c - r)
        right = (c + r, c)
        bottom = (c, c + r)
        left = (c - r, c)
        
        # Outer diamond
        draw.polygon([top, right, bottom, left], outline=(0, 0, 0), width=6)
        
        # Internal cross lines
        draw.line([left, right], fill=(0, 0, 0), width=4)
        draw.line([top, bottom], fill=(0, 0, 0), width=4)
        
        # Diagonal divisions
        draw.line([top, left], fill=(0, 0, 0), width=3)
        draw.line([top, right], fill=(0, 0, 0), width=3)
        draw.line([bottom, left], fill=(0, 0, 0), width=3)
        draw.line([bottom, right], fill=(0, 0, 0), width=3)
    
    def _get_house_coords(self, house: int) -> Tuple[int, int]:
        """Get center coordinates for each house"""
        c = self.center
        r = self.radius
        offset = r * 0.58
        
        coords = {
            1: (c, c),
            2: (c - int(offset*0.68), c - int(r*0.72)),
            3: (c - int(r*0.72), c - int(offset*0.68)),
            4: (c - int(r*0.72), c),
            5: (c - int(r*0.72), c + int(offset*0.68)),
            6: (c - int(offset*0.68), c + int(r*0.72)),
            7: (c, c + int(r*0.72)),
            8: (c + int(offset*0.68), c + int(r*0.72)),
            9: (c + int(r*0.72), c + int(offset*0.68)),
            10: (c + int(r*0.72), c),
            11: (c + int(r*0.72), c - int(offset*0.68)),
            12: (c + int(offset*0.68), c - int(r*0.72))
        }
        return coords.get(house, (c, c))
    
    def _draw_house_numbers(self, draw: ImageDraw.Draw):
        """Draw house numbers 1-12"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 38)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 38)
            except:
                font = ImageFont.load_default()
        
        for house in range(1, 13):
            x, y = self._get_house_coords(house)
            text = str(house)
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - w//2, y - h - 50), text, fill=(0, 0, 128), font=font)
    
    def _place_planets(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Place all planets in their houses"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 26)
            except:
                font = ImageFont.load_default()
        
        # Group planets by house
        house_planets = {}
        for pos in positions:
            if pos.house not in house_planets:
                house_planets[pos.house] = []
            house_planets[pos.house].append(pos)
        
        # Draw each house's planets
        for house, planets in house_planets.items():
            x, y = self._get_house_coords(house)
            
            for i, pos in enumerate(planets):
                py = y + (i * 35)
                color = PLANET_COLORS[pos.planet]
                
                text = pos.planet.value[:2]
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_all_drishti_arrows(self, draw: ImageDraw.Draw, positions: List[PlanetPosition]):
        """Draw ALL Graha Drishti arrows - THIS IS THE CORE FEATURE"""
        
        arrow_count = 0
        
        for pos in positions:
            # Get aspect offsets for this planet
            aspects = GRAHA_DRISHTI[pos.planet]
            color = PLANET_COLORS[pos.planet]
            
            for aspect_offset in aspects:
                # Calculate target house
                target_house = ((pos.house - 1 + aspect_offset) % 12) + 1
                
                # Get coordinates
                x1, y1 = self._get_house_coords(pos.house)
                x2, y2 = self._get_house_coords(target_house)
                
                # Draw arrow with planet's color
                self._draw_arrow(draw, x1, y1, x2, y2, color)
                arrow_count += 1
        
        st.write(f"**✅ Drew {arrow_count} Graha Drishti arrows**")
    
    def _draw_arrow(self, draw: ImageDraw.Draw, x1: int, y1: int, 
                    x2: int, y2: int, color: Tuple[int, int, int]):
        """Draw a single arrow with arrowhead"""
        
        # Make semi-transparent
        r, g, b = color
        rgba = (r, g, b, 110)
        
        # Main line
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=5)
        
        # Calculate arrow head
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 30
        arrow_angle = math.pi / 5.5
        
        # Two lines forming arrow head
        x3 = x2 - arrow_len * math.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * math.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * math.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * math.sin(angle + arrow_angle)
        
        # Draw filled triangle arrowhead
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== INTERPRETATION ====================

def generate_analysis_table(positions: List[PlanetPosition]) -> pd.DataFrame:
    """Generate complete Vedic analysis"""
    data = []
    
    for pos in positions:
        aspects_to = [str((pos.house - 1 + off) % 12 + 1) 
                     for off in GRAHA_DRISHTI[pos.planet]]
        
        data.append({
            'Planet': pos.planet.value,
            'House': pos.house,
            'Degrees': f"{int(pos.degrees)}°" if pos.degrees else "N/A",
            'Body Part': KALAPURUSHA_BODY[pos.house],
            'Life Area': HOUSE_SIGNIFICATIONS[pos.house],
            'Nature': PLANET_NATURE[pos.planet],
            'Aspects To Houses': ", ".join(aspects_to),
            'Gemstone': PLANET_GEMSTONE[pos.planet],
            'Donation Grain': PLANET_GRAIN[pos.planet]
        })
    
    return pd.DataFrame(data)

def generate_health_analysis(positions: List[PlanetPosition]) -> Dict[int, List[str]]:
    """Health impact on each body part"""
    impacts = {i: [] for i in range(1, 13)}
    
    # Direct placement
    for pos in positions:
        effect = "✅ Strengthens" if PLANET_NATURE[pos.planet] == "Benefic" else "⚠️ Challenges"
        impacts[pos.house].append(f"{effect}: {pos.planet.value} in this house")
    
    # Aspects
    for pos in positions:
        for offset in GRAHA_DRISHTI[pos.planet]:
            target = ((pos.house - 1 + offset) % 12) + 1
            effect = "➡️ Supports" if PLANET_NATURE[pos.planet] == "Benefic" else "➡️ Stresses"
            impacts[target].append(f"{effect}: {pos.planet.value} aspects from H{pos.house}")
    
    return impacts

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(page_title="Kalapurusha Jyotish", page_icon="🕉️", layout="wide")
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    st.markdown("**Classical Graha Drishti Analysis with Body Part Mapping**")
    
    with st.sidebar:
        st.header("📖 About Kalapurusha")
        st.info("""
        **Kalapurusha** = Cosmic Time Being
        
        Each house = body part:
        1-Head, 2-Face, 3-Throat, 4-Heart, 
        5-Stomach, 6-Intestines, 7-Abdomen,
        8-Genitals, 9-Thighs, 10-Knees,
        11-Calves, 12-Feet
        """)
        
        st.header("🔮 Graha Drishti")
        st.code("""
Mars: 4th, 7th, 8th
Jupiter: 5th, 7th, 9th
Saturn: 3rd, 7th, 10th
Rahu/Ketu: 5th, 7th, 9th
Others: 7th only
        """)
    
    uploaded = st.file_uploader("📤 Upload Lagna Kundali", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Kundali")
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
        
        if st.button("🔮 Analyze with Kalapurusha", type="primary", use_container_width=True):
            with st.spinner("Reading planets and computing Graha Drishti..."):
                try:
                    # OCR
                    ocr = VedicKundaliOCR()
                    positions = ocr.extract_planets(image)
                    
                    st.success(f"✅ Detected {len(positions)} planets")
                    
                    # Show what was found
                    found = ", ".join([f"{p.planet.value}(H{p.house})" for p in positions])
                    st.write(f"**Planets:** {found}")
                    
                    if len(positions) < 3:
                        st.warning("⚠️ Only detected few planets. OCR may need improvement.")
                    
                    # Generate chart
                    drawer = KalapurushaChartDrawer(1600)
                    chart = drawer.draw_complete_chart(positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart with Drishti")
                        st.image(chart, use_column_width=True)
                    
                    st.divider()
                    st.header("📊 Vedic Analysis")
                    df = generate_analysis_table(positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    st.header("🏥 Health Analysis (Kalapurusha)")
                    health = generate_health_analysis(positions)
                    
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
                        st.download_button(
                            "⬇️ Download Chart",
                            buf.getvalue(),
                            "kalapurusha_chart.png",
                            "image/png",
                            use_container_width=True
                        )
                    
                    with col_b:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Report",
                            csv,
                            "vedic_report.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload your North Indian diamond kundali to begin")
        
        st.markdown("### ✨ What This System Does")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**🔍 Auto OCR**")
            st.write("Reads all planets automatically")
        with col_b:
            st.markdown("**➡️ Graha Drishti**")
            st.write("Draws ALL aspect arrows")
        with col_c:
            st.markdown("**💊 Health Map**")
            st.write("Body part influence analysis")

if __name__ == "__main__":
    main()
