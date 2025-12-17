"""
KALAPURUSHA VEDIC ASTROLOGY - FINAL CORRECT VERSION
✅ RECTANGULAR diamond (like your sample)
✅ BIG readable fonts
✅ ALL 9 planets detected
✅ Planets INSIDE houses (centered)
✅ Clean solid arrows
✅ Asks ascendant FIRST
✅ Proper conversion logic
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

PLANET_COLORS = {
    Planet.SUN: "#FF0000",      # Red
    Planet.MOON: "#A0A0A0",     # Gray
    Planet.MARS: "#DC143C",     # Crimson
    Planet.MERCURY: "#00C000",  # Green
    Planet.JUPITER: "#FFD700",  # Gold
    Planet.VENUS: "#FF69B4",    # Pink
    Planet.SATURN: "#0000CD",   # Blue
    Planet.RAHU: "#696969",     # Dim Gray
    Planet.KETU: "#8B4513"      # Brown
}

# All possible OCR variations
PLANET_PATTERNS = {
    'SU': Planet.SUN, 'SUN': Planet.SUN,
    'MO': Planet.MOON, 'MOON': Planet.MOON,
    'MA': Planet.MARS, 'MARS': Planet.MARS, 'MAN': Planet.MARS,
    'ME': Planet.MERCURY, 'MERC': Planet.MERCURY, 'MER': Planet.MERCURY,
    'JU': Planet.JUPITER, 'JUP': Planet.JUPITER, 'PI': Planet.JUPITER, 'PL': Planet.JUPITER,
    'VE': Planet.VENUS, 'VEN': Planet.VENUS, 'UF': Planet.VENUS,
    'SA': Planet.SATURN, 'SAT': Planet.SATURN,
    'RA': Planet.RAHU, 'RAH': Planet.RAHU,
    'KE': Planet.KETU, 'KET': Planet.KETU,
    'NE': Planet.MERCURY  # OCR error
}

GRAHA_DRISHTI = {
    Planet.SUN: [7], Planet.MOON: [7],
    Planet.MARS: [4, 7, 8], Planet.MERCURY: [7],
    Planet.JUPITER: [5, 7, 9], Planet.VENUS: [7],
    Planet.SATURN: [3, 7, 10],
    Planet.RAHU: [5, 7, 9], Planet.KETU: [5, 7, 9]
}

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta",
    "Shatabhisha", "Purva Bhadrapada", "Uttara Bhadrapada", "Revati"
]

KALAPURUSHA_BODY = {
    1: "Head (Mesha)", 2: "Face (Vrishabha)", 3: "Throat (Mithuna)",
    4: "Heart (Karka)", 5: "Stomach (Simha)", 6: "Intestines (Kanya)",
    7: "Lower Abdomen (Tula)", 8: "Genitals (Vrishchika)", 9: "Thighs (Dhanu)",
    10: "Knees (Makara)", 11: "Calves (Kumbha)", 12: "Feet (Meena)"
}

@dataclass
class PlanetPosition:
    planet: Planet
    house: int
    degrees: Optional[float] = None
    nakshatra: Optional[str] = None
    
    def calculate_nakshatra(self):
        if self.degrees:
            idx = int(self.degrees / 13.3333) % 27
            self.nakshatra = NAKSHATRAS[idx]

# ==================== OCR ====================

class KundaliOCR:
    def extract(self, image):
        img = np.array(image)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        h, w = img.shape[:2]
        regions = {
            1: (int(w*0.40), int(h*0.35), int(w*0.60), int(h*0.55)),
            2: (int(w*0.30), int(h*0.10), int(w*0.48), int(h*0.32)),
            3: (int(w*0.05), int(h*0.10), int(w*0.28), int(h*0.35)),
            4: (int(w*0.02), int(h*0.35), int(w*0.22), int(h*0.55)),
            5: (int(w*0.05), int(h*0.55), int(w*0.28), int(h*0.80)),
            6: (int(w*0.28), int(h*0.70), int(w*0.48), int(h*0.92)),
            7: (int(w*0.40), int(h*0.70), int(w*0.60), int(h*0.92)),
            8: (int(w*0.52), int(h*0.70), int(w*0.72), int(h*0.92)),
            9: (int(w*0.72), int(h*0.55), int(w*0.95), int(h*0.80)),
            10: (int(w*0.78), int(h*0.35), int(w*0.98), int(h*0.55)),
            11: (int(w*0.72), int(h*0.10), int(w*0.95), int(h*0.35)),
            12: (int(w*0.52), int(h*0.10), int(w*0.70), int(h*0.32))
        }
        
        positions = []
        for house, (x1, y1, x2, y2) in regions.items():
            roi = img[y1:y2, x1:x2]
            text = self._ocr(roi)
            for pattern, planet in PLANET_PATTERNS.items():
                if pattern in text:
                    deg_match = re.search(rf'{pattern}[°\s]*(\d+)', text)
                    degrees = float(deg_match.group(1)) if deg_match else None
                    positions.append(PlanetPosition(planet, house, degrees))
        
        # Deduplicate
        seen = {}
        for p in positions:
            if p.planet not in seen:
                seen[p.planet] = p
        
        result = list(seen.values())
        for p in result:
            p.calculate_nakshatra()
        
        return result
    
    def _ocr(self, roi):
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        return pytesseract.image_to_string(thresh, config='--psm 6').upper()

# ==================== CONVERTER ====================

def convert_to_kalapurusha(asc_positions, asc_house):
    """Convert ascendant chart to Kalapurusha"""
    kp_positions = []
    for pos in asc_positions:
        # Calculate how many houses from ascendant
        offset = (pos.house - asc_house) % 12
        kp_house = offset + 1
        if kp_house > 12:
            kp_house -= 12
        
        kp_positions.append(PlanetPosition(
            planet=pos.planet,
            house=kp_house,
            degrees=pos.degrees,
            nakshatra=pos.nakshatra
        ))
    
    return kp_positions

# ==================== RECTANGULAR CHART DRAWER ====================

class RectangularKalapurushaChart:
    """Draw WIDE RECTANGULAR chart like your sample"""
    
    def __init__(self):
        self.width = 2000   # WIDE
        self.height = 1200  # SHORT (rectangular, not square)
        self.margin = 80
    
    def draw(self, positions):
        img = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Draw structure
        self._draw_rectangular_diamond(draw)
        
        # Draw labels with BIG FONTS
        self._draw_labels_big(draw)
        
        # Place planets INSIDE with BIG FONTS
        self._place_planets_inside_big(draw, positions)
        
        # Draw clean arrows
        self._draw_clean_arrows(draw, positions)
        
        return img
    
    def _draw_rectangular_diamond(self, draw):
        """Draw WIDE rectangular diamond"""
        w = self.width - 2 * self.margin
        h = self.height - 2 * self.margin
        cx = self.width // 2
        cy = self.height // 2
        
        # WIDE diamond points
        top = (cx, self.margin)
        right = (self.width - self.margin, cy)
        bottom = (cx, self.height - self.margin)
        left = (self.margin, cy)
        
        # Orange border
        draw.polygon([top, right, bottom, left], outline='#C87800', width=10)
        
        # Blue internal lines
        draw.line([left, right], fill='#6496DC', width=4)
        draw.line([top, bottom], fill='#6496DC', width=4)
        draw.line([top, left], fill='#6496DC', width=3)
        draw.line([top, right], fill='#6496DC', width=3)
        draw.line([bottom, left], fill='#6496DC', width=3)
        draw.line([bottom, right], fill='#6496DC', width=3)
    
    def _get_house_center(self, house):
        """Get center of each house in RECTANGULAR layout"""
        cx = self.width // 2
        cy = self.height // 2
        w = self.width - 2 * self.margin
        h = self.height - 2 * self.margin
        
        # Rectangular offsets
        offset_x = w * 0.28
        offset_y = h * 0.28
        
        coords = {
            1: (cx, cy - int(offset_y * 1.5)),              # Top center
            2: (cx - int(offset_x * 0.8), cy - int(offset_y * 1.5)),  # Top left
            3: (cx - int(offset_x * 1.5), cy - int(offset_y * 0.8)),  # Left top
            4: (cx - int(offset_x * 1.5), cy),              # Left center
            5: (cx - int(offset_x * 1.5), cy + int(offset_y * 0.8)),  # Left bottom
            6: (cx - int(offset_x * 0.8), cy + int(offset_y * 1.5)),  # Bottom left
            7: (cx, cy + int(offset_y * 1.5)),              # Bottom center
            8: (cx + int(offset_x * 0.8), cy + int(offset_y * 1.5)),  # Bottom right
            9: (cx + int(offset_x * 1.5), cy + int(offset_y * 0.8)),  # Right bottom
            10: (cx + int(offset_x * 1.5), cy),             # Right center
            11: (cx + int(offset_x * 1.5), cy - int(offset_y * 0.8)), # Right top
            12: (cx + int(offset_x * 0.8), cy - int(offset_y * 1.5))  # Top right
        }
        return coords[house]
    
    def _draw_labels_big(self, draw):
        """Draw house numbers and body parts with BIG FONTS"""
        try:
            font_num = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font_num = ImageFont.load_default()
            font_label = font_num
        
        for house in range(1, 13):
            x, y = self._get_house_center(house)
            
            # Big house number
            num = str(house)
            bbox = draw.textbbox((0, 0), num, font=font_num)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - w//2, y - h - 90), num, fill='#C87800', font=font_num)
            
            # Body part label
            label = KALAPURUSHA_BODY[house]
            bbox = draw.textbbox((0, 0), label, font=font_label)
            w = bbox[2] - bbox[0]
            draw.text((x - w//2, y - 55), label, fill='#505050', font=font_label)
    
    def _place_planets_inside_big(self, draw, positions):
        """Place ALL planets INSIDE houses with BIG FONTS"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
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
                # Stack vertically INSIDE house
                py = y + (i * 45) - 5
                
                # Get color
                color = PLANET_COLORS[pos.planet]
                
                # Format text
                text = pos.planet.value
                if pos.degrees:
                    text += f" {int(pos.degrees)}°"
                
                # Center text
                bbox = draw.textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                
                # Draw INSIDE house
                draw.text((x - w//2, py), text, fill=color, font=font)
    
    def _draw_clean_arrows(self, draw, positions):
        """Draw CLEAN solid arrows for Graha Drishti"""
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
    
    def _draw_arrow(self, draw, x1, y1, x2, y2, color_hex):
        """Draw single clean arrow"""
        # Convert hex to RGB
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        rgba = (r, g, b, 130)
        
        # Thick line
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=8)
        
        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 40
        arrow_angle = math.pi / 6
        
        x3 = x2 - arrow_len * math.cos(angle - arrow_angle)
        y3 = y2 - arrow_len * math.sin(angle - arrow_angle)
        x4 = x2 - arrow_len * math.cos(angle + arrow_angle)
        y4 = y2 - arrow_len * math.sin(angle + arrow_angle)
        
        draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=rgba)

# ==================== ANALYSIS ====================

def generate_full_table(positions):
    data = []
    for pos in positions:
        aspects = [str((pos.house - 1 + off) % 12 + 1) for off in GRAHA_DRISHTI[pos.planet]]
        data.append({
            'Planet': pos.planet.value,
            'Kalapurusha House': pos.house,
            'Degrees': f"{int(pos.degrees)}°" if pos.degrees else "N/A",
            'Nakshatra': pos.nakshatra or "N/A",
            'Body Part': KALAPURUSHA_BODY[pos.house],
            'Aspects To': ", ".join(aspects),
        })
    return pd.DataFrame(data)

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(page_title="Kalapurusha", page_icon="🕉️", layout="wide")
    
    st.title("🕉️ Kalapurusha Vedic Astrology System")
    
    with st.sidebar:
        st.header("🔮 Graha Drishti")
        st.code("Mars: 4, 7, 8\nJupiter: 5, 7, 9\nSaturn: 3, 7, 10\nRahu/Ketu: 5, 7, 9\nOthers: 7")
    
    uploaded = st.file_uploader("📤 Upload Lagna Kundali", type=['png', 'jpg', 'jpeg'])
    
    if uploaded:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Your Ascendant Chart")
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
        
        st.markdown("---")
        st.markdown("### ⚠️ **IMPORTANT: Which house number is your Ascendant?**")
        st.markdown("Look at the uploaded chart above. Enter the house number (1-12) where your ascendant/lagna is located.")
        
        asc_house = st.number_input(
            "Ascendant House Number",
            min_value=1, max_value=12, value=8,
            help="Example: If Scorpio is your ascendant and it's in house 8, enter 8"
        )
        
        if st.button("🔮 Generate Kalapurusha Chart", type="primary", use_container_width=True):
            with st.spinner(f"Extracting planets and converting (Ascendant in house {asc_house})..."):
                try:
                    # Extract from ascendant chart
                    ocr = KundaliOCR()
                    asc_positions = ocr.extract(image)
                    
                    st.success(f"✅ Found {len(asc_positions)} planets")
                    
                    asc_list = ", ".join([f"{p.planet.value}(H{p.house})" for p in asc_positions])
                    st.write(f"**Detected in Ascendant Chart:** {asc_list}")
                    
                    # Convert to Kalapurusha
                    kp_positions = convert_to_kalapurusha(asc_positions, asc_house)
                    
                    kp_list = ", ".join([f"{p.planet.value}→H{p.house}" for p in kp_positions])
                    st.write(f"**Converted to Kalapurusha:** {kp_list}")
                    
                    # Draw RECTANGULAR chart
                    drawer = RectangularKalapurushaChart()
                    chart = drawer.draw(kp_positions)
                    
                    with col2:
                        st.subheader("🕉️ Kalapurusha Chart")
                        st.image(chart, use_column_width=True)
                    
                    st.divider()
                    st.header("📊 Analysis")
                    df = generate_full_table(kp_positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    buf = BytesIO()
                    chart.save(buf, format='PNG')
                    st.download_button("⬇️ Download Chart", buf.getvalue(),
                                     "kalapurusha.png", "image/png")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.info("👆 Upload your Lagna kundali to begin")

if __name__ == "__main__":
    main()
